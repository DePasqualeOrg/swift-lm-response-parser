// Copyright © Anthony DePasquale

import Foundation

/// Parser for the MiniMax M2 reasoning + tool-call format.
///
/// **Reasoning convention.** MiniMax M2 chat templates inject the `<think>`
/// opener into the prompt itself, so model output begins inside reasoning
/// and ends at the first `</think>`. Everything after `</think>` is the
/// regular response text.
///
/// **Tool-call wire shape.** Tool calls use a custom XML envelope:
///
/// ```text
/// <minimax:tool_call>
///   <invoke name="get_weather">
///     <parameter name="city">Paris</parameter>
///     <parameter name="days">5</parameter>
///   </invoke>
/// </minimax:tool_call>
/// ```
///
/// Multiple `<invoke>` blocks may appear inside a single tool-call envelope.
/// Parameter values are bare strings on the wire; the parser uses the JSON
/// schema from the `tools` constructor argument to coerce values to the
/// appropriate type (e.g., `"5"` → `5` for an integer parameter). When no
/// schema is available, values stay as strings.
///
/// **Incremental streaming for long string values.** A `<parameter>` whose
/// schema (or default) treats it as a string streams its content
/// character-for-character via `function_call_arguments.delta` events, so a
/// long string parameter reaches the consumer as it arrives rather than
/// buffering until `</invoke>`. Non-string values (integers, booleans,
/// objects, arrays) buffer until `</parameter>` closes; emitting the
/// formatted JSON value at close keeps the streamed text type-correct and
/// parseable in cumulative form.
///
/// The streaming approach is the "rebuild and diff" pattern from vLLM's
/// tool-call parsers (see `glm4_moe_tool_parser.py`'s
/// `_build_args_json_so_far`): on every `process(_:)` we rebuild the
/// args-JSON-so-far for the active `<invoke>` from its inner XML and emit
/// only the suffix that hasn't been sent yet. We pick this shape over
/// SGLang's per-character state machine because it composes with the
/// cursor-and-diff pattern other parsers in this package use, makes the
/// streaming/non-streaming oracle test trivial, and centralises
/// partial-marker hold-back. We depart from vLLM in one place: vLLM emits
/// raw partial content for non-string values mid-stream, which produces
/// non-prefix-stable diffs and transiently invalid JSON; we instead
/// buffer non-string partials until `</parameter>` closes them.
struct MiniMaxM2Parser: ResponseFormatParser {
  /// Initial reasoning phase. Default is ``reasoning`` because the chat
  /// template places the `<think>` opener in the prompt – the model output
  /// itself begins inside reasoning. Pass ``normal`` only if you've
  /// already consumed `</think>` in a prior parser invocation.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let thinkEnd = "</think>"
  private static let envelopeStart = "<minimax:tool_call>"
  private static let envelopeEnd = "</minimax:tool_call>"
  private static let invokeStart = "<invoke"
  private static let invokeEnd = "</invoke>"
  private static let parameterStart = "<parameter"
  private static let parameterEnd = "</parameter>"

  private var buffer: String = ""

  private enum Phase { case reasoning, normal }
  private var phase: Phase

  private var sentReasoningIdx: Int = 0
  /// Cursor over the buffer for non-tool-call content emission and
  /// envelope/invoke advancement. Mirrors vLLM's `_sent_content_idx`.
  private var sentContentIdx: Int = 0

  /// True between `<minimax:tool_call>` and `</minimax:tool_call>`. The
  /// envelope itself carries no payload; it brackets one or more
  /// `<invoke>` blocks.
  private var insideEnvelope: Bool = false

  private var openReasoning: OpenReasoning?
  private var openMessage: OpenMessage?

  /// One slot per `<invoke>` seen so far, complete or in-flight. Indexed
  /// in encounter order. Mirrors vLLM's `streamed_args_for_tool`.
  private var toolCalls: [OpenToolCall] = []

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  /// Function name → parameter name → list of candidate JSON schema
  /// types, indexed at construction. Mirrors sglang's
  /// `_get_param_types_from_config`: for `anyOf: [string, integer]` the
  /// list is `["string", "integer"]`. The coercion path tries each
  /// candidate in `integer > number > boolean > object > array > string`
  /// priority so a numeric string serializes as a number when both
  /// integer and string are valid.
  private let argumentSchemas: [String: [String: [String]]]

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenToolCall {
    var id: String
    var callId: String
    var outputIndex: Int
    var name: String
    /// Offset into `buffer` where the `<invoke ...>` open tag's body
    /// starts (one past the closing `>`). The args-text we feed into
    /// `buildInvokeArgsJsonSoFar` runs from here to either `</invoke>`
    /// or the safe end of the buffer.
    var bodyStartOffset: Int
    /// The args-JSON-so-far text we've already emitted as deltas.
    var streamedArgs: String = ""
    var closed: Bool = false
  }

  /// When true, the alternate chat template (which omits the `<think>`
  /// opener from the prompt) is in use. The parser skips reasoning
  /// extraction entirely and prepends the literal `"<think>"` to the
  /// first message delta it emits, mirroring vLLM's
  /// `MiniMaxM2AppendThinkReasoningParser`. With this mode on, the
  /// `initialState` argument is ignored – the parser starts in normal
  /// phase regardless.
  private let appendThink: Bool

  /// Set when ``appendThink`` is on and the literal `"<think>"` hasn't
  /// yet been prepended to a message. Cleared on the first message
  /// delta emission.
  private var pendingThinkPrepend: Bool

  init(
    initialState: InitialState = .reasoning,
    tools: [ToolSpec] = [],
    appendThink: Bool = false,
  ) {
    if appendThink {
      // Skip reasoning extraction entirely – the appendThink mode
      // collapses reasoning into content with a literal `<think>`
      // marker prepended.
      phase = .normal
    } else {
      switch initialState {
        case .normal: phase = .normal
        case .reasoning: phase = .reasoning
      }
    }
    self.appendThink = appendThink
    pendingThinkPrepend = appendThink
    argumentSchemas = MiniMaxM2Parser.buildSchemaTable(from: tools)
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return scan(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = scan(isEnd: true)
    if openReasoning != nil {
      events.append(contentsOf: closeReasoning(status: .incomplete))
    }
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    for index in toolCalls.indices where !toolCalls[index].closed {
      events.append(contentsOf: closeToolCall(at: index, status: .incomplete))
    }
    return events
  }

  private static func buildSchemaTable(from tools: [ToolSpec]) -> [String: [String: [String]]] {
    var table: [String: [String: [String]]] = [:]
    for tool in tools {
      guard let function = tool["function"] as? [String: Any],
            let name = function["name"] as? String,
            let parameters = function["parameters"] as? [String: Any],
            let properties = parameters["properties"] as? [String: Any]
      else { continue }

      var paramTypes: [String: [String]] = [:]
      for (paramName, schema) in properties {
        paramTypes[paramName] = extractTypesFromJsonSchema(schema)
      }
      if !paramTypes.isEmpty {
        table[name] = paramTypes
      }
    }
    return table
  }

  // MARK: Scan loop

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var lastPhase: Phase? = nil
    while lastPhase != phase {
      lastPhase = phase
      switch phase {
        case .reasoning:
          events.append(contentsOf: scanReasoning(isEnd: isEnd))
        case .normal:
          events.append(contentsOf: scanNormal(isEnd: isEnd))
      }
    }
    return events
  }

  // MARK: Reasoning phase

  private mutating func scanReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)
    let endChars = Array(MiniMaxM2Parser.thinkEnd)

    let endIdx = bufChars.firstIndexOf(substring: MiniMaxM2Parser.thinkEnd, after: sentReasoningIdx)

    let safeEnd: Int
    if let endIdx {
      safeEnd = endIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: endChars)
      safeEnd = bufChars.count - overlap
    }

    if safeEnd > sentReasoningIdx {
      let chunk = String(bufChars[sentReasoningIdx ..< safeEnd])
      if !chunk.isEmpty {
        if openReasoning == nil {
          events.append(contentsOf: openReasoningItem())
        }
        if var r = openReasoning {
          r.emittedText += chunk
          openReasoning = r
          events.append(.reasoningDelta(.init(
            itemId: r.id,
            outputIndex: r.outputIndex,
            contentIndex: 0,
            delta: chunk,
            sequenceNumber: takeSequence(),
          )))
        }
      }
      sentReasoningIdx = safeEnd
    }

    guard let endIdx, endIdx == safeEnd else { return events }

    sentContentIdx = endIdx + endChars.count
    events.append(contentsOf: closeReasoning(status: .completed))
    phase = .normal
    return events
  }

  // MARK: Normal phase

  private mutating func scanNormal(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)

    // The loop drives forward through a small state machine over three
    // positions: outside any envelope, inside the envelope but between
    // invokes, and inside an open invoke (streaming its args).
    while sentContentIdx < bufChars.count || hasOpenInvoke() {
      // (1) An invoke is in-flight: re-process it, emit any new
      //     args fragment, close it if `</invoke>` has arrived.
      if let activeIndex = openInvokeIndex() {
        let bodyStart = toolCalls[activeIndex].bodyStartOffset
        let endIdx = bufChars.firstIndexOf(
          substring: MiniMaxM2Parser.invokeEnd, after: bodyStart,
        )
        let isComplete: Bool
        let isTruncated: Bool
        let bodyEnd: Int
        if let endIdx {
          isComplete = true
          isTruncated = false
          bodyEnd = endIdx
        } else if isEnd {
          // EOS without `</invoke>`. Treat the invoke as a
          // logical close so partial values route through
          // `coerceParameter`; the `.incomplete` status is set
          // at finalize.
          isComplete = true
          isTruncated = true
          bodyEnd = bufChars.count
        } else {
          isComplete = false
          isTruncated = false
          let inner = Array(bufChars[bodyStart ..< bufChars.count])
          let overlap = partialOverlap(
            suffixOf: inner, with: Array(MiniMaxM2Parser.invokeEnd),
          )
          bodyEnd = bufChars.count - overlap
        }
        let innerText = String(bufChars[bodyStart ..< bodyEnd])
        events.append(contentsOf: processInvoke(
          index: activeIndex, innerText: innerText,
          isComplete: isComplete, isTruncated: isTruncated,
        ))
        if isComplete, !isTruncated {
          sentContentIdx = endIdx! + MiniMaxM2Parser.invokeEnd.count
          continue
        } else {
          break
        }
      }

      // (2) Outside any envelope: emit content up to the next
      //     `<minimax:tool_call>`, then enter the envelope.
      if !insideEnvelope {
        let nextStart = bufChars.firstIndexOf(
          substring: MiniMaxM2Parser.envelopeStart, after: sentContentIdx,
        )
        let contentEnd: Int
        if let nextStart {
          contentEnd = nextStart
        } else if isEnd {
          contentEnd = bufChars.count
        } else {
          let overlap = partialOverlap(
            suffixOf: bufChars,
            with: Array(MiniMaxM2Parser.envelopeStart),
          )
          contentEnd = bufChars.count - overlap
        }
        if contentEnd > sentContentIdx {
          events.append(contentsOf: emitContent(
            from: sentContentIdx, to: contentEnd, chars: bufChars,
          ))
          sentContentIdx = contentEnd
        }
        guard let nextStart, nextStart == sentContentIdx else { break }
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        sentContentIdx = nextStart + MiniMaxM2Parser.envelopeStart.count
        insideEnvelope = true
        continue
      }

      // (3) Inside the envelope, no open invoke. Skip whitespace /
      //     unrecognised chars to the next `<invoke>` or
      //     `</minimax:tool_call>` (whichever comes first).
      let nextInvoke = bufChars.firstIndexOf(
        substring: MiniMaxM2Parser.invokeStart, after: sentContentIdx,
      )
      let nextEnvEnd = bufChars.firstIndexOf(
        substring: MiniMaxM2Parser.envelopeEnd, after: sentContentIdx,
      )

      let firstBoundary: (offset: Int, isInvoke: Bool)? = switch (nextInvoke, nextEnvEnd) {
        case let (.some(i), .some(e)) where i < e:
          (i, true)
        case (.some, .some):
          (nextEnvEnd!, false)
        case let (.some(i), .none):
          (i, true)
        case let (.none, .some(e)):
          (e, false)
        case (.none, .none):
          nil
      }

      guard let boundary = firstBoundary else { break }
      sentContentIdx = boundary.offset

      if !boundary.isInvoke {
        sentContentIdx += MiniMaxM2Parser.envelopeEnd.count
        insideEnvelope = false
        continue
      }

      // We're at a `<invoke...`. Find the closing `>` of the open
      // tag. The name attribute lives between the prefix and the `>`.
      let openTagStart = boundary.offset + MiniMaxM2Parser.invokeStart.count
      guard let openTagEnd = firstIndexOf(
        character: ">", in: bufChars, after: openTagStart,
      ) else {
        // Open tag still incomplete; wait for more bytes.
        break
      }
      let openTagAttrs = String(bufChars[openTagStart ..< openTagEnd])
      guard let invokeName = extractAttribute(named: "name", from: openTagAttrs) else {
        // Malformed `<invoke ...>` without a name attribute. Skip
        // past the open tag and continue scanning.
        sentContentIdx = openTagEnd + 1
        continue
      }

      let outputIndex = takeOutputIndex()
      let id = IDFactory.make(.functionCall)
      let callId = IDFactory.make(.callId)
      let bodyStart = openTagEnd + 1
      toolCalls.append(OpenToolCall(
        id: id,
        callId: callId,
        outputIndex: outputIndex,
        name: invokeName,
        bodyStartOffset: bodyStart,
      ))
      events.append(.outputItemAdded(.init(
        item: .functionCall(.init(
          id: id, callId: callId, name: invokeName,
          arguments: "", status: .inProgress,
        )),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )))
      sentContentIdx = bodyStart
      // Loop iteration (1) will pick up this open invoke.
      continue
    }

    return events
  }

  private func hasOpenInvoke() -> Bool {
    for tc in toolCalls where !tc.closed {
      return true
    }
    return false
  }

  private func openInvokeIndex() -> Int? {
    for (i, tc) in toolCalls.enumerated() where !tc.closed {
      return i
    }
    return nil
  }

  private func firstIndexOf(character: Character, in chars: [Character], after: Int) -> Int? {
    var i = after
    while i < chars.count {
      if chars[i] == character { return i }
      i += 1
    }
    return nil
  }

  private mutating func emitContent(from: Int, to end: Int, chars: [Character]) -> [ResponseStreamingEvent] {
    var chunk = String(chars[from ..< end])
    if chunk.isEmpty { return [] }
    // appendThink mode: prepend the literal `<think>` to the first
    // message chunk. Mirrors vLLM's
    // `MiniMaxM2AppendThinkReasoningParser`, which puts the marker
    // back into the content because the alternate chat template
    // doesn't inject it into the prompt.
    if pendingThinkPrepend {
      chunk = "<think>" + chunk
      pendingThinkPrepend = false
    }
    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem())
    }
    if var msg = openMessage {
      msg.emittedText += chunk
      openMessage = msg
      events.append(.outputTextDelta(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        delta: chunk,
        sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  // MARK: Per-invoke streaming

  private mutating func processInvoke(
    index: Int, innerText: String, isComplete: Bool, isTruncated: Bool = false,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let name = toolCalls[index].name
    let argsSoFar = buildInvokeArgsJsonSoFar(
      toolName: name, innerText: innerText, isComplete: isComplete,
    )
    if argsSoFar.count > toolCalls[index].streamedArgs.count {
      let diffStart = argsSoFar.index(
        argsSoFar.startIndex, offsetBy: toolCalls[index].streamedArgs.count,
      )
      let diff = String(argsSoFar[diffStart...])
      toolCalls[index].streamedArgs = argsSoFar
      events.append(.functionCallArgumentsDelta(.init(
        itemId: toolCalls[index].id,
        outputIndex: toolCalls[index].outputIndex,
        delta: diff,
        sequenceNumber: takeSequence(),
      )))
    }
    if isComplete, !isTruncated, !toolCalls[index].closed {
      events.append(contentsOf: closeToolCall(at: index, status: .completed))
    }
    return events
  }

  /// Build the JSON arguments string the consumer should see for the
  /// active `<invoke>` at this point in the stream. Closing `}` is
  /// appended only when `isComplete` (i.e., `</invoke>` has arrived).
  ///
  /// Mirrors vLLM's `_build_args_json_so_far` (in `glm4_moe_tool_parser.py`)
  /// adapted to MiniMax's `<parameter name="K">VALUE</parameter>` shape.
  /// Departure from vLLM: when a non-string `<parameter>` is open but
  /// not yet closed, we skip the partial entirely so the value emits as
  /// one piece at `</parameter>` close, formatted via the schema's
  /// coercion rules – this avoids the non-prefix-stable diff pitfall in
  /// vLLM's raw-emit-then-reformat path.
  private func buildInvokeArgsJsonSoFar(
    toolName: String, innerText: String, isComplete: Bool,
  ) -> String {
    var parts: [String] = []

    // Walk all complete `<parameter ...>VALUE</parameter>` blocks.
    var cursor = innerText.startIndex
    var lastConsumedEnd = innerText.startIndex
    while let openRange = innerText.range(of: MiniMaxM2Parser.parameterStart, range: cursor ..< innerText.endIndex) {
      guard let openTagEnd = innerText.range(of: ">", range: openRange.upperBound ..< innerText.endIndex) else {
        break
      }
      guard let closeRange = innerText.range(of: MiniMaxM2Parser.parameterEnd, range: openTagEnd.upperBound ..< innerText.endIndex) else {
        break
      }
      let openTagAttrs = String(innerText[openRange.upperBound ..< openTagEnd.lowerBound])
      cursor = closeRange.upperBound
      lastConsumedEnd = closeRange.upperBound
      guard let key = extractAttribute(named: "name", from: openTagAttrs) else {
        continue
      }
      let rawValue = String(innerText[openTagEnd.upperBound ..< closeRange.lowerBound])
      // Mirror vLLM's `param_value.strip()` (in
      // `minimax_m2_tool_parser.py`): leading and trailing whitespace
      // around `<parameter>VALUE</parameter>` is template noise, not
      // value content. Stripping unconditionally aligns the
      // cumulative output with vLLM's oracle for both string and
      // non-string types.
      let trimmedValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
      let coerced = coerceParameter(funcName: toolName, paramName: key, rawValue: trimmedValue)
      parts.append(jsonEncodeString(key) + ": " + coerced)
    }

    // Detect a partial parameter: an open `<parameter ...>` after the
    // last consumed close, with no matching `</parameter>` yet.
    let tailRange = lastConsumedEnd ..< innerText.endIndex
    if let lastParamOpen = innerText.range(of: MiniMaxM2Parser.parameterStart, range: tailRange),
       let openTagEnd = innerText.range(of: ">", range: lastParamOpen.upperBound ..< innerText.endIndex),
       innerText.range(of: MiniMaxM2Parser.parameterEnd, range: openTagEnd.upperBound ..< innerText.endIndex) == nil
    {
      let openTagAttrs = String(innerText[lastParamOpen.upperBound ..< openTagEnd.lowerBound])
      if let partialKey = extractAttribute(named: "name", from: openTagAttrs) {
        var partialContent = String(innerText[openTagEnd.upperBound...])
        let overlap = partialOverlap(
          suffixOf: Array(partialContent),
          with: Array(MiniMaxM2Parser.parameterEnd),
        )
        if overlap > 0 {
          partialContent.removeLast(overlap)
        }

        if isComplete {
          // `</invoke>` arrived without `</parameter>`. Treat the
          // partial as the final value so the diff naturally
          // closes any open quotes. Strip outer whitespace to
          // mirror vLLM's `param_value.strip()`.
          let trimmed = partialContent.trimmingCharacters(in: .whitespacesAndNewlines)
          let coerced = coerceParameter(
            funcName: toolName, paramName: partialKey, rawValue: trimmed,
          )
          parts.append(jsonEncodeString(partialKey) + ": " + coerced)
        } else if isStreamableAsString(funcName: toolName, paramName: partialKey) {
          // Stream as an open-quoted string; the closing `"`
          // emits when `</parameter>` makes the pair complete.
          // Strip leading and trailing whitespace so the
          // cumulative emit matches vLLM's stripped value at
          // close. The trim is prefix-stable: trailing
          // whitespace held back this scan becomes interior the
          // moment non-whitespace follows, so the next delta
          // simply emits the (previously held whitespace + new
          // content) chunk. Whitespace that turns out to be
          // genuine trailing – i.e. immediately followed by
          // `</parameter>` – is dropped at close, matching
          // vLLM's strip.
          let trimmed = partialContent.trimmingCharacters(in: .whitespacesAndNewlines)
          let escaped = jsonEscapeStringContent(trimmed)
          parts.append(jsonEncodeString(partialKey) + ": \"" + escaped)
        }
        // Non-string partial: skip – emit nothing for this pair
        // until `</parameter>` closes it.
      }
    }

    if parts.isEmpty {
      return isComplete ? "{}" : ""
    }
    var joined = "{" + parts.joined(separator: ", ")
    if isComplete {
      joined += "}"
    }
    return joined
  }

  /// True when a string-typed parameter should stream incrementally as
  /// a JSON string. False if any candidate type would require buffering
  /// to coerce (numeric / boolean / object / array) and **also** false
  /// for unschema'd parameters – vLLM's
  /// `_convert_param_value_with_types` final fallback tries `json.loads`
  /// before defaulting to a string, so unschema'd values must buffer to
  /// the close to attempt the JSON parse. Mixed schemas like
  /// `anyOf: [string, integer]` therefore buffer.
  private func isStreamableAsString(funcName: String, paramName: String) -> Bool {
    guard let table = argumentSchemas[funcName], let types = table[paramName] else {
      return false
    }
    return types.allSatisfy { isStringAlias($0) }
  }

  private func isStringAlias(_ type: String) -> Bool {
    let t = type.lowercased()
    return t == "string" || t == "str" || t == "text" || t == "varchar" || t == "char" || t == "enum"
  }

  /// JSON-escape the *contents* of a string (between but not including
  /// the surrounding quotes). Returns `""` for empty input.
  private func jsonEscapeStringContent(_ s: String) -> String {
    if s.isEmpty { return "" }
    let encoded = jsonEncodeString(s)
    guard encoded.count >= 2 else { return "" }
    return String(encoded.dropFirst().dropLast())
  }

  private func extractAttribute(named attr: String, from tagAttrs: String) -> String? {
    // Match `name="value"` or `name='value'` (case-insensitive attr key).
    let patterns = [
      #"\#(attr)\s*=\s*"([^"]*)""#,
      #"\#(attr)\s*=\s*'([^']*)'"#,
    ]
    for pattern in patterns {
      guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
        continue
      }
      let range = NSRange(tagAttrs.startIndex..., in: tagAttrs)
      if let match = regex.firstMatch(in: tagAttrs, range: range), match.numberOfRanges >= 2 {
        if let r = Range(match.range(at: 1), in: tagAttrs) {
          return String(tagAttrs[r])
        }
      }
    }
    return nil
  }

  // MARK: Tool-call close

  private mutating func closeToolCall(at index: Int, status: ItemStatus) -> [ResponseStreamingEvent] {
    guard !toolCalls[index].closed else { return [] }
    toolCalls[index].closed = true
    let argsText = toolCalls[index].streamedArgs
    let doneItem = ResponseFunctionToolCall(
      id: toolCalls[index].id,
      callId: toolCalls[index].callId,
      name: toolCalls[index].name,
      arguments: argsText,
      status: status,
    )
    return [
      .functionCallArgumentsDone(.init(
        itemId: toolCalls[index].id,
        outputIndex: toolCalls[index].outputIndex,
        arguments: argsText,
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .functionCall(doneItem),
        outputIndex: toolCalls[index].outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  // MARK: Item open/close

  private mutating func openReasoningItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.reasoning)
    let outputIndex = takeOutputIndex()
    openReasoning = OpenReasoning(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .reasoning(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id,
        outputIndex: outputIndex,
        contentIndex: 0,
        part: .reasoningText(.init(text: "")),
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeReasoning(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let r = openReasoning else { return [] }
    openReasoning = nil
    let part = ReasoningTextContent(text: r.emittedText)
    return [
      .reasoningDone(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        text: r.emittedText,
        sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        part: .reasoningText(part),
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .reasoning(.init(
          id: r.id,
          content: [.reasoningText(part)],
          status: status,
        )),
        outputIndex: r.outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func openMessageItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.message)
    let outputIndex = takeOutputIndex()
    openMessage = OpenMessage(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .message(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id,
        outputIndex: outputIndex,
        contentIndex: 0,
        part: .outputText(.init(text: "")),
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeMessage(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let msg = openMessage else { return [] }
    openMessage = nil
    let part = ResponseOutputText(text: msg.emittedText)
    return [
      .outputTextDone(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        text: msg.emittedText,
        sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        part: .outputText(part),
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .message(.init(
          id: msg.id,
          content: [.outputText(part)],
          status: status,
        )),
        outputIndex: msg.outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  // MARK: Coercion

  private func coerceParameter(funcName: String, paramName: String, rawValue: String) -> String {
    // Defensive trim: callers should already strip but mirroring
    // vLLM's `value.lower() in (...)` post-strip check costs nothing
    // and keeps the null sentinel resilient to direct callers.
    let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    let lowered = trimmed.lowercased()
    if lowered == "null" || lowered == "none" || lowered == "nil" {
      return "null"
    }
    if let table = argumentSchemas[funcName], let types = table[paramName] {
      return coerce(rawValue: trimmed, schemaTypes: types)
    }
    // No schema entry: vLLM and SGLang both default to `["string"]`
    // in `_get_param_types_from_config`, so the value is emitted as a
    // JSON string regardless of whether it parses as JSON. A bare `5`
    // becomes `"5"`, `true` becomes `"true"`. Pinned by vLLM's
    // `test_header_and_params_in_separate_chunks` which asserts
    // `"days": "5"` for an unschema'd numeric parameter.
    return jsonEncodeString(trimmed)
  }

  /// Try each candidate schema type in `integer > number > boolean >
  /// object > array > string` priority order, falling through when the
  /// raw value can't be parsed as the candidate's type. Mirrors sglang's
  /// `_convert_param_value_with_types`.
  private func coerce(rawValue: String, schemaTypes: [String]) -> String {
    let normalized = Set(schemaTypes.map { $0.lowercased() })

    for candidate in MiniMaxM2Parser.typePriority {
      guard normalized.contains(candidate) || normalized.contains(where: { hasPrefix(candidate, in: $0) }) else {
        continue
      }
      if let coerced = tryCoerce(rawValue: rawValue, candidate: candidate) {
        return coerced
      }
    }
    // Fallback path mirrors sglang's `_convert_param_value_with_types`
    // tail: `try: return json.loads(value) except: return value`. A bare
    // `42` becomes the integer `42`, `[1,2]` stays as an array, and
    // unparseable tokens like `Beijing` fall back to a JSON string.
    let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    if !trimmed.isEmpty, isValidJSON(trimmed) {
      return trimmed
    }
    return jsonEncodeString(rawValue)
  }

  /// Match candidate `int` against schema-emitted aliases like `int32`,
  /// `uint`, `long`, etc. (sglang's qwen3 detector accepts these spellings;
  /// MiniMax schemas tend to be cleaner, but covering them here costs
  /// nothing.)
  private func hasPrefix(_ candidate: String, in normalized: String) -> Bool {
    switch candidate {
      case "integer":
        normalized.hasPrefix("int") || normalized.hasPrefix("uint") || normalized.hasPrefix("long")
      case "number":
        normalized.hasPrefix("num") || normalized.hasPrefix("float") || normalized == "double"
      default:
        false
    }
  }

  private func tryCoerce(rawValue: String, candidate: String) -> String? {
    switch candidate {
      case "integer":
        if let i = Int(rawValue) { return String(i) }
        return nil
      case "number":
        if let d = Double(rawValue) {
          if d.truncatingRemainder(dividingBy: 1) == 0,
             !rawValue.contains("."), !rawValue.lowercased().contains("e")
          {
            if let i = Int(exactly: d) { return String(i) }
            return rawValue
          }
          return rawValue
        }
        return nil
      case "boolean":
        switch rawValue.lowercased() {
          case "true", "1", "yes", "on": return "true"
          case "false", "0", "no", "off": return "false"
          default: return nil
        }
      case "object", "array":
        if isValidJSON(rawValue) { return rawValue }
        return nil
      case "string":
        return jsonEncodeString(rawValue)
      default:
        return nil
    }
  }

  private static let typePriority = [
    "integer",
    "number",
    "boolean",
    "object",
    "array",
    "string",
  ]

  private func jsonEncodeString(_ s: String) -> String {
    guard let data = try? JSONSerialization.data(
      withJSONObject: [s], options: [.withoutEscapingSlashes],
    ) else {
      return "\"\(s)\""
    }
    if let str = String(data: data, encoding: .utf8) {
      return String(str.dropFirst().dropLast())
    }
    return "\"\(s)\""
  }

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
