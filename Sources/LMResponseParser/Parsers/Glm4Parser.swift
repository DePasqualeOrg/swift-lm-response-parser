// Copyright © Anthony DePasquale

import Foundation

/// Parser for the GLM 4.x tool-call format.
///
/// **Wire shape.** GLM 4 emits tool calls as a custom XML envelope where
/// the function name appears on the first line and parameter pairs are
/// expressed as `<arg_key>` / `<arg_value>` tags:
///
/// ```text
/// <tool_call>get_weather
/// <arg_key>city</arg_key>
/// <arg_value>Beijing</arg_value>
/// <arg_key>date</arg_key>
/// <arg_value>2024-06-27</arg_value>
/// </tool_call>
/// ```
///
/// **Schema-driven coercion.** All values are bare strings on the wire.
/// The parser uses the JSON schema from the `tools` constructor argument
/// to coerce values to integers, numbers, booleans, objects, and arrays
/// where the schema declares those types. Without a matching schema entry,
/// values stay as strings.
///
/// **Incremental streaming for long string values.** A `<arg_value>` whose
/// schema (or default) treats it as a string streams its content
/// character-for-character via `function_call_arguments.delta` events, so a
/// 4000-character code block reaches the consumer as it arrives rather
/// than buffering until `</tool_call>`. Non-string values (integers,
/// booleans, objects, arrays) buffer until `</arg_value>` closes; emitting
/// the formatted JSON value at close keeps the streamed text type-correct
/// and parseable in cumulative form.
///
/// The streaming approach is the "rebuild and diff" pattern from vLLM's
/// `glm4_moe_tool_parser.py` (introduced in vLLM Issue #32829 to fix the
/// long-content latency problem): on every `process(_:)` we rebuild the
/// args-JSON-so-far for each tool-call region from the inner XML and
/// emit only the suffix that hasn't been sent yet. SGLang's
/// `glm4_moe_detector.py` solves the same problem with a five-state
/// machine and a per-character cursor; we picked the vLLM shape because
/// it composes with the cursor-and-diff pattern other parsers in this
/// package already use, makes the streaming/non-streaming oracle test
/// trivial (concatenated streamed deltas equal the one-shot output by
/// construction), and centralises partial-marker hold-back in a single
/// helper rather than dispersing it across states. We borrow one idea
/// from SGLang – caching the value type once at `<arg_value>` open
/// rather than re-looking-it-up per chunk – and depart from vLLM in one
/// place: vLLM emits raw partial content for non-string values mid-stream
/// (its `_build_args_json_so_far` line where the comment reads "Non-string
/// partial: include raw content, no wrapping"), which produces transiently
/// invalid JSON in the cumulative stream and is a non-prefix-stable diff
/// in the unschema'd-string case; we instead buffer non-string partials
/// until `</arg_value>` closes them, trading a small loss of granularity
/// for type-correctness.
///
/// **Note on tag name.** GLM 4's `<tool_call>` opening tag overlaps with
/// the Hermes format. Dispatch is gated on the model name (or model_type),
/// so a Hermes model is never routed here.
///
/// **Optional reasoning preamble.** When constructed with
/// ``acceptThink`` true, a leading `<think>...</think>` block is
/// extracted as a reasoning item before the tool-call scan. Mirrors
/// vLLM's `DeepSeekV3ReasoningWithThinkingParser` (the reasoning
/// parser registered for `glm45`), which delegates to the R1 reasoning
/// shape and defaults thinking on. The `<think>` opener is optional –
/// chat templates often inject it into the prompt, in which case the
/// model emits only `</think>` to close.
struct Glm4Parser: ResponseFormatParser {
  /// Initial reasoning phase. Used by continuation requests on
  /// thinking-enabled checkpoints (GLM 4.5+) whose `priorOutput`
  /// ended either inside or after the `<think>...</think>` block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let toolCallStart = "<tool_call>"
  private static let toolCallEnd = "</tool_call>"
  private static let argKeyOpen = "<arg_key>"
  private static let argKeyClose = "</arg_key>"
  private static let argValueOpen = "<arg_value>"
  private static let argValueClose = "</arg_value>"

  // Active accumulated output. Consumed prefixes are pruned after each scan.
  private var buffer: String = ""

  /// Cursor over the buffer for non-tool-call content emission. Advances
  /// past whatever has already been emitted to the consumer (either as
  /// message text or as a completed tool-call region). Mirrors vLLM's
  /// `_sent_content_idx`.
  private var sentContentIdx: Int = 0

  private var openMessage: OpenMessage?

  /// One slot per `<tool_call>` seen so far, complete or in-flight.
  /// Indexed in encounter order. Mirrors vLLM's `streamed_args_for_tool`
  /// and `prev_tool_call_arr`.
  private var toolCalls: [OpenToolCall] = []

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private let argumentSchemas: [String: [String: String]]

  private let acceptThink: Bool
  private var thinkPreamble: ThinkPreambleExtractor

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenToolCall {
    var id: String
    var callId: String
    var outputIndex: Int
    var name: String?
    /// The args-JSON-so-far text we've already emitted as deltas. The
    /// next emit is the suffix of the freshly rebuilt args-JSON beyond
    /// this. vLLM's `streamed_args_for_tool[i]`.
    var streamedArgs: String = ""
    var closed: Bool = false
  }

  /// - Parameters:
  ///   - tools: Tool specs used to coerce raw `<arg_value>` strings
  ///     into typed JSON values via the parameter schema.
  ///   - acceptThink: When true, scans for a leading `<think>...</think>`
  ///     reasoning preamble before the tool-call body. Mirrors vLLM's
  ///     `DeepSeekV3ReasoningWithThinkingParser` (the reasoning parser
  ///     vLLM registers for `glm45`). Off by default; the GLM 4 base
  ///     models don't emit reasoning markers.
  ///   - initialState: Used by continuation requests on thinking-enabled
  ///     GLM 4.5+ checkpoints. ``InitialState/reasoning`` resumes
  ///     mid-`<think>`; ``InitialState/normal`` skips the preamble.
  ///     Ignored when ``acceptThink`` is false.
  init(
    tools: [ToolSpec] = [],
    acceptThink: Bool = false,
    initialState: InitialState = .reasoning,
  ) {
    argumentSchemas = Glm4Parser.buildSchemaTable(from: tools)
    self.acceptThink = acceptThink
    if acceptThink {
      let preambleState: ThinkPreambleExtractor.InitialState = switch initialState {
        case .normal: .normal
        case .reasoning: .reasoning
      }
      thinkPreamble = ThinkPreambleExtractor(
        initialState: preambleState,
        implicitEndTokens: [Self.toolCallStart],
      )
    } else {
      thinkPreamble = ThinkPreambleExtractor(initialState: .normal)
    }
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    var events: [ResponseStreamingEvent] = []
    if acceptThink, thinkPreamble.phase != .done {
      events.append(contentsOf: thinkPreamble.drain(
        buffer: &buffer,
        isEnd: false,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
    }
    if thinkPreamble.phase == .done {
      events.append(contentsOf: scan(isEnd: false))
    }
    pruneConsumedPrefix()
    return events
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    if acceptThink, thinkPreamble.phase != .done {
      events.append(contentsOf: thinkPreamble.drain(
        buffer: &buffer,
        isEnd: true,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
    }
    events.append(contentsOf: scan(isEnd: true))
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    // Any tool call still open at EOS is a truncated emission.
    for index in toolCalls.indices where !toolCalls[index].closed && toolCalls[index].name != nil {
      events.append(contentsOf: closeToolCall(at: index, status: .incomplete))
    }
    if acceptThink {
      events.append(contentsOf: thinkPreamble.finalizeIfOpen(nextSequence: &nextSequence))
    }
    return events
  }

  private static func buildSchemaTable(from tools: [ToolSpec]) -> [String: [String: String]] {
    var table: [String: [String: String]] = [:]
    for tool in tools {
      guard let function = tool["function"] as? [String: Any],
            let name = function["name"] as? String,
            let parameters = function["parameters"] as? [String: Any],
            let properties = parameters["properties"] as? [String: Any]
      else { continue }
      var paramTypes: [String: String] = [:]
      for (paramName, schema) in properties {
        if let inferred = inferTypeFromJsonSchema(schema) {
          paramTypes[paramName] = inferred
        }
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
    let bufChars = Array(buffer)

    // Walk forward through the buffer. Each iteration either emits
    // pre-tool-call content and drops into the next tool-call region,
    // or, at end of buffer, emits any remaining safe content and exits.
    while sentContentIdx < bufChars.count {
      // Find next <tool_call> at or after the cursor.
      let nextStart = bufChars.firstIndexOf(substring: Glm4Parser.toolCallStart, after: sentContentIdx)

      // Emit message content from the cursor up to either the next
      // `<tool_call>` or the safe content end (suffix held back by
      // `<tool_call>` partial overlap when not at end of input).
      let contentEnd: Int
      if let nextStart {
        contentEnd = nextStart
      } else if isEnd {
        contentEnd = bufChars.count
      } else {
        let overlap = partialOverlap(suffixOf: bufChars, with: Array(Glm4Parser.toolCallStart))
        contentEnd = bufChars.count - overlap
      }
      if contentEnd > sentContentIdx {
        events.append(contentsOf: emitContent(
          from: sentContentIdx, to: contentEnd, chars: bufChars,
        ))
        sentContentIdx = contentEnd
      }

      guard let nextStart, nextStart == sentContentIdx else {
        // No tool-call region to enter; either we've run out of
        // buffer or we're holding a partial-marker suffix.
        break
      }

      // Close any open message before entering the tool call region.
      if openMessage != nil {
        events.append(contentsOf: closeMessage(status: .completed))
      }

      // Inner-text bounds. The closing `</tool_call>` may not have
      // arrived yet; in that case we still process whatever args
      // text is available (with `</tool_call>` partial-overlap held
      // back from the inner text the same way vLLM does).
      //
      // At EOS without `</tool_call>` we treat the region as a
      // logical close (`isComplete=true`) so any partial value is
      // routed through `coerceParameter` instead of being lost.
      // `isTruncated` keeps the auto-close path from marking the
      // call as `.completed` – finalize will close it as
      // `.incomplete` afterwards.
      let innerStart = nextStart + Glm4Parser.toolCallStart.count
      let endIdx = bufChars.firstIndexOf(substring: Glm4Parser.toolCallEnd, after: innerStart)
      let isComplete: Bool
      let isTruncated: Bool
      let innerEnd: Int
      if let endIdx {
        isComplete = true
        isTruncated = false
        innerEnd = endIdx
      } else if isEnd {
        isComplete = true
        isTruncated = true
        innerEnd = bufChars.count
      } else {
        isComplete = false
        isTruncated = false
        let inner = bufChars[innerStart ..< bufChars.count]
        let overlap = partialOverlap(suffixOf: Array(inner), with: Array(Glm4Parser.toolCallEnd))
        innerEnd = bufChars.count - overlap
      }
      let innerText = String(bufChars[innerStart ..< innerEnd])

      // Map this region to its tool-call slot (one slot per region,
      // in encounter order). Re-process the same slot on subsequent
      // calls until `</tool_call>` arrives – the diff against
      // `streamedArgs` makes the re-processing idempotent.
      let regionIndex = countCompletedToolCalls()
      if regionIndex >= toolCalls.count {
        let outputIndex = takeOutputIndex()
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
          outputIndex: outputIndex,
        ))
      }

      events.append(contentsOf: processToolCallRegion(
        index: regionIndex, innerText: innerText,
        isComplete: isComplete, isTruncated: isTruncated,
      ))

      if isComplete, !isTruncated {
        sentContentIdx = endIdx! + Glm4Parser.toolCallEnd.count
        continue
      } else {
        // Tool call still streaming. We re-enter this region on
        // the next process() call; the cursor stays at the
        // `<tool_call>` opener so the find-next-region step
        // re-finds it.
        break
      }
    }

    return events
  }

  /// Number of tool-call slots that have already been closed. The next
  /// region we encounter (whether or not it's complete in this buffer)
  /// is at this index.
  private func countCompletedToolCalls() -> Int {
    var n = 0
    for tc in toolCalls where tc.closed {
      n += 1
    }
    return n
  }

  /// Drop any prefix that has already been emitted as message text or
  /// structurally consumed as closed tool-call regions. An incomplete
  /// `<tool_call>` remains in the buffer so the rebuild-and-diff path can
  /// continue from the active region on the next chunk.
  private mutating func pruneConsumedPrefix() {
    guard sentContentIdx > 0 else { return }

    buffer.removeFirst(sentContentIdx)
    sentContentIdx = 0

    while let first = toolCalls.first, first.closed {
      toolCalls.removeFirst()
    }
  }

  private mutating func emitContent(from: Int, to end: Int, chars: [Character]) -> [ResponseStreamingEvent] {
    let chunk = String(chars[from ..< end])
    if chunk.isEmpty { return [] }
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

  // MARK: Per-region streaming

  private mutating func processToolCallRegion(
    index: Int, innerText: String, isComplete: Bool, isTruncated: Bool = false,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    // Tool name comes before the first `\n` or `<arg_key>`. Until the
    // delimiter arrives we hold the slot but emit nothing — except when
    // the region is logically closed (GLM 4.7 zero-arg form
    // `<tool_call>name</tool_call>`); in that case the whole trimmed
    // inner text is the name. Mirrors vLLM's `Glm47MoeModelToolParser`
    // regex `<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>`.
    let nameOrNil = extractToolName(from: innerText, isComplete: isComplete)

    if toolCalls[index].name == nil {
      guard let name = nameOrNil else { return events }
      toolCalls[index].name = name
      let openItem = ResponseFunctionToolCall(
        id: toolCalls[index].id,
        callId: toolCalls[index].callId,
        name: name,
        arguments: "",
        status: .inProgress,
      )
      events.append(.outputItemAdded(.init(
        item: .functionCall(openItem),
        outputIndex: toolCalls[index].outputIndex,
        sequenceNumber: takeSequence(),
      )))
    }

    guard let name = toolCalls[index].name else { return events }

    // Rebuild the args JSON the consumer should see at this point in
    // the stream and diff against what we've already sent.
    let argsSoFar = buildArgsJsonSoFar(toolName: name, innerText: innerText, isComplete: isComplete)
    if argsSoFar.count > toolCalls[index].streamedArgs.count {
      let diffStart = argsSoFar.index(argsSoFar.startIndex, offsetBy: toolCalls[index].streamedArgs.count)
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

  /// Tool name is the run of characters before the first `\n` or
  /// `<arg_key>` in the inner text. When the region is logically closed
  /// and neither delimiter is present, the whole trimmed inner is the
  /// name (GLM 4.7 zero-arg form). Returns nil while a delimiter could
  /// still arrive.
  private func extractToolName(from inner: String, isComplete: Bool) -> String? {
    let nlRange = inner.range(of: "\n")
    let keyRange = inner.range(of: Glm4Parser.argKeyOpen)
    let cut: String.Index
    switch (nlRange, keyRange) {
      case let (.some(n), .some(k)): cut = min(n.lowerBound, k.lowerBound)
      case let (.some(n), .none): cut = n.lowerBound
      case let (.none, .some(k)): cut = k.lowerBound
      case (.none, .none):
        guard isComplete else { return nil }
        let trimmedAll = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmedAll.isEmpty ? nil : trimmedAll
    }
    let trimmed = inner[..<cut].trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
  }

  /// Build the JSON-arguments-so-far string the consumer should see for
  /// this tool call at this point in the stream. Closing `}` is
  /// appended only when `isComplete` (i.e., `</tool_call>` arrived).
  ///
  /// Mirrors vLLM's `_build_args_json_so_far` with one departure: when a
  /// non-string `<arg_value>` is open but not yet closed, vLLM emits
  /// raw partial content (which produces non-prefix-stable diffs and
  /// transiently invalid JSON in the cumulative stream); we instead skip
  /// the partial entirely so the value emits in one piece at
  /// `</arg_value>` close, formatted via the schema's coercion rules.
  private func buildArgsJsonSoFar(toolName: String, innerText: String, isComplete: Bool) -> String {
    var parts: [String] = []

    // Walk all complete <arg_key>...</arg_key><arg_value>...</arg_value> pairs.
    var cursor = innerText.startIndex
    var lastConsumedEnd = innerText.startIndex
    while let keyOpen = innerText.range(of: Glm4Parser.argKeyOpen, range: cursor ..< innerText.endIndex) {
      guard let keyClose = innerText.range(of: Glm4Parser.argKeyClose, range: keyOpen.upperBound ..< innerText.endIndex) else {
        break
      }
      guard let valOpen = innerText.range(of: Glm4Parser.argValueOpen, range: keyClose.upperBound ..< innerText.endIndex) else {
        break
      }
      guard let valClose = innerText.range(of: Glm4Parser.argValueClose, range: valOpen.upperBound ..< innerText.endIndex) else {
        break
      }
      let key = String(innerText[keyOpen.upperBound ..< keyClose.lowerBound])
        .trimmingCharacters(in: .whitespacesAndNewlines)
      let rawValue = String(innerText[valOpen.upperBound ..< valClose.lowerBound])
      let coerced = coerceParameter(funcName: toolName, paramName: key, rawValue: rawValue)
      parts.append(jsonEncodeString(key) + ": " + coerced)
      cursor = valClose.upperBound
      lastConsumedEnd = valClose.upperBound
    }

    // Detect a partial value: an `<arg_value>` opened after the last
    // consumed pair without a matching `</arg_value>`.
    let tailRange = lastConsumedEnd ..< innerText.endIndex
    if let lastValOpen = innerText.range(of: Glm4Parser.argValueOpen, range: tailRange),
       innerText.range(of: Glm4Parser.argValueClose, range: lastValOpen.upperBound ..< innerText.endIndex) == nil
    {
      // Find the key that pairs with this open value: the last
      // `<arg_key>...</arg_key>` strictly before the open.
      var partialKey: String? = nil
      var keyCursor = lastConsumedEnd
      while let ko = innerText.range(of: Glm4Parser.argKeyOpen, range: keyCursor ..< lastValOpen.lowerBound),
            let kc = innerText.range(of: Glm4Parser.argKeyClose, range: ko.upperBound ..< lastValOpen.lowerBound)
      {
        partialKey = String(innerText[ko.upperBound ..< kc.lowerBound])
          .trimmingCharacters(in: .whitespacesAndNewlines)
        keyCursor = kc.upperBound
      }

      if let partialKey {
        // Partial content runs from after `<arg_value>` to the end
        // of inner-text. Hold back any `</arg_value>` partial-tag
        // overlap so we never stream a tag fragment as content.
        var partialContent = String(innerText[lastValOpen.upperBound...])
        let overlap = partialOverlap(
          suffixOf: Array(partialContent),
          with: Array(Glm4Parser.argValueClose),
        )
        if overlap > 0 {
          partialContent.removeLast(overlap)
        }

        if isComplete {
          // Tool call ended without `</arg_value>`. Treat the
          // partial as the final value so the diff naturally
          // closes any open quotes.
          let coerced = coerceParameter(
            funcName: toolName, paramName: partialKey, rawValue: partialContent,
          )
          parts.append(jsonEncodeString(partialKey) + ": " + coerced)
        } else if isStreamableAsString(funcName: toolName, paramName: partialKey) {
          // Stream as an open-quoted string: `"key": "<escaped...`.
          // The closing `"` lands when `</arg_value>` arrives and
          // the pair becomes complete.
          let escaped = jsonEscapeStringContent(partialContent)
          parts.append(jsonEncodeString(partialKey) + ": \"" + escaped)
        }
        // Non-string partial: skip – emit nothing for this pair
        // until `</arg_value>` closes it.
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
  /// a JSON string. False for numeric / boolean / object / array types
  /// **and for unschema'd parameters**, which are both buffered until
  /// `</arg_value>` so that the close path can attempt JSON parsing –
  /// matching vLLM's `_is_string_type` (which returns False with no
  /// tools) and `_deserialize` (which JSON-parses at close).
  private func isStreamableAsString(funcName: String, paramName: String) -> Bool {
    guard let table = argumentSchemas[funcName], let type = table[paramName] else {
      return false
    }
    let t = type.lowercased()
    return t == "string" || t == "str" || t == "text" || t == "varchar" || t == "char" || t == "enum"
  }

  /// JSON-escape the *contents* of a string (between but not including
  /// the surrounding quotes). Returns `""` for empty input.
  private func jsonEscapeStringContent(_ s: String) -> String {
    if s.isEmpty { return "" }
    let encoded = jsonEncodeString(s)
    // `jsonEncodeString` returns `"...escaped..."`; drop the quotes.
    guard encoded.count >= 2 else { return "" }
    return String(encoded.dropFirst().dropLast())
  }

  // MARK: Tool-call close

  private mutating func closeToolCall(at index: Int, status: ItemStatus) -> [ResponseStreamingEvent] {
    guard !toolCalls[index].closed, let name = toolCalls[index].name else { return [] }
    toolCalls[index].closed = true
    // For truncated calls the streaming code may have emitted bytes
    // with an open string and no closing `}`. Append the missing
    // closes so the `arguments` field is parseable JSON; the consumer
    // can still detect truncation via the `incomplete` status. Done
    // calls (`status == .completed`) come through with already-valid
    // args from the streaming closer.
    let argsText: String = if status == .completed {
      toolCalls[index].streamedArgs
    } else {
      defensivelyCloseJSON(toolCalls[index].streamedArgs)
    }
    let doneItem = ResponseFunctionToolCall(
      id: toolCalls[index].id,
      callId: toolCalls[index].callId,
      name: name,
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

  /// Append the closes (string quote, then any unmatched `}`/`]`) needed
  /// to turn the partial-streamed args into parseable JSON. Empty input
  /// stays empty.
  private func defensivelyCloseJSON(_ partial: String) -> String {
    if partial.isEmpty { return partial }
    var stack: [Character] = []
    var inString = false
    var escape = false
    for c in partial {
      if inString {
        if escape { escape = false; continue }
        if c == "\\" { escape = true; continue }
        if c == "\"" { inString = false }
        continue
      }
      if c == "\"" { inString = true; continue }
      if c == "{" { stack.append("}") }
      else if c == "[" { stack.append("]") }
      else if c == "}" || c == "]" {
        _ = stack.popLast()
      }
    }
    var out = partial
    if inString { out.append("\"") }
    while let close = stack.popLast() {
      out.append(close)
    }
    return out
  }

  // MARK: Message open/close

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
        itemId: msg.id, outputIndex: msg.outputIndex, contentIndex: 0,
        text: msg.emittedText, sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: msg.id, outputIndex: msg.outputIndex, contentIndex: 0,
        part: .outputText(part), sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .message(.init(id: msg.id, content: [.outputText(part)], status: status)),
        outputIndex: msg.outputIndex, sequenceNumber: takeSequence(),
      )),
    ]
  }

  // MARK: Coercion

  private func coerceParameter(funcName: String, paramName: String, rawValue: String) -> String {
    let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.lowercased() == "null" { return "null" }
    if let table = argumentSchemas[funcName], let type = table[paramName] {
      return coerce(rawValue: rawValue, schemaType: type)
    }
    // No schema entry: try JSON parsing first, mirroring vLLM's
    // `_deserialize` and sglang's `parse_arguments` strategy chain
    // (json.loads → ast.literal_eval → string fallback). A bare `5`
    // becomes the integer `5`, `{"a":1}` stays as an object, and
    // unparseable bare tokens like `Beijing` fall back to a JSON
    // string. The trimmed value is what we encode, matching vLLM's
    // `value.strip()` in `extract_tool_calls`.
    if !trimmed.isEmpty, isValidJSON(trimmed) {
      return trimmed
    }
    return jsonEncodeString(trimmed)
  }

  private func coerce(rawValue: String, schemaType: String) -> String {
    let type = schemaType.lowercased()
    let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    if type == "string" || type == "str" || type == "text" {
      return jsonEncodeString(rawValue)
    }
    if type.hasPrefix("int") {
      if let i = Int(trimmed) { return String(i) }
      return jsonEncodeString(rawValue)
    }
    if type.hasPrefix("num") || type.hasPrefix("float") || type == "double" {
      if let d = Double(trimmed) {
        if d.truncatingRemainder(dividingBy: 1) == 0,
           !trimmed.contains("."), !trimmed.lowercased().contains("e")
        {
          if let i = Int(exactly: d) { return String(i) }
          return trimmed
        }
        return trimmed
      }
      return jsonEncodeString(rawValue)
    }
    if type == "boolean" || type == "bool" {
      // Mirror vLLM (`_deserialize`) and sglang (`parse_arguments`):
      // accept JSON booleans (`true`/`false`) and Python literals
      // (`True`/`False`, via the lowercase fold). For anything else,
      // fall through to JSON-parse-then-string so `1` and `0` emit
      // as numbers (not booleans), `null` emits as null, and bare
      // tokens like `yes`/`maybe` emit as JSON strings – none of
      // these would coerce to bool in either Python reference.
      let lower = trimmed.lowercased()
      if lower == "true" { return "true" }
      if lower == "false" { return "false" }
      if !trimmed.isEmpty, isValidJSON(trimmed) { return trimmed }
      return jsonEncodeString(rawValue)
    }
    if type == "object" || type == "array" {
      if isValidJSON(trimmed) {
        return trimmed
      }
      return jsonEncodeString(rawValue)
    }
    return jsonEncodeString(rawValue)
  }

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
