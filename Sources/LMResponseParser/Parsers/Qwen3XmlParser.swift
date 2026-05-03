// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Qwen 3 Coder / Qwen 3.5 XML tool-call format.
///
/// **Wire shape.** Tool calls use a nested XML structure with literal text
/// values rather than JSON arguments:
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=city>Tokyo</parameter>
/// <parameter=units>metric</parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Multiple tool calls in a single response are emitted in sequence, each
/// in its own `<tool_call>` envelope. Text outside `<tool_call>` blocks is
/// normal message content. Reasoning content uses the `<think>` …
/// `</think>` shape from the Qwen base parser.
///
/// **Argument coercion.** Parameter values arrive as raw strings; the
/// parser consults the `tools` JSON-schema (passed at construction) to coerce
/// each value to the right JSON type before composing the
/// `arguments` string. Without a matching schema entry, the value is kept
/// as a string. Type names recognized: `integer`, `int`, `number`, `float`,
/// `boolean`, `bool`, `array`, `object`, `string` (default). The literal
/// `null` decodes to JSON null.
struct Qwen3XmlParser: ResponseFormatParser {
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  /// Envelope tokens. Defaults are the Qwen 3 Coder / 3.5 spec; Seed-OSS
  /// passes its `<seed:…>`-prefixed equivalents through the init params,
  /// mirroring the way LongCat parameterizes ``HermesParser``. The inner
  /// `<function=` / `<parameter=` markers are identical for every model
  /// in this family and so remain static.
  private let thinkStart: String
  private let thinkEnd: String
  private let toolCallStart: String
  private let toolCallEnd: String

  /// Step-3.5-Flash quirk: the model habitually emits a stray `\n`
  /// immediately before and/or after `</think>`. With this flag on, the
  /// reasoning chunk's trailing `\n` is dropped when it sits right
  /// before `</think>`, and the message chunk's leading `\n` is dropped
  /// when it appears right after `</think>`. Mirrors vLLM's
  /// `Step3p5ReasoningParser`.
  private let trimNewlineAroundThinkEnd: Bool

  /// Deferred leading-`\n` skip after `</think>`. Set when the parser
  /// exits reasoning via `.thinkEnd` but the next byte hasn't yet
  /// arrived; consumed on the next scanNormal pass that sees the byte.
  private var pendingLeadingNewlineSkip: Bool = false
  private static let functionStart = "<function="
  private static let functionEnd = "</function>"
  private static let parameterStart = "<parameter="
  private static let parameterEnd = "</parameter>"

  // Active accumulated output. Consumed prefixes are pruned after each scan.
  private var buffer: String = ""

  private enum Phase { case reasoning, normal }
  private var phase: Phase

  /// Whether the beginning of the current reasoning block has already been
  /// checked for an optional model-emitted `<think>` marker. Kept separate
  /// from `sentReasoningIdx` so buffer pruning can rebase the cursor to zero
  /// without making a later literal `<think>` look like a fresh opener.
  private var reasoningStartResolved: Bool = false

  /// True until the normal phase emits content or opens a function call.
  /// Closed tool-call slots used to preserve this "start of normal output"
  /// fact implicitly; pruning removes those slots, so keep the fact directly.
  private var normalPhaseCanStartReasoning: Bool = true

  private var sentReasoningIdx: Int = 0

  /// Cursor into `buffer` for the XML state machine in the normal phase.
  /// Advances past every character the parser has either emitted (as
  /// outputText / arguments delta) or recognized as structural noise.
  private var parsedIdx: Int = 0

  private var openReasoning: OpenReasoning?
  private var openMessage: OpenMessage?
  private var toolCalls: [OpenToolCall] = []

  /// Whether the cursor is currently between `<tool_call>` and the
  /// matching `</tool_call>`. Tracked independently from the open
  /// tool-call slot's `closed` flag because `</function>` closes the
  /// item-event lifecycle but the wire envelope only closes on the
  /// outer `</tool_call>` – text between the two ought to be discarded
  /// as structural noise, not emitted as message content.
  private var insideToolCallEnvelope: Bool = false

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  /// Tool argument schemas, indexed by function name. Empty when the
  /// caller didn't supply a tools spec; the parser then treats every
  /// parameter as a string.
  private let argumentSchemas: [String: [String: String]]

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
    /// Allocated lazily, when `<function=NAME>` is parsed and we're
    /// about to emit `output_item.added`. A `<tool_call>` opener
    /// without any inner `<function=...>` leaves this nil, so the
    /// failed envelope doesn't burn an output slot and the next
    /// item's index stays consecutive. Mirrors DeepSeekR1Parser's
    /// pattern; the prior eager-allocation-then-recycle approach
    /// was correct but relied on a non-obvious invariant (no other
    /// slot opens between `<tool_call>` and `</tool_call>`).
    var outputIndex: Int?
    var name: String?
    /// Cumulative arguments JSON string emitted so far, used to compute
    /// each delta as the difference against new fragments.
    var argsEmitted: String = ""
    /// Number of parameters appended to the JSON object so far.
    var paramCount: Int = 0
    /// Whether the JSON object opening `{` has been emitted.
    var openedJSON: Bool = false
    var closed: Bool = false
  }

  init(
    initialState: InitialState = .normal,
    tools: [ToolSpec] = [],
    thinkStart: String = "<think>",
    thinkEnd: String = "</think>",
    toolCallStart: String = "<tool_call>",
    toolCallEnd: String = "</tool_call>",
    trimNewlineAroundThinkEnd: Bool = false,
  ) {
    switch initialState {
      case .normal: phase = .normal
      case .reasoning: phase = .reasoning
    }
    argumentSchemas = Qwen3XmlParser.buildSchemaTable(from: tools)
    self.thinkStart = thinkStart
    self.thinkEnd = thinkEnd
    self.toolCallStart = toolCallStart
    self.toolCallEnd = toolCallEnd
    self.trimNewlineAroundThinkEnd = trimNewlineAroundThinkEnd
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

  // MARK: Schema construction

  /// Build a lookup from function name → (parameter name → schema type)
  /// from a list of OpenAI-format tool specs. Resilient to missing keys
  /// at any level – tools that don't declare parameter types simply
  /// produce no entry, and the parser falls back to string values.
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
    pruneConsumedPrefix()
    return events
  }

  // MARK: Reasoning phase (matches QwenParser semantics)

  private mutating func scanReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)

    let thinkStartChars = Array(thinkStart)
    let thinkEndChars = Array(thinkEnd)
    let toolStartChars = Array(toolCallStart)

    if !reasoningStartResolved {
      if sentReasoningIdx == 0,
         bufChars.count >= thinkStartChars.count,
         Array(bufChars[0 ..< thinkStartChars.count]) == thinkStartChars
      {
        sentReasoningIdx = thinkStartChars.count
        reasoningStartResolved = true
      } else if sentReasoningIdx == 0, !isEnd {
        // Buffer might still grow into a leading `<think>`.
        let limit = Swift.min(bufChars.count, thinkStartChars.count - 1)
        if limit > 0, bufChars[..<limit].elementsEqual(thinkStartChars[..<limit]), limit == bufChars.count {
          return events
        }
        reasoningStartResolved = true
      } else {
        reasoningStartResolved = true
      }
    }

    let endIdx = bufChars.firstIndexOf(substring: thinkEnd, after: sentReasoningIdx)
    let toolIdx = bufChars.firstIndexOf(substring: toolCallStart, after: sentReasoningIdx)

    let exitIdx: Int?
    let exitMarker: ExitMarker?
    switch (endIdx, toolIdx) {
      case let (.some(e), .some(t)) where e <= t:
        exitIdx = e; exitMarker = .thinkEnd
      case (.some, .some):
        exitIdx = toolIdx; exitMarker = .toolCall
      case (.some, .none):
        exitIdx = endIdx; exitMarker = .thinkEnd
      case (.none, .some):
        exitIdx = toolIdx; exitMarker = .toolCall
      case (.none, .none):
        exitIdx = nil; exitMarker = nil
    }

    let safeEnd: Int
    if let exitIdx {
      safeEnd = exitIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let endOverlap = partialOverlap(suffixOf: bufChars, with: thinkEndChars)
      let toolOverlap = partialOverlap(suffixOf: bufChars, with: toolStartChars)
      safeEnd = bufChars.count - Swift.max(endOverlap, toolOverlap)
    }

    if safeEnd > sentReasoningIdx {
      // Step-3.5 trim: when chunk would end in `\n` and we don't yet
      // know whether `</think>` follows, hold the `\n` (don't advance
      // sentReasoningIdx past it). On the call where `</think>` is
      // found, the held `\n` is dropped instead of emitted.
      var chunkEnd = safeEnd
      var holdNewline = false
      if trimNewlineAroundThinkEnd, exitIdx == nil,
         safeEnd > sentReasoningIdx, bufChars[safeEnd - 1] == "\n"
      {
        chunkEnd = safeEnd - 1
        holdNewline = true
      } else if trimNewlineAroundThinkEnd, exitMarker == .thinkEnd,
                safeEnd > sentReasoningIdx, bufChars[safeEnd - 1] == "\n"
      {
        // Trailing `\n` immediately before `</think>` – drop it.
        chunkEnd = safeEnd - 1
      }
      let chunk = String(bufChars[sentReasoningIdx ..< chunkEnd])
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
      sentReasoningIdx = holdNewline ? safeEnd - 1 : safeEnd
    }

    guard let exitIdx, exitIdx == safeEnd, let exitMarker else { return events }
    switch exitMarker {
      case .thinkEnd:
        parsedIdx = exitIdx + thinkEndChars.count
        // Step-3.5 trim: drop a leading `\n` immediately after
        // `</think>` so it doesn't bleed into the message text. Defer
        // when the byte hasn't arrived yet (streaming case).
        if trimNewlineAroundThinkEnd {
          if parsedIdx < bufChars.count, bufChars[parsedIdx] == "\n" {
            parsedIdx += 1
          } else if parsedIdx >= bufChars.count {
            pendingLeadingNewlineSkip = true
          }
        }
      case .toolCall:
        parsedIdx = exitIdx
    }
    events.append(contentsOf: closeReasoning(status: .completed))
    reasoningStartResolved = false
    phase = .normal
    return events
  }

  private enum ExitMarker { case thinkEnd, toolCall }

  // MARK: Normal phase: cursor-based XML state machine

  private mutating func scanNormal(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    // Older Qwen 3 templates may emit `<think>`; honor it when we are
    // still at the very start of normal-phase processing.
    if normalPhaseCanStartReasoning, openMessage == nil, openReasoning == nil {
      if let _ = transitionToReasoningIfMarkerPresent(isEnd: isEnd) {
        return events
      }
    }

    // Step 1: emit any text before the next structural tag as content.
    events.append(contentsOf: emitNormalText(isEnd: isEnd))

    // Step 2: walk tags. Each iteration consumes one tag or breaks.
    while parsedIdx < buffer.count {
      let slice = buffer.dropFirst(parsedIdx)
      // Tool call boundary tags ----
      if slice.hasPrefix(toolCallStart) {
        parsedIdx += toolCallStart.count
        insideToolCallEnvelope = true
        openToolCallSlotIfNeeded()
        continue
      }
      if slice.hasPrefix(toolCallEnd) {
        parsedIdx += toolCallEnd.count
        insideToolCallEnvelope = false
        if let index = toolCalls.indices.last, !toolCalls[index].closed {
          if toolCalls[index].name == nil {
            // No valid `<function=…>` was seen inside this
            // `<tool_call>` envelope. With lazy outputIndex
            // allocation the slot never claimed an index, so
            // we just drop it – no recycle needed.
            toolCalls.remove(at: index)
          } else if toolCalls[index].openedJSON {
            // Close the slot if `</function>` finalized
            // arguments but we deferred the item-done event
            // until the outer envelope closed.
            events.append(contentsOf: closeToolCall(at: index, status: .completed))
          }
        }
        continue
      }
      // Function open: <function=NAME>
      if slice.hasPrefix(Qwen3XmlParser.functionStart) {
        let afterPrefixIdx = parsedIdx + Qwen3XmlParser.functionStart.count
        guard let nameEnd = buffer.firstIndex(of: ">", from: afterPrefixIdx) else { break }
        // Reject malformed `<function=name` shapes where the closing
        // `>` is missing and a different `<` opens before it (e.g.,
        // a stray `<parameter=...>` inside the function tag).
        if let openLt = buffer.firstIndex(of: "<", from: afterPrefixIdx),
           openLt < nameEnd
        {
          parsedIdx = openLt
          continue
        }
        let name = String(buffer[buffer.index(buffer.startIndex, offsetBy: afterPrefixIdx) ..< buffer.index(buffer.startIndex, offsetBy: nameEnd)])
        events.append(contentsOf: openFunction(name: name))
        parsedIdx = nameEnd + 1
        continue
      }
      if slice.hasPrefix(Qwen3XmlParser.functionEnd) {
        events.append(contentsOf: closeFunction())
        parsedIdx += Qwen3XmlParser.functionEnd.count
        continue
      }
      // Parameter: <parameter=KEY>value...</parameter>
      if slice.hasPrefix(Qwen3XmlParser.parameterStart) {
        let afterPrefixIdx = parsedIdx + Qwen3XmlParser.parameterStart.count
        guard let nameEnd = buffer.firstIndex(of: ">", from: afterPrefixIdx) else { break }
        let valueStart = nameEnd + 1
        guard let (valueEnd, terminatorLen) = findParameterValueEnd(from: valueStart) else { break }
        let key = String(buffer[buffer.index(buffer.startIndex, offsetBy: afterPrefixIdx) ..< buffer.index(buffer.startIndex, offsetBy: nameEnd)])
        var rawValue = String(buffer[buffer.index(buffer.startIndex, offsetBy: valueStart) ..< buffer.index(buffer.startIndex, offsetBy: valueEnd)])
        if rawValue.hasPrefix("\n") { rawValue.removeFirst() }
        if rawValue.hasSuffix("\n") { rawValue.removeLast() }
        events.append(contentsOf: appendParameter(key: key, rawValue: rawValue))
        parsedIdx = valueEnd + terminatorLen
        continue
      }
      // Inside the tool-call envelope (or the wrapperless fallback –
      // an open `<function=…>` block without a `<tool_call>` outer
      // wrapper), the next tag isn't recognized: walk forward to the
      // next `<` if there is one; otherwise wait. All text in this
      // region is structural noise (whitespace between tags), not
      // message content.
      if insideToolCallEnvelope || (toolCalls.last?.closed == false) {
        if slice.first == "<" {
          if !isEnd, couldStillBecomeATag(slice: String(slice)) {
            return events // wait for more bytes
          }
          parsedIdx += 1
          continue
        } else {
          if let nextLt = slice.firstIndex(of: "<") {
            let dist = slice.distance(from: slice.startIndex, to: nextLt)
            parsedIdx += dist
            continue
          } else {
            parsedIdx += slice.count
            continue
          }
        }
      }
      break
    }

    return events
  }

  /// Find the end of a parameter value starting at `from`. Returns the
  /// (end index, terminator length) tuple; the terminator length is the
  /// number of characters to skip past the marker – `</parameter>` skips
  /// 12 chars while a peeked `<parameter=` or `</function>` skips 0.
  /// Returns nil when the value isn't terminated yet (more bytes needed).
  private func findParameterValueEnd(from start: Int) -> (end: Int, terminatorLen: Int)? {
    let pStart = buffer.index(buffer.startIndex, offsetBy: start)
    let endTokenIdx = buffer.range(of: Qwen3XmlParser.parameterEnd, range: pStart ..< buffer.endIndex)
    let nextParamIdx = buffer.range(of: Qwen3XmlParser.parameterStart, range: pStart ..< buffer.endIndex)
    let funcEndIdx = buffer.range(of: Qwen3XmlParser.functionEnd, range: pStart ..< buffer.endIndex)

    var candidates: [(idx: Int, len: Int)] = []
    if let r = endTokenIdx {
      candidates.append((buffer.distance(from: buffer.startIndex, to: r.lowerBound), Qwen3XmlParser.parameterEnd.count))
    }
    if let r = nextParamIdx {
      candidates.append((buffer.distance(from: buffer.startIndex, to: r.lowerBound), 0))
    }
    if let r = funcEndIdx {
      candidates.append((buffer.distance(from: buffer.startIndex, to: r.lowerBound), 0))
    }
    guard let chosen = candidates.min(by: { $0.idx < $1.idx }) else { return nil }
    return (chosen.idx, chosen.len)
  }

  private func couldStillBecomeATag(slice: String) -> Bool {
    for tag in [
      toolCallStart,
      toolCallEnd,
      Qwen3XmlParser.functionStart,
      Qwen3XmlParser.functionEnd,
      Qwen3XmlParser.parameterStart,
      Qwen3XmlParser.parameterEnd,
    ] {
      if tag.hasPrefix(slice) { return true }
    }
    return false
  }

  /// Emit normal-message text up to the next structural tag (or up to
  /// the safe end of the buffer when no tag is found and we're not at
  /// the end of input).
  private mutating func emitNormalText(isEnd: Bool) -> [ResponseStreamingEvent] {
    // Inside an open tool call, no normal text: parameter values are
    // emitted as JSON deltas, and structural whitespace is discarded.
    if let last = toolCalls.last, !last.closed { return [] }

    let bufChars = Array(buffer)
    // Step-3.5 trim: deferred leading-`\n` skip after `</think>`. Resolve
    // it now that more bytes have arrived.
    if pendingLeadingNewlineSkip, parsedIdx < bufChars.count {
      if bufChars[parsedIdx] == "\n" {
        parsedIdx += 1
      }
      pendingLeadingNewlineSkip = false
    }
    let toolStartChars = Array(toolCallStart)
    let funcStartChars = Array(Qwen3XmlParser.functionStart)

    // Halt at the earlier of `<tool_call>` or `<function=` so the
    // wrapperless fallback (`<function=…>` without an enclosing
    // `<tool_call>`) reaches the function handler instead of being
    // emitted as plain message content.
    let toolStartIdx = bufChars.firstIndexOf(substring: toolCallStart, after: parsedIdx)
    let funcStartIdx = bufChars.firstIndexOf(substring: Qwen3XmlParser.functionStart, after: parsedIdx)
    let firstStart: Int? = switch (toolStartIdx, funcStartIdx) {
      case let (.some(t), .some(f)): Swift.min(t, f)
      case let (.some(t), .none): t
      case let (.none, .some(f)): f
      case (.none, .none): nil
    }

    let sendableEnd: Int
    if let firstStart {
      sendableEnd = firstStart
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      let toolOverlap = partialOverlap(suffixOf: bufChars, with: toolStartChars)
      let funcOverlap = partialOverlap(suffixOf: bufChars, with: funcStartChars)
      sendableEnd = bufChars.count - Swift.max(toolOverlap, funcOverlap)
    }
    guard sendableEnd > parsedIdx else { return [] }

    let chunk = String(bufChars[parsedIdx ..< sendableEnd])
    parsedIdx = sendableEnd
    normalPhaseCanStartReasoning = false
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

  private mutating func transitionToReasoningIfMarkerPresent(isEnd: Bool) -> [ResponseStreamingEvent]? {
    let bufChars = Array(buffer)
    let thinkStartChars = Array(thinkStart)
    let cursor = parsedIdx
    let available = bufChars.count - cursor

    if available >= thinkStartChars.count {
      if Array(bufChars[cursor ..< cursor + thinkStartChars.count]) == thinkStartChars {
        sentReasoningIdx = cursor + thinkStartChars.count
        reasoningStartResolved = true
        phase = .reasoning
        return []
      }
      return nil
    }
    if available > 0 {
      let slice = Array(bufChars[cursor ..< bufChars.count])
      if Array(thinkStartChars[0 ..< slice.count]) == slice, !isEnd {
        return []
      }
    }
    return nil
  }

  // MARK: Tool-call slot lifecycle

  private mutating func openToolCallSlotIfNeeded() {
    // A new <tool_call> opens a fresh slot. The function name (and
    // therefore the outputIndex) is allocated lazily in
    // openFunction, so a `<tool_call></tool_call>` envelope without
    // any inner `<function=…>` doesn't burn a slot.
    if let last = toolCalls.last, !last.closed {
      // Already inside an open tool call (shouldn't normally happen,
      // but be defensive for malformed input).
      return
    }
    toolCalls.append(OpenToolCall(
      id: IDFactory.make(.functionCall),
      callId: IDFactory.make(.callId),
    ))
  }

  private mutating func openFunction(name: String) -> [ResponseStreamingEvent] {
    normalPhaseCanStartReasoning = false

    // Defensive: if no <tool_call> was seen but we got <function=…>,
    // open a slot now.
    if toolCalls.last?.closed != false {
      toolCalls.append(OpenToolCall(
        id: IDFactory.make(.functionCall),
        callId: IDFactory.make(.callId),
      ))
    }

    var events: [ResponseStreamingEvent] = []
    // Close any open message before opening the function item.
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }

    let index = toolCalls.count - 1
    var call = toolCalls[index]
    call.name = name
    // Allocate the outputIndex now that we're committing to emit
    // output_item.added.
    let outputIndex = takeOutputIndex()
    call.outputIndex = outputIndex
    toolCalls[index] = call

    let openItem = ResponseFunctionToolCall(
      id: call.id,
      callId: call.callId,
      name: name,
      arguments: "",
      status: .inProgress,
    )
    events.append(.outputItemAdded(.init(
      item: .functionCall(openItem),
      outputIndex: outputIndex,
      sequenceNumber: takeSequence(),
    )))
    return events
  }

  private mutating func appendParameter(key: String, rawValue: String) -> [ResponseStreamingEvent] {
    guard let index = toolCalls.indices.last, !toolCalls[index].closed else { return [] }
    var call = toolCalls[index]
    guard let funcName = call.name, let outputIndex = call.outputIndex else { return [] }

    var fragment = ""
    if !call.openedJSON {
      fragment += "{"
      call.openedJSON = true
    }
    if call.paramCount > 0 {
      fragment += ", "
    }
    let coercedJSON = coerceParameter(funcName: funcName, paramName: key, rawValue: rawValue)
    fragment += jsonEncodeString(key) + ": " + coercedJSON
    call.paramCount += 1
    call.argsEmitted += fragment
    toolCalls[index] = call

    return [.functionCallArgumentsDelta(.init(
      itemId: call.id,
      outputIndex: outputIndex,
      delta: fragment,
      sequenceNumber: takeSequence(),
    ))]
  }

  private mutating func closeFunction() -> [ResponseStreamingEvent] {
    guard let index = toolCalls.indices.last, !toolCalls[index].closed,
          toolCalls[index].name != nil,
          let outputIndex = toolCalls[index].outputIndex else { return [] }
    var call = toolCalls[index]

    // Emit the closing `}` (or `{}` for parameter-less calls). The
    // matching `</tool_call>` triggers the actual item-done event, so
    // any structural whitespace between `</function>` and
    // `</tool_call>` stays inside the envelope and is discarded.
    let closingFragment = if !call.openedJSON {
      "{}"
    } else {
      "}"
    }
    call.argsEmitted += closingFragment
    call.openedJSON = true
    toolCalls[index] = call
    return [
      .functionCallArgumentsDelta(.init(
        itemId: call.id,
        outputIndex: outputIndex,
        delta: closingFragment,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeToolCall(at index: Int, status: ItemStatus) -> [ResponseStreamingEvent] {
    var call = toolCalls[index]
    guard !call.closed, let name = call.name, let outputIndex = call.outputIndex else { return [] }
    call.closed = true
    toolCalls[index] = call

    let doneItem = ResponseFunctionToolCall(
      id: call.id,
      callId: call.callId,
      name: name,
      arguments: call.argsEmitted,
      status: status,
    )
    return [
      .functionCallArgumentsDone(.init(
        itemId: call.id,
        outputIndex: outputIndex,
        arguments: call.argsEmitted,
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .functionCall(doneItem),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  /// Drop any buffer prefix that has already been emitted or structurally
  /// consumed. Active XML tags, held marker suffixes, and open tool-call
  /// envelopes remain in the buffer so later chunks can complete them.
  private mutating func pruneConsumedPrefix() {
    let dropCount: Int = switch phase {
      case .reasoning:
        sentReasoningIdx
      case .normal:
        parsedIdx
    }
    guard dropCount > 0 else { return }

    buffer.removeFirst(dropCount)
    rebase(&sentReasoningIdx, dropping: dropCount)
    rebase(&parsedIdx, dropping: dropCount)

    while let first = toolCalls.first, first.closed {
      toolCalls.removeFirst()
    }
  }

  private func rebase(_ cursor: inout Int, dropping dropCount: Int) {
    cursor = Swift.max(0, cursor - dropCount)
  }

  // MARK: Argument coercion

  private func coerceParameter(funcName: String, paramName: String, rawValue: String) -> String {
    if rawValue.lowercased() == "null" {
      return "null"
    }
    guard let table = argumentSchemas[funcName], let type = table[paramName] else {
      return jsonEncodeString(rawValue)
    }
    return coerce(rawValue: rawValue, schemaType: type)
  }

  private func coerce(rawValue: String, schemaType: String) -> String {
    let type = schemaType.lowercased()
    if type == "string" || type == "str" || type == "text" || type == "varchar" || type == "char" || type == "enum" {
      return jsonEncodeString(rawValue)
    }
    if type.hasPrefix("int") || type.hasPrefix("uint") || type.hasPrefix("long") || type.hasPrefix("short") || type.hasPrefix("unsigned") {
      if let i = Int(rawValue) { return String(i) }
      if isIntegerLiteral(rawValue) { return rawValue.trimmingCharacters(in: .whitespacesAndNewlines) }
      return jsonEncodeString(rawValue)
    }
    if type.hasPrefix("num") || type.hasPrefix("float") || type == "double" {
      if let d = Double(rawValue) {
        if d.truncatingRemainder(dividingBy: 1) == 0, !rawValue.contains("."), !rawValue.lowercased().contains("e") {
          if let i = Int(exactly: d) { return String(i) }
          return rawValue
        }
        return rawValue
      }
      return jsonEncodeString(rawValue)
    }
    if type == "boolean" || type == "bool" || type == "binary" {
      switch rawValue.lowercased() {
        case "true": return "true"
        case "false": return "false"
        default: return "false"
      }
    }
    if type == "object" || type.hasPrefix("dict") || type == "array" || type == "arr" || type == "sequence" || type.hasPrefix("list") {
      // Try parse-as-JSON first.
      if isValidJSON(rawValue) {
        return rawValue
      }
      // Then try Python-literal-eval-style fallback for `'a'`, `True`, etc.
      if let pythonized = pythonLiteralToJSON(rawValue) {
        return pythonized
      }
      return jsonEncodeString(rawValue)
    }
    // Unknown type: mirror sglang's `else` branch in `_convert_param_value`,
    // which attempts `ast.literal_eval` before degenerating to a string.
    // A bare `42` becomes the integer `42`, `True` becomes `true`, and
    // unparseable tokens like `Beijing` fall back to a JSON string.
    if let pythonized = pythonLiteralToJSON(rawValue) {
      return pythonized
    }
    return jsonEncodeString(rawValue)
  }

  private func isIntegerLiteral(_ input: String) -> Bool {
    let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return false }
    let digits = trimmed.first == "-" ? trimmed.dropFirst() : trimmed[...]
    return !digits.isEmpty && digits.allSatisfy(\.isNumber)
  }

  private func pythonLiteralToJSON(_ input: String) -> String? {
    // Mirrors sglang's `ast.literal_eval` fallback when strict JSON
    // decode fails. ``PythonLiteral`` is structurally aware, so tokens
    // like `True`/`False`/`None` inside string contents are preserved.
    let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    guard let (json, end) = PythonLiteral.parseValue(
      in: trimmed,
      from: trimmed.startIndex,
    ) else { return nil }
    let tail = trimmed[end...].trimmingCharacters(in: .whitespacesAndNewlines)
    guard tail.isEmpty else { return nil }
    return isValidJSON(json) ? json : nil
  }

  private func jsonEncodeString(_ s: String) -> String {
    // Use JSONSerialization with the value wrapped in an array, then
    // strip the array brackets, to get spec-compliant JSON string
    // escaping including unicode handling.
    if let data = try? JSONSerialization.data(withJSONObject: [s], options: []),
       let encoded = String(data: data, encoding: .utf8),
       encoded.count >= 2
    {
      return String(encoded.dropFirst().dropLast())
    }
    // Fallback: manual escape.
    var out = "\""
    for ch in s {
      switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out.append(ch)
      }
    }
    out += "\""
    return out
  }

  // MARK: Item open/close (reasoning + message)

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

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}

// MARK: Buffer helpers

private extension String {
  /// Locate the first occurrence of `character` at-or-after `offset`,
  /// returning the integer offset (or nil when not found).
  func firstIndex(of character: Character, from offset: Int) -> Int? {
    guard offset <= count else { return nil }
    let start = index(startIndex, offsetBy: offset)
    guard let r = self[start ..< endIndex].firstIndex(of: character) else { return nil }
    return distance(from: startIndex, to: r)
  }
}
