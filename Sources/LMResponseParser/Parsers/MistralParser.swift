// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Mistral tool-call format(s).
///
/// **Wire shape.** Mistral models emit one of three formats, all signaled
/// by a shared `[TOOL_CALLS` prefix:
///
/// 1. **JSON-array (pre-v11 canonical)** – `[TOOL_CALLS] [{"name": "...", "arguments": {...}}, ...]`.
///    The closing `]` closes the array; multiple calls live as siblings.
/// 2. **Compact with `[ARGS]` separator (sglang templates, some HF tokenizers)**
///    – `[TOOL_CALLS]name[ARGS]{...}`.
/// 3. **Compact without separator (vLLM v11+ canonical)** –
///    `[TOOL_CALLS]name{...}`. The name runs from the marker to the first
///    `{`. Multiple calls are concatenated without a separator.
///
/// Plain text outside the markers is normal message content. Mistral
/// has no reasoning markers; this parser does not emit reasoning items.
struct MistralParser: ResponseFormatParser {
  /// Initial reasoning phase. ``reasoning`` is used by continuation
  /// requests on Magistral models whose `priorOutput` ended inside an
  /// unclosed `[THINK]` block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let toolCallsMarker = "[TOOL_CALLS"
  private static let argsMarker = "[ARGS"
  // Magistral reasoning preamble. Distinct from base Mistral / Mixtral,
  // which never emit these tokens.
  private static let thinkStart = "[THINK]"
  private static let thinkEnd = "[/THINK]"

  private var buffer: String = ""
  private var parsedIdx: Int = 0

  // When true, the parser scans for a leading `[THINK]...[/THINK]`
  // reasoning preamble before the tool-call phase. Off for base
  // Mistral / Mixtral; on for Magistral checkpoints.
  private let acceptThink: Bool

  private var openMessage: OpenMessage?
  private var openReasoning: OpenReasoning?
  private var toolCalls: [OpenToolCall] = []

  // Reasoning phase. ``unknown`` is the entry state when
  // ``acceptThink`` is true: hold a leading-prefix overlap until we
  // know whether `[THINK]` is forming. ``inThink`` is set after
  // consuming `[THINK]`. ``done`` is the post-`[/THINK]` state (and
  // the only state used when ``acceptThink`` is false – there's no
  // reasoning to look for).
  private var thinkPhase: ThinkPhase

  private enum ThinkPhase {
    case unknown
    case inThink
    case done
  }

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

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
    var arguments: String
  }

  /// - Parameters:
  ///   - acceptThink: When true, the parser scans for a leading
  ///     `[THINK]...[/THINK]` reasoning preamble. Off by default
  ///     (base Mistral / Mixtral never emit these tokens); on for
  ///     Magistral. Mirrors vLLM's `MistralReasoningParser`.
  ///   - initialState: Used by continuation requests on Magistral
  ///     models whose `priorOutput` ended inside an unclosed
  ///     `[THINK]` block – pass ``InitialState/reasoning`` to start
  ///     already inside the reasoning phase.
  init(
    acceptThink: Bool = false,
    initialState: InitialState = .normal,
  ) {
    self.acceptThink = acceptThink
    if !acceptThink {
      thinkPhase = .done
    } else {
      switch initialState {
        case .normal:
          thinkPhase = .unknown
        case .reasoning:
          thinkPhase = .inThink
      }
    }
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    var events = drainReasoning(isEnd: false)
    if thinkPhase == .done {
      events.append(contentsOf: scan(isEnd: false))
    }
    return events
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = drainReasoning(isEnd: true)
    if thinkPhase == .done {
      events.append(contentsOf: scan(isEnd: true))
    }
    if openReasoning != nil {
      events.append(contentsOf: closeReasoning(status: .incomplete))
    }
    // A `[TOOL_CALLS]` marker followed by an unparseable JSON array (or
    // a truncated compact `name[ARGS]…`) leaves bytes the scan loop
    // can't consume. Surface them as plain message content so the user
    // sees what the model produced – vLLM's `extract_tool_calls`
    // failure path returns the same bytes in the message body.
    if parsedIdx < buffer.count {
      let remaining = String(buffer.dropFirst(parsedIdx))
      parsedIdx = buffer.count
      events.append(contentsOf: emitContent(remaining))
    }
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    return events
  }

  // MARK: Reasoning preamble

  /// Drive the reasoning state machine before the tool-call scan kicks
  /// in. It may emit ordinary content before `[THINK]`, then consumes
  /// the reasoning block until the preamble is resolved – at which
  /// point ``thinkPhase`` becomes ``done`` and the regular `scan` runs.
  private mutating func drainReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    if thinkPhase == .done { return events }

    // Scan from the buffer head (we strip consumed bytes from the
    // front rather than advancing parsedIdx, so the tool-call scan
    // sees the post-reasoning buffer cleanly).
    while thinkPhase != .done {
      switch thinkPhase {
        case .unknown:
          if buffer.isEmpty { return events }

          let startRange = buffer.range(of: Self.thinkStart)
          let endRange = buffer.range(of: Self.thinkEnd)
          let toolRange = buffer.range(of: Self.toolCallsMarker)

          // A stray `[/THINK]` before any valid opener is stripped
          // from emitted content, matching vLLM's Mistral reasoning
          // Case 3.
          if let endRange,
             startRange == nil || endRange.lowerBound < startRange!.lowerBound,
             toolRange == nil || endRange.lowerBound < toolRange!.lowerBound
          {
            buffer.removeSubrange(endRange)
            continue
          }

          // Once a tool-call marker appears before any `[THINK]`,
          // reasoning did not start; let the regular Mistral scanner
          // handle content + tools from the current buffer.
          if let toolRange,
             startRange == nil || toolRange.lowerBound < startRange!.lowerBound
          {
            thinkPhase = .done
            continue
          }

          if let startRange {
            let prefix = String(buffer[buffer.startIndex ..< startRange.lowerBound])
            if !prefix.isEmpty {
              events.append(contentsOf: emitContent(prefix))
            }
            if openMessage != nil {
              events.append(contentsOf: closeMessage(status: .completed))
            }
            let consumed = buffer.distance(from: buffer.startIndex, to: startRange.upperBound)
            buffer.removeFirst(consumed)
            thinkPhase = .inThink
            continue
          }

          if !isEnd {
            let bufChars = Array(buffer)
            let overlap = [
              partialOverlap(suffixOf: bufChars, with: Array(Self.thinkStart)),
              partialOverlap(suffixOf: bufChars, with: Array(Self.thinkEnd)),
              partialOverlap(suffixOf: bufChars, with: Array(Self.toolCallsMarker)),
            ].max() ?? 0
            let safeEnd = bufChars.count - overlap
            if safeEnd > 0 {
              let chunk = String(bufChars[0 ..< safeEnd])
              events.append(contentsOf: emitContent(chunk))
              buffer.removeFirst(safeEnd)
            }
            return events
          }
          thinkPhase = .done

        case .inThink:
          if let endRange = buffer.range(of: Self.thinkEnd) {
            let text = String(buffer[buffer.startIndex ..< endRange.lowerBound])
            if !text.isEmpty {
              events.append(contentsOf: emitReasoningDelta(text: text))
            }
            let consumed = buffer.distance(from: buffer.startIndex, to: endRange.upperBound)
            buffer.removeFirst(consumed)
            events.append(contentsOf: closeReasoning(status: .completed))
            thinkPhase = .done
            continue
          }
          let bufChars = Array(buffer)
          let endChars = Array(Self.thinkEnd)
          let overlap = isEnd ? 0 : partialOverlap(suffixOf: bufChars, with: endChars)
          let safeEnd = bufChars.count - overlap
          if safeEnd > 0 {
            let chunk = String(bufChars[0 ..< safeEnd])
            events.append(contentsOf: emitReasoningDelta(text: chunk))
            buffer.removeFirst(safeEnd)
          }
          return events

        case .done:
          break
      }
    }
    return events
  }

  private mutating func emitReasoningDelta(text: String) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var events: [ResponseStreamingEvent] = []
    if openReasoning == nil {
      events.append(contentsOf: openReasoningItem())
    }
    if var r = openReasoning {
      r.emittedText += text
      openReasoning = r
      events.append(.reasoningTextDelta(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        delta: text,
        sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  private mutating func openReasoningItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.reasoning)
    let outputIndex = takeOutputIndex()
    openReasoning = OpenReasoning(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .reasoning(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex, sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id, outputIndex: outputIndex, contentIndex: 0,
        part: .reasoningText(.init(text: "")), sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeReasoning(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let r = openReasoning else { return [] }
    openReasoning = nil
    let part = ReasoningTextContent(text: r.emittedText)
    return [
      .reasoningTextDone(.init(
        itemId: r.id, outputIndex: r.outputIndex, contentIndex: 0,
        text: r.emittedText, sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: r.id, outputIndex: r.outputIndex, contentIndex: 0,
        part: .reasoningText(part), sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .reasoning(.init(id: r.id, content: [.reasoningText(part)], status: status)),
        outputIndex: r.outputIndex, sequenceNumber: takeSequence(),
      )),
    ]
  }

  // MARK: Scan

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    // Loop while progress is made: each iteration either emits content
    // up to the next marker, or consumes one tool call. When neither
    // is possible (waiting for more bytes), the loop exits.
    var didProgress = true
    while didProgress {
      didProgress = false
      let beforeIdx = parsedIdx

      events.append(contentsOf: emitContentUpToMarker(isEnd: isEnd))
      if parsedIdx > beforeIdx { didProgress = true }

      if let consumed = tryConsumeToolCall(isEnd: isEnd) {
        events.append(contentsOf: consumed)
        didProgress = true
      }
    }
    return events
  }

  private mutating func emitContentUpToMarker(isEnd: Bool) -> [ResponseStreamingEvent] {
    let bufChars = Array(buffer)
    let markerChars = Array(MistralParser.toolCallsMarker)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: MistralParser.toolCallsMarker, after: parsedIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: markerChars)
      sendableEnd = bufChars.count - overlap
    }
    guard sendableEnd > parsedIdx else { return [] }

    let chunk = String(bufChars[parsedIdx ..< sendableEnd])
    parsedIdx = sendableEnd
    return emitContent(chunk)
  }

  private mutating func emitContent(_ chunk: String) -> [ResponseStreamingEvent] {
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

  /// Try to consume a single tool call (compact or JSON-array form)
  /// starting at `parsedIdx`. Returns the events if a complete tool call
  /// was consumed, or nil when more bytes are needed (or the cursor is
  /// not at a marker).
  private mutating func tryConsumeToolCall(isEnd: Bool) -> [ResponseStreamingEvent]? {
    let bufChars = Array(buffer)
    guard parsedIdx < bufChars.count else { return nil }
    let markerChars = Array(MistralParser.toolCallsMarker)
    let suffixCount = bufChars.count - parsedIdx
    guard suffixCount >= markerChars.count,
          bufChars[parsedIdx ..< (parsedIdx + markerChars.count)].elementsEqual(markerChars)
    else { return nil }

    // Skip past `[TOOL_CALLS`. The literal `]` after it is optional.
    var i = parsedIdx + markerChars.count
    if i < bufChars.count, bufChars[i] == "]" {
      i += 1
    }
    // Skip whitespace.
    while i < bufChars.count, bufChars[i].isWhitespace {
      i += 1
    }
    if i >= bufChars.count { return nil }

    let next = bufChars[i]

    if next == "[" {
      // JSON-array form: `[{...}, ...]`.
      return tryConsumeJsonArray(arrayStart: i, isEnd: isEnd)
    } else {
      // Compact form: `name[ARGS]{...}`.
      return tryConsumeCompact(nameStart: i, isEnd: isEnd)
    }
  }

  /// Consume `[TOOL_CALLS]name[ARGS]{...}` (sglang) or `[TOOL_CALLS]name{...}`
  /// (vLLM v11+). When `[ARGS` is present before the first `{`, it
  /// delimits the name; otherwise the first `{` does. Returns nil if
  /// more bytes are needed; if `isEnd` and the JSON payload isn't
  /// complete, marks the call as incomplete.
  private mutating func tryConsumeCompact(nameStart: Int, isEnd: Bool) -> [ResponseStreamingEvent]? {
    let bufChars = Array(buffer)
    let argsChars = Array(MistralParser.argsMarker)

    let argsIdx = bufChars.firstIndexOf(substring: MistralParser.argsMarker, after: nameStart)
    let braceIdx: Int? = {
      var k = nameStart
      while k < bufChars.count {
        if bufChars[k] == "{" { return k }
        k += 1
      }
      return nil
    }()

    // Pick the earliest delimiter. vLLM's separator-less compact form
    // (`[TOOL_CALLS]get_weather{"city":"Paris"}`) lands in the `.none, .some`
    // branch; sglang's `[ARGS]` form lands in the first branch when
    // `[ARGS` precedes any `{` inside the args object.
    let nameEnd: Int
    let jsonStart: Int
    switch (argsIdx, braceIdx) {
      case let (.some(a), .some(b)) where a < b:
        nameEnd = a
        var k = a + argsChars.count
        if k < bufChars.count, bufChars[k] == "]" { k += 1 }
        while k < bufChars.count, bufChars[k].isWhitespace {
          k += 1
        }
        if k >= bufChars.count { return nil }
        jsonStart = k
      case let (_, .some(b)):
        nameEnd = b
        jsonStart = b
      case (.some, .none), (.none, .none):
        return nil
    }

    let name = String(bufChars[nameStart ..< nameEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
    if name.isEmpty { return nil }

    let opening = bufChars[jsonStart]
    guard opening == "{" || opening == "[" else { return nil }

    guard let (jsonText, endIdx, complete) = extractJSONValue(in: bufChars, from: jsonStart) else {
      return nil
    }
    if !complete, !isEnd { return nil }

    let events = emitToolCall(name: name, arguments: jsonText, status: complete ? .completed : .incomplete)
    parsedIdx = endIdx
    return events
  }

  private mutating func tryConsumeJsonArray(arrayStart: Int, isEnd: Bool) -> [ResponseStreamingEvent]? {
    let bufChars = Array(buffer)
    guard let (arrayText, endIdx, complete) = extractJSONValue(in: bufChars, from: arrayStart) else {
      return nil
    }
    if !complete {
      if !isEnd { return nil }
      // Truncated array at end-of-stream: we don't try to recover
      // a partial tool name from the unbalanced JSON – the
      // unparsed bytes flow through to `finalize()` and surface
      // as message content via `emitContent`. This matches both
      // vLLM (`extract_tool_calls` returns the raw text as
      // `content` on parse failure) and sglang (`detect_and_parse`
      // includes the trailing bytes in `combined_normal`).
      return nil
    }

    guard let data = arrayText.data(using: .utf8),
          let parsed = try? JSONSerialization.jsonObject(with: data)
    else {
      return nil
    }

    let calls: [[String: Any]]
    if let array = parsed as? [[String: Any]] {
      calls = array
    } else if let single = parsed as? [String: Any] {
      calls = [single]
    } else {
      return nil
    }

    var events: [ResponseStreamingEvent] = []
    for call in calls {
      guard let name = call["name"] as? String else { continue }
      let args = call["arguments"] ?? call["parameters"] ?? [:]
      let argsText: String = if let s = args as? String {
        s
      } else if let data = try? JSONSerialization.data(
        // `.sortedKeys` enforces deterministic key order; Foundation
        // dictionaries don't preserve insertion order, so without it
        // the serialized form would shuffle across calls. Diverges
        // from sglang/vLLM, which emit in declaration order via
        // Python dicts' insertion-order guarantee.
        withJSONObject: args, options: [.sortedKeys, .withoutEscapingSlashes],
      ), let s = String(data: data, encoding: .utf8) {
        s
      } else {
        "{}"
      }
      events.append(contentsOf: emitToolCall(name: name, arguments: argsText, status: .completed))
    }
    parsedIdx = endIdx
    return events
  }

  /// Extract a JSON value (object or array) starting at `start` from a
  /// character array, walking through string literals correctly.
  /// Returns (text, end-exclusive index, isComplete).
  private func extractJSONValue(
    in chars: [Character],
    from start: Int,
  ) -> (text: String, end: Int, complete: Bool)? {
    guard start < chars.count else { return nil }
    let opening = chars[start]
    guard opening == "{" || opening == "[" else { return nil }
    let closing: Character = opening == "{" ? "}" : "]"
    var depth = 0
    var inString = false
    var escape = false
    for i in start ..< chars.count {
      let c = chars[i]
      if inString {
        if escape { escape = false; continue }
        if c == "\\" { escape = true; continue }
        if c == "\"" { inString = false }
        continue
      }
      if c == "\"" { inString = true; continue }
      if c == opening { depth += 1 }
      else if c == closing {
        depth -= 1
        if depth == 0 {
          return (String(chars[start ..< i + 1]), i + 1, true)
        }
      }
    }
    // Unbalanced: return what we have so far as incomplete.
    return (String(chars[start ..< chars.count]), chars.count, false)
  }

  private mutating func emitToolCall(
    name: String,
    arguments: String,
    status: ItemStatus,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    let id = IDFactory.make(.functionCall)
    // Mistral chat templates enforce a 9-character alphanumeric
    // `tool_call.id` and insert it verbatim into the rendered prompt.
    // The generic `call_…` shape would raise a Jinja exception when
    // history is rendered through the upstream Mistral templates, so
    // mint a format-appropriate ID here. Mirrors vLLM's
    // `MistralToolCall.generate_random_id`.
    let callId = IDFactory.makeMistralStrict()
    let outputIndex = takeOutputIndex()
    let openItem = ResponseFunctionToolCall(
      id: id,
      callId: callId,
      name: name,
      arguments: "",
      status: .inProgress,
    )
    let doneItem = ResponseFunctionToolCall(
      id: id,
      callId: callId,
      name: name,
      arguments: arguments,
      status: status,
    )
    events.append(.outputItemAdded(.init(
      item: .functionCall(openItem),
      outputIndex: outputIndex,
      sequenceNumber: takeSequence(),
    )))
    if !arguments.isEmpty {
      events.append(.functionCallArgumentsDelta(.init(
        itemId: id,
        outputIndex: outputIndex,
        delta: arguments,
        sequenceNumber: takeSequence(),
      )))
    }
    events.append(.functionCallArgumentsDone(.init(
      itemId: id,
      outputIndex: outputIndex,
      name: name,
      arguments: arguments,
      sequenceNumber: takeSequence(),
    )))
    events.append(.outputItemDone(.init(
      item: .functionCall(doneItem),
      outputIndex: outputIndex,
      sequenceNumber: takeSequence(),
    )))
    toolCalls.append(OpenToolCall(
      id: id, callId: callId, outputIndex: outputIndex, name: name, arguments: arguments,
    ))
    return events
  }

  // MARK: Open/close helpers

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
