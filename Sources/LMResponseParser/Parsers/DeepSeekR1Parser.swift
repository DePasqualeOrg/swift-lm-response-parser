// Copyright © Anthony DePasquale

import Foundation

/// Parser for the DeepSeek R1 family.
///
/// **Wire shape.** Reasoning content is wrapped in `<think>` … `</think>`
/// and always appears before any tool call. Tool calls follow the
/// DeepSeek V3 protocol, which uses CJK-bracket delimiters and embeds
/// arguments as a fenced JSON code block:
///
/// ```text
/// <｜tool▁calls▁begin｜>
/// <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
/// ```json
/// {"arg": "value"}
/// ```<｜tool▁call▁end｜>
/// <｜tool▁calls▁end｜>
/// ```
///
/// Multiple tool calls are emitted in sequence inside the
/// `<｜tool▁calls▁begin｜>` … `<｜tool▁calls▁end｜>` envelope, each in its
/// own `<｜tool▁call▁begin｜>` … `<｜tool▁call▁end｜>` block. Plain text
/// outside the envelope is normal message content.
///
/// **Reasoning by default.** R1 always begins in reasoning mode, mirroring
/// SGLang's `DeepSeekR1Detector` (`force_reasoning=True`) and vLLM's
/// `DeepSeekR1ReasoningParser`. The original DeepSeek-R1 release emits
/// reasoning text directly without a leading `<think>`; only R1-0528
/// emits the opener. Both shapes are handled by the default
/// ``InitialState/reasoning`` – when `<think>` (optionally preceded by
/// whitespace) is present at the start of the stream, it's stripped;
/// otherwise content streams as reasoning until `</think>`. Pass
/// ``InitialState/normal`` to skip reasoning entirely (e.g. when the
/// caller has already extracted reasoning from prior output).
///
/// **Nemotron V3 swap mode.** Pass `swapWhenContentEmpty: true` to
/// enable the Nemotron V3 quirk: when the model emits no `</think>`
/// (the chat template ran with `enable_thinking=False` /
/// `force_nonempty_content=True`), the buffered text is replayed as a
/// message item instead of a reasoning item. Mirrors vLLM's
/// `NemotronV3ReasoningParser`. Cost: in this mode, scanReasoning
/// holds output until the exit-marker decision is reached, so the
/// consumer sees the deltas in a single batch at the resolve point
/// rather than as they arrive.
struct DeepSeekR1Parser: ResponseFormatParser {
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let thinkStart = "<think>"
  private static let thinkEnd = "</think>"
  private static let toolCallsBegin = "<｜tool▁calls▁begin｜>"
  private static let toolCallsEnd = "<｜tool▁calls▁end｜>"
  private static let toolCallBegin = "<｜tool▁call▁begin｜>"
  private static let toolCallEnd = "<｜tool▁call▁end｜>"
  private static let toolSep = "<｜tool▁sep｜>"
  /// Opening fence for the JSON code block, including the trailing
  /// newline that separates the fence from the JSON body. Tracking the
  /// newline as part of the fence (rather than skipping it after the
  /// fact) makes streaming behave: we don't open the function-call item
  /// until the whole opener is present, so the args stream never
  /// contains the leading newline.
  private static let jsonFenceOpen = "```json\n"
  /// Closing fence is `\n``` `` ``` ``. Treating the leading newline as
  /// part of the fence means streaming args don't contain the trailing
  /// newline either.
  private static let jsonFenceClose = "\n```"

  private var buffer: String = ""

  private enum Phase { case reasoning, normal }
  private var phase: Phase

  private var sentReasoningIdx: Int = 0
  private var parsedIdx: Int = 0

  private var openReasoning: OpenReasoning?
  private var openMessage: OpenMessage?
  private var toolCalls: [OpenToolCall] = []

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  /// Whether the cursor is currently between `<｜tool▁calls▁begin｜>`
  /// and the matching `<｜tool▁calls▁end｜>`. Used to suppress text
  /// emission for whitespace between individual tool calls.
  private var insideToolCallsEnvelope: Bool = false

  /// Whether the cursor is currently between `<｜tool▁call▁begin｜>` and
  /// the matching `<｜tool▁call▁end｜>` (one specific call's payload).
  private var insideSingleToolCall: Bool = false

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
    /// Allocated lazily, once the name has been parsed and we're
    /// about to emit `output_item.added`. Truncation between
    /// `<｜tool▁call▁begin｜>` and the name leaves this nil so no slot
    /// is consumed and the next item's index stays consecutive.
    /// Mirrors DeepSeekV3Parser's pattern.
    var outputIndex: Int?
    var name: String?
    /// Cumulative arguments emitted so far (just the JSON body inside
    /// the fenced code block). Streaming deltas equal the difference
    /// between successive scans.
    var argsEmitted: String = ""
    var closed: Bool = false
  }

  /// When true, defer reasoning emission until either an exit marker
  /// arrives (`</think>` or `<｜tool▁calls▁begin｜>`) or finalize is
  /// called. Reasoning-state text is held in ``heldReasoningText``;
  /// at finalize without an exit marker the held text is flushed as a
  /// message item, matching Nemotron V3's content/reasoning swap.
  private let swapWhenContentEmpty: Bool

  /// Held text accumulated during reasoning state when
  /// ``swapWhenContentEmpty`` is on. Cleared once flushed.
  private var heldReasoningText: String = ""

  /// True when ``swapWhenContentEmpty`` is on and the parser still
  /// hasn't decided whether the prefix is reasoning or content.
  private var holdingReasoning: Bool

  init(
    initialState: InitialState = .reasoning,
    swapWhenContentEmpty: Bool = false,
  ) {
    switch initialState {
      case .normal: phase = .normal
      case .reasoning: phase = .reasoning
    }
    self.swapWhenContentEmpty = swapWhenContentEmpty
    holdingReasoning = swapWhenContentEmpty && initialState == .reasoning
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return scan(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = scan(isEnd: true)
    // Nemotron V3 swap: if the stream ended without an exit marker
    // proving the held text was reasoning, flush it as a message
    // instead. The `</think>` exit-marker path in scanReasoning has
    // already cleared `holdingReasoning` if it fired.
    if holdingReasoning {
      events.append(contentsOf: flushHeldReasoningAsMessage())
    }
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

    let thinkStartChars = Array(DeepSeekR1Parser.thinkStart)
    let thinkEndChars = Array(DeepSeekR1Parser.thinkEnd)
    let toolBeginChars = Array(DeepSeekR1Parser.toolCallsBegin)

    if sentReasoningIdx == 0 {
      // Tolerate leading whitespace before an optional `<think>`
      // opener. R1-0528 emits `<think>`; the original R1 emits
      // reasoning content directly with no opener. If `<think>`
      // (optionally preceded by whitespace) is present, both the
      // whitespace and the marker are skipped so the reasoning
      // delta starts at the first real character.
      var cursor = 0
      while cursor < bufChars.count, bufChars[cursor].isWhitespace {
        cursor += 1
      }
      let available = bufChars.count - cursor
      if cursor == bufChars.count {
        // Buffer is whitespace-only: hold for more bytes so we
        // can decide whether `<think>` follows.
        if !isEnd { return events }
        sentReasoningIdx = bufChars.count
      } else if available >= thinkStartChars.count,
                Array(bufChars[cursor ..< cursor + thinkStartChars.count]) == thinkStartChars
      {
        sentReasoningIdx = cursor + thinkStartChars.count
      } else if available < thinkStartChars.count, !isEnd,
                Array(bufChars[cursor ..< bufChars.count]) == Array(thinkStartChars[..<available])
      {
        // Buffer ends mid-`<think>` after whitespace: hold.
        return events
      }
      // Otherwise: not a `<think>` opener. Leave sentReasoningIdx
      // at 0 so the model's reasoning content (R1 base) streams
      // from byte 0, including any leading characters.
    }

    let endIdx = bufChars.firstIndexOf(substring: DeepSeekR1Parser.thinkEnd, after: sentReasoningIdx)
    let toolIdx = bufChars.firstIndexOf(substring: DeepSeekR1Parser.toolCallsBegin, after: sentReasoningIdx)

    let exitIdx: Int?
    let exitMarker: ExitMarker?
    switch (endIdx, toolIdx) {
      case let (.some(e), .some(t)) where e <= t:
        exitIdx = e; exitMarker = .thinkEnd
      case (.some, .some):
        exitIdx = toolIdx; exitMarker = .toolCalls
      case (.some, .none):
        exitIdx = endIdx; exitMarker = .thinkEnd
      case (.none, .some):
        exitIdx = toolIdx; exitMarker = .toolCalls
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
      let toolOverlap = partialOverlap(suffixOf: bufChars, with: toolBeginChars)
      safeEnd = bufChars.count - Swift.max(endOverlap, toolOverlap)
    }

    if safeEnd > sentReasoningIdx {
      let chunk = String(bufChars[sentReasoningIdx ..< safeEnd])
      if !chunk.isEmpty {
        if holdingReasoning {
          // Nemotron V3 swap mode: buffer until an exit marker (or
          // finalize) decides whether this is reasoning or content.
          heldReasoningText += chunk
        } else {
          if openReasoning == nil {
            events.append(contentsOf: openReasoningItem())
          }
          if var r = openReasoning {
            r.emittedText += chunk
            openReasoning = r
            events.append(.reasoningTextDelta(.init(
              itemId: r.id,
              outputIndex: r.outputIndex,
              contentIndex: 0,
              delta: chunk,
              sequenceNumber: takeSequence(),
            )))
          }
        }
      }
      sentReasoningIdx = safeEnd
    }

    guard let exitIdx, exitIdx == safeEnd, let exitMarker else { return events }
    // An exit marker arrived – we now know the held prefix is reasoning,
    // so flush it as such before transitioning out of reasoning state.
    if holdingReasoning {
      events.append(contentsOf: flushHeldReasoningAsReasoning())
    }
    switch exitMarker {
      case .thinkEnd:
        parsedIdx = exitIdx + thinkEndChars.count
      case .toolCalls:
        parsedIdx = exitIdx
    }
    events.append(contentsOf: closeReasoning(status: .completed))
    phase = .normal
    return events
  }

  /// Flush ``heldReasoningText`` as a reasoning item (open + delta +
  /// close). Called when an exit marker proves the held prefix was
  /// thinking, not content.
  private mutating func flushHeldReasoningAsReasoning() -> [ResponseStreamingEvent] {
    holdingReasoning = false
    let held = heldReasoningText
    heldReasoningText = ""
    guard !held.isEmpty else { return [] }
    var events = openReasoningItem()
    if var r = openReasoning {
      r.emittedText += held
      openReasoning = r
      events.append(.reasoningTextDelta(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        delta: held,
        sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  /// Flush ``heldReasoningText`` as a message item. Called by
  /// ``finalize()`` when the stream ended without an exit marker,
  /// triggering the Nemotron V3 swap.
  private mutating func flushHeldReasoningAsMessage() -> [ResponseStreamingEvent] {
    holdingReasoning = false
    let held = heldReasoningText
    heldReasoningText = ""
    guard !held.isEmpty else { return [] }
    var events = openMessageItem()
    if var msg = openMessage {
      msg.emittedText += held
      openMessage = msg
      events.append(.outputTextDelta(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        delta: held,
        sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  private enum ExitMarker { case thinkEnd, toolCalls }

  // MARK: Normal phase: cursor-based tag walker

  private mutating func scanNormal(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    if openMessage == nil, openReasoning == nil, toolCalls.isEmpty {
      if let _ = transitionToReasoningIfMarkerPresent(isEnd: isEnd) {
        return events
      }
    }

    events.append(contentsOf: emitNormalText(isEnd: isEnd))

    // Walk structural markers and emit deltas for each completed call.
    while parsedIdx < buffer.count {
      let slice = buffer.dropFirst(parsedIdx)

      if slice.hasPrefix(DeepSeekR1Parser.toolCallsBegin) {
        parsedIdx += DeepSeekR1Parser.toolCallsBegin.count
        insideToolCallsEnvelope = true
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        continue
      }
      if slice.hasPrefix(DeepSeekR1Parser.toolCallsEnd) {
        parsedIdx += DeepSeekR1Parser.toolCallsEnd.count
        insideToolCallsEnvelope = false
        continue
      }
      if slice.hasPrefix(DeepSeekR1Parser.toolCallBegin) {
        parsedIdx += DeepSeekR1Parser.toolCallBegin.count
        insideSingleToolCall = true
        // outputIndex is allocated lazily in `parseFunctionHeader`
        // once the name is known, so a truncated header doesn't
        // burn an output slot.
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
        ))
        continue
      }
      if slice.hasPrefix(DeepSeekR1Parser.toolCallEnd) {
        parsedIdx += DeepSeekR1Parser.toolCallEnd.count
        insideSingleToolCall = false
        if let index = toolCalls.indices.last, !toolCalls[index].closed,
           toolCalls[index].name != nil
        {
          events.append(contentsOf: closeToolCall(at: index, status: .completed))
        }
        continue
      }

      if insideSingleToolCall, let last = toolCalls.indices.last, !toolCalls[last].closed {
        if toolCalls[last].name == nil {
          // Looking for `function<｜tool▁sep｜>NAME\n` then `\n```json\n` to start args.
          let advanced = parseFunctionHeader(at: parsedIdx, isEnd: isEnd, callIndex: last, events: &events)
          if advanced { continue } else { return events }
        } else {
          // Inside the JSON code block. Consume up to the closing fence.
          let advanced = parseArguments(at: parsedIdx, isEnd: isEnd, callIndex: last, events: &events)
          if advanced { continue } else { return events }
        }
      }

      if insideToolCallsEnvelope {
        if slice.first == "<", couldStillBecomeATag(slice: String(slice)), !isEnd {
          return events
        }
        if let nextLt = slice.firstIndex(of: "<") {
          let offset = slice.distance(from: slice.startIndex, to: nextLt)
          if offset == 0 {
            // We landed back on a `<` that already failed every
            // tag prefix and the `couldStillBecomeATag` check
            // above. Advance past it so the loop makes
            // progress.
            parsedIdx += 1
          } else {
            // Advance up to the next `<` and let the next
            // iteration's prefix checks decide whether it
            // begins a tag.
            parsedIdx += offset
          }
          if parsedIdx >= buffer.count { return events }
          continue
        } else {
          parsedIdx += slice.count
          continue
        }
      }

      break
    }

    return events
  }

  /// Parse `function<｜tool▁sep｜>NAME\n` followed by `` ```json ``.
  /// Returns true when the header was fully consumed and the cursor
  /// advanced; false when more bytes are needed.
  private mutating func parseFunctionHeader(
    at start: Int,
    isEnd _: Bool,
    callIndex: Int,
    events: inout [ResponseStreamingEvent],
  ) -> Bool {
    let bufChars = Array(buffer)
    let sepChars = Array(DeepSeekR1Parser.toolSep)

    // Find the separator anywhere from `start` forward (the leading
    // "function" or other prefix is informational and can be ignored).
    guard let sepIdx = bufChars.firstIndexOf(substring: DeepSeekR1Parser.toolSep, after: start) else {
      return false
    }
    let nameStart = sepIdx + sepChars.count

    // Name runs to the next newline.
    var nameEnd: Int? = nil
    var i = nameStart
    while i < bufChars.count {
      if bufChars[i] == "\n" { nameEnd = i; break }
      i += 1
    }
    guard let nameEnd else { return false }
    // sglang's deepseekv3_detector.py strips the captured name. Swift
    // V3.1 already trims; R1 originally didn't, which surfaced
    // whitespace inside `<｜tool▁sep｜> name \n` to consumers as part
    // of the function name. Trim to match.
    let name = String(bufChars[nameStart ..< nameEnd])
      .trimmingCharacters(in: .whitespacesAndNewlines)

    // Find the opening JSON fence after the name. The fence already
    // includes its trailing newline, so when this match succeeds the
    // body cursor is positioned exactly at the first JSON character.
    let fenceOpenChars = Array(DeepSeekR1Parser.jsonFenceOpen)
    guard let fenceIdx = bufChars.firstIndexOf(substring: DeepSeekR1Parser.jsonFenceOpen, after: nameEnd) else {
      return false
    }
    let argsStart = fenceIdx + fenceOpenChars.count

    if name.isEmpty {
      // Drop malformed nameless calls before allocating an output slot.
      // This mirrors the defensive V3.1/V3.2 behavior and keeps
      // downstream dispatch from seeing an unusable function call.
      if callIndex < toolCalls.count {
        toolCalls.remove(at: callIndex)
      }
      insideSingleToolCall = false
      parsedIdx = argsStart
      return true
    }

    // Open the function-call item. Allocate the outputIndex now –
    // earlier truncation paths leave it nil so the slot is recycled.
    var call = toolCalls[callIndex]
    call.name = name
    let outputIndex = takeOutputIndex()
    call.outputIndex = outputIndex
    toolCalls[callIndex] = call

    events.append(.outputItemAdded(.init(
      item: .functionCall(.init(
        id: call.id,
        callId: call.callId,
        name: name,
        arguments: "",
        status: .inProgress,
      )),
      outputIndex: outputIndex,
      sequenceNumber: takeSequence(),
    )))

    parsedIdx = argsStart
    return true
  }

  /// Stream argument bytes from the current cursor up to the next
  /// closing fence. The closing fence is `\n```` (newline, three
  /// backticks). Emits a delta for any new chars and, when the fence is
  /// found, advances the cursor past it.
  private mutating func parseArguments(
    at start: Int,
    isEnd: Bool,
    callIndex: Int,
    events: inout [ResponseStreamingEvent],
  ) -> Bool {
    let bufChars = Array(buffer)
    let fenceCloseChars = Array(DeepSeekR1Parser.jsonFenceClose)
    let endTokenChars = Array(DeepSeekR1Parser.toolCallEnd)

    let fenceIdx = bufChars.firstIndexOf(substring: DeepSeekR1Parser.jsonFenceClose, after: start)
    let endIdx = bufChars.firstIndexOf(substring: DeepSeekR1Parser.toolCallEnd, after: start)

    let stopIdx: Int?
    let stopIsFence: Bool
    switch (fenceIdx, endIdx) {
      case let (.some(f), .some(e)) where f <= e:
        stopIdx = f; stopIsFence = true
      case (.some, .some):
        stopIdx = endIdx; stopIsFence = false
      case (.some, .none):
        stopIdx = fenceIdx; stopIsFence = true
      case (.none, .some):
        stopIdx = endIdx; stopIsFence = false
      case (.none, .none):
        stopIdx = nil; stopIsFence = false
    }

    let safeEnd: Int
    if let stopIdx {
      safeEnd = stopIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let fenceOverlap = partialOverlap(suffixOf: bufChars, with: fenceCloseChars)
      let endOverlap = partialOverlap(suffixOf: bufChars, with: endTokenChars)
      safeEnd = bufChars.count - Swift.max(fenceOverlap, endOverlap)
    }

    let newArgs = String(bufChars[start ..< safeEnd])
    if !newArgs.isEmpty {
      var call = toolCalls[callIndex]
      call.argsEmitted += newArgs
      toolCalls[callIndex] = call
      if let outputIndex = call.outputIndex {
        events.append(.functionCallArgumentsDelta(.init(
          itemId: call.id,
          outputIndex: outputIndex,
          delta: newArgs,
          sequenceNumber: takeSequence(),
        )))
      }
    }

    guard let stopIdx, stopIdx == safeEnd else {
      parsedIdx = safeEnd
      return false
    }
    if stopIsFence {
      parsedIdx = stopIdx + fenceCloseChars.count
    } else {
      parsedIdx = stopIdx
    }
    return true
  }

  private mutating func emitNormalText(isEnd: Bool) -> [ResponseStreamingEvent] {
    if insideToolCallsEnvelope { return [] }

    let bufChars = Array(buffer)
    let beginChars = Array(DeepSeekR1Parser.toolCallsBegin)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: DeepSeekR1Parser.toolCallsBegin, after: parsedIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: beginChars)
      sendableEnd = bufChars.count - overlap
    }
    guard sendableEnd > parsedIdx else { return [] }

    let chunk = String(bufChars[parsedIdx ..< sendableEnd])
    parsedIdx = sendableEnd
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
    let thinkStartChars = Array(DeepSeekR1Parser.thinkStart)
    // Skip leading whitespace so a buffer like `   <think>` still
    // routes to the reasoning phase. The whitespace itself gets
    // consumed as part of the marker boundary, matching how the
    // reference parsers strip pre-`<think>` formatting.
    var cursor = parsedIdx
    while cursor < bufChars.count, bufChars[cursor].isWhitespace {
      cursor += 1
    }
    let available = bufChars.count - cursor

    if available >= thinkStartChars.count {
      if Array(bufChars[cursor ..< cursor + thinkStartChars.count]) == thinkStartChars {
        parsedIdx = cursor + thinkStartChars.count
        sentReasoningIdx = parsedIdx
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
    // If only whitespace has arrived so far and the stream isn't
    // closed, hold back to see if `<think>` follows.
    if available == 0, cursor > parsedIdx, !isEnd {
      return []
    }
    return nil
  }

  private func couldStillBecomeATag(slice: String) -> Bool {
    for tag in [
      DeepSeekR1Parser.toolCallsBegin,
      DeepSeekR1Parser.toolCallsEnd,
      DeepSeekR1Parser.toolCallBegin,
      DeepSeekR1Parser.toolCallEnd,
      DeepSeekR1Parser.toolSep,
    ] {
      if tag.hasPrefix(slice) { return true }
    }
    return false
  }

  // MARK: Item open/close

  private mutating func closeToolCall(at index: Int, status: ItemStatus) -> [ResponseStreamingEvent] {
    var call = toolCalls[index]
    guard !call.closed, let name = call.name, let outputIndex = call.outputIndex else {
      return []
    }
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
        name: name,
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
      .reasoningTextDone(.init(
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
