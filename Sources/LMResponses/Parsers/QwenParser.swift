// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Qwen 2.5 / Qwen 3 base format.
///
/// **Wire shape.** Tool calls use the Hermes-style `<tool_call>` …
/// `</tool_call>` envelope with a JSON object inside. Reasoning content is
/// wrapped in `<think>` … `</think>` and always appears before any tool
/// call. A typical response looks like:
///
/// ```text
/// <think>The user wants the weather.</think>
/// I'll look that up.
/// <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
/// ```
///
/// **Reasoning phase.** Two scenarios produce reasoning content:
///
/// 1. Older Qwen 3 templates put `<think>` in the model output; the parser
///    strips the leading marker and accumulates reasoning until `</think>`.
/// 2. Newer Qwen 3 / Qwen 3.5 templates put `<think>` in the prompt itself,
///    so the model output starts already inside reasoning. Construct the
///    parser with ``InitialState/reasoning`` (the factory does this when
///    `priorOutput` contains an unclosed reasoning marker) and only
///    `</think>` appears in the stream.
///
/// **Implicit reasoning end.** Qwen 3.5 may emit `<tool_call>` without a
/// preceding `</think>`. The parser treats `<tool_call>` while in the
/// reasoning phase as an implicit reasoning close, then begins the tool
/// call as in the Hermes path.
///
/// **Qwen 2.5.** Models without a reasoning phase never emit `<think>`. The
/// parser stays in the normal phase from the start and behaves like the
/// Hermes parser.
struct QwenParser: ResponseFormatParser {
  /// Initial reasoning phase. Set to ``reasoning`` when the parser should
  /// start already inside a `<think>` block (typically because the chat
  /// template placed the opener in the prompt).
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let thinkStart = "<think>"
  private static let thinkEnd = "</think>"
  private static let toolCallStart = "<tool_call>"
  private static let toolCallEnd = "</tool_call>"

  // Active accumulated output. Consumed prefixes are pruned after each scan.
  private var buffer: String = ""

  private enum Phase { case reasoning, normal }
  private var phase: Phase

  /// Whether the beginning of the current reasoning block has already been
  /// checked for an optional model-emitted `<think>` marker. Kept separate
  /// from `sentReasoningIdx` so buffer pruning can rebase the cursor to zero
  /// without making a later literal `<think>` look like a fresh opener.
  private var reasoningStartResolved: Bool = false

  /// True until the normal phase emits content or sees a tool-call region.
  /// Closed tool-call slots used to preserve this "start of normal output"
  /// fact implicitly; pruning removes those slots, so keep the fact directly.
  private var normalPhaseCanStartReasoning: Bool = true

  /// In reasoning phase: index in `buffer` of the next character to
  /// classify as reasoning text. Advances past the leading `<think>`
  /// marker if the model emits one.
  private var sentReasoningIdx: Int = 0

  /// In normal phase: index in `buffer` of the next character to classify
  /// as content or as the start of a tool-call region. Initialized when
  /// the reasoning phase ends; equals the position right after `</think>`,
  /// or the position of the implicit-end `<tool_call>`.
  private var sentContentIdx: Int = 0

  /// Start of the normal-phase slice that may contain tool-call regions.
  /// Unlike `sentContentIdx`, this does not advance after each closed
  /// tool call; keeping the scan base stable preserves region indexes
  /// when later tool calls arrive in separate chunks.
  private var toolRegionScanIdx: Int = 0

  private var openReasoning: OpenReasoning?
  private var openMessage: OpenMessage?
  private var toolCalls: [OpenToolCall] = []

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

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
    /// Allocated lazily, once the tool name has been parsed and we're
    /// about to emit `output_item.added`. Truncation before the name
    /// arrives leaves this nil so no slot is consumed and the next
    /// item's index stays consecutive. Mirrors HermesParser.
    var outputIndex: Int?
    var name: String?
    var argsEmitted: String = ""
    var closed: Bool = false
  }

  init(initialState: InitialState = .normal) {
    switch initialState {
      case .normal:
        phase = .normal
      case .reasoning:
        phase = .reasoning
    }
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return scan(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = scan(isEnd: true)
    // Anything still open at EOS gets flushed and closed.
    if openReasoning != nil {
      // Reasoning ran to end of stream without seeing `</think>` or a
      // tool call. Treat the open reasoning item as truncated content.
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

    // Loop while the phase keeps changing. A single chunk can carry
    // a normal → reasoning transition (older Qwen 3 templates emit
    // `<think>`) and then a reasoning → normal transition (the
    // matching `</think>` later in the same chunk), so two iterations
    // are possible. Once the phase stabilizes for an iteration, the
    // scan is done for this call.
    var lastPhase: Phase? = nil
    while lastPhase != phase {
      lastPhase = phase
      switch phase {
        case .reasoning:
          let (reasoningEvents, _) = scanReasoning(isEnd: isEnd)
          events.append(contentsOf: reasoningEvents)
        case .normal:
          events.append(contentsOf: scanNormal(isEnd: isEnd))
      }
    }

    pruneConsumedPrefix()
    return events
  }

  // MARK: Reasoning phase

  /// Process reasoning-phase input. Returns `(events, exited)` where
  /// `exited` is true when reasoning ended in this scan, signaling the
  /// outer loop to immediately drive the normal phase as well.
  private mutating func scanReasoning(isEnd: Bool) -> ([ResponseStreamingEvent], Bool) {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)

    let thinkStartChars = Array(QwenParser.thinkStart)
    let thinkEndChars = Array(QwenParser.thinkEnd)
    let toolStartChars = Array(QwenParser.toolCallStart)

    // Strip a leading `<think>` if the model emitted one (older Qwen 3
    // templates). Resolve this once per reasoning block; after pruning,
    // `sentReasoningIdx` may be zero again while reasoning is still open.
    if !reasoningStartResolved {
      if sentReasoningIdx == 0,
         bufChars.count >= thinkStartChars.count,
         Array(bufChars[0 ..< thinkStartChars.count]) == thinkStartChars
      {
        sentReasoningIdx = thinkStartChars.count
        reasoningStartResolved = true
      } else if sentReasoningIdx == 0, !isEnd {
        // Buffer might still grow into a leading `<think>` – hold back
        // any partial prefix so we don't accidentally emit `<thi` as
        // reasoning text and then realize it was a marker.
        let leadingOverlap = leadingPartialOverlap(of: bufChars, with: thinkStartChars)
        if leadingOverlap > 0, leadingOverlap == bufChars.count {
          return (events, false)
        }
        reasoningStartResolved = true
      } else {
        reasoningStartResolved = true
      }
    }

    // Find the next phase-ending marker: either `</think>` (explicit) or
    // `<tool_call>` (implicit reasoning end). The earlier marker wins.
    let endIdx = bufChars.firstIndexOf(substring: QwenParser.thinkEnd, after: sentReasoningIdx)
    let toolIdx = bufChars.firstIndexOf(substring: QwenParser.toolCallStart, after: sentReasoningIdx)

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
      // No marker yet – hold back any suffix that could grow into one.
      let endOverlap = partialOverlap(suffixOf: bufChars, with: thinkEndChars)
      let toolOverlap = partialOverlap(suffixOf: bufChars, with: toolStartChars)
      safeEnd = bufChars.count - Swift.max(endOverlap, toolOverlap)
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

    guard let exitIdx, exitIdx == safeEnd, let exitMarker else {
      return (events, false)
    }

    // Close reasoning and transition to normal phase.
    switch exitMarker {
      case .thinkEnd:
        // Skip past `</think>`; normal-phase content begins right after.
        sentContentIdx = exitIdx + thinkEndChars.count
      case .toolCall:
        // Don't skip – the normal-phase scanner needs to see `<tool_call>`
        // at this position to open the tool call.
        sentContentIdx = exitIdx
    }
    toolRegionScanIdx = sentContentIdx
    events.append(contentsOf: closeReasoning(status: .completed))
    reasoningStartResolved = false
    phase = .normal
    return (events, true)
  }

  private enum ExitMarker { case thinkEnd, toolCall }

  /// Length of the longest prefix of `tag` that the buffer's leading
  /// characters could still grow into (i.e., the buffer is so far a strict
  /// prefix of `tag`). 0 means no overlap.
  private func leadingPartialOverlap(of chars: [Character], with tag: [Character]) -> Int {
    let limit = Swift.min(chars.count, tag.count - 1)
    if limit <= 0 { return 0 }
    var k = limit
    while k > 0 {
      if chars[..<k].elementsEqual(tag[..<k]) {
        return k
      }
      k -= 1
    }
    return 0
  }

  // MARK: Normal phase (Hermes-style)

  private mutating func scanNormal(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    // Older Qwen 3 templates have the model emit `<think>` itself. If we
    // see it before any content / tool call has been opened, transition
    // back to reasoning phase. (Only honored at the very start of
    // normal-phase scanning – once content or a tool has been emitted,
    // a stray `<think>` would just be content.)
    if normalPhaseCanStartReasoning, openMessage == nil, openReasoning == nil {
      if let transitionEvents = transitionToReasoningIfMarkerPresent(isEnd: isEnd) {
        return transitionEvents
      }
    }

    // Plain content is flushed before each region – including between
    // consecutive `<tool_call>...</tool_call>` blocks in the same chunk
    // – and once more after the loop for trailing text.
    let regions = extractToolCallRegions()
    if !regions.isEmpty {
      normalPhaseCanStartReasoning = false
    }
    for (index, region) in regions.enumerated() {
      events.append(contentsOf: flushContent(isEnd: isEnd))
      if index >= toolCalls.count {
        // First sighting of this region – track it without taking
        // an output_index slot yet. Allocation happens lazily once
        // the name is parsed (see `processRegion`). Mirrors Hermes.
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
        ))
      }
      events.append(contentsOf: processRegion(at: index, region: region, isEnd: isEnd))
    }
    events.append(contentsOf: flushContent(isEnd: isEnd))

    return events
  }

  /// If the buffer at `sentContentIdx` starts with (or could complete
  /// into) `<think>`, transition to reasoning phase. Returns an empty
  /// event list when the transition happened, or nil when the buffer
  /// does not begin with `<think>`.
  private mutating func transitionToReasoningIfMarkerPresent(isEnd: Bool) -> [ResponseStreamingEvent]? {
    let bufChars = Array(buffer)
    let thinkStartChars = Array(QwenParser.thinkStart)
    let cursor = sentContentIdx
    let available = bufChars.count - cursor

    if available >= thinkStartChars.count {
      // Enough buffer to decide.
      if Array(bufChars[cursor ..< cursor + thinkStartChars.count]) == thinkStartChars {
        sentReasoningIdx = cursor + thinkStartChars.count
        reasoningStartResolved = true
        phase = .reasoning
        return []
      }
      return nil
    }

    // Buffer too short to confirm – check if what we have is a strict
    // prefix of `<think>`. If so, hold (return empty events but don't
    // commit to either phase). If not, the buffer cannot grow into
    // `<think>` and we proceed with normal processing.
    if available > 0 {
      let slice = Array(bufChars[cursor ..< bufChars.count])
      if Array(thinkStartChars[0 ..< slice.count]) == slice {
        if !isEnd {
          return [] // hold and wait for more
        }
      }
    }
    return nil
  }

  private mutating func flushContent(isEnd: Bool) -> [ResponseStreamingEvent] {
    let bufChars = Array(buffer)
    let toolStartChars = Array(QwenParser.toolCallStart)
    let toolEndChars = Array(QwenParser.toolCallEnd)
    let thinkStartChars = Array(QwenParser.thinkStart)
    let thinkEndChars = Array(QwenParser.thinkEnd)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: QwenParser.toolCallStart, after: sentContentIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      // Hold back any partial-tag overlap on either marker so a
      // chunk ending in `<tool_cal` or `</tool_cal` doesn't leak
      // raw bytes that would later complete to a real marker.
      // `<think>` / `</think>` are also held back so Trinity-style
      // tool-call-inside-`<think>` input doesn't leak fragments of
      // the closing think marker as message text.
      // Mirrors sglang's `_clean_normal_text` close-tag handling.
      let openOverlap = partialOverlap(suffixOf: bufChars, with: toolStartChars)
      let closeOverlap = partialOverlap(suffixOf: bufChars, with: toolEndChars)
      let thinkOpenOverlap = partialOverlap(suffixOf: bufChars, with: thinkStartChars)
      let thinkCloseOverlap = partialOverlap(suffixOf: bufChars, with: thinkEndChars)
      sendableEnd = bufChars.count - Swift.max(
        openOverlap, closeOverlap, thinkOpenOverlap, thinkCloseOverlap,
      )
    }
    guard sendableEnd > sentContentIdx else { return [] }

    var chunk = String(bufChars[sentContentIdx ..< sendableEnd])
    sentContentIdx = sendableEnd
    normalPhaseCanStartReasoning = false
    // Strip any stray `</tool_call>` literal that landed in plain
    // content. The cursor-advance in `processRegion` already skips
    // past close tags that legitimately follow open tags; this
    // handles bare close tags emitted by the model on their own.
    if chunk.contains(QwenParser.toolCallEnd) {
      chunk = chunk.replacingOccurrences(of: QwenParser.toolCallEnd, with: "")
    }
    // Strip stray `<think>` / `</think>` literals. These appear in
    // Trinity-style output where tool calls live inside `<think>...
    // </think>`: the `<tool_call>` triggers an implicit reasoning
    // end, and the trailing `</think>` after the tool call would
    // otherwise leak as message content. Mirrors SGLang's
    // `TrinityDetector._strip_think_tags`.
    if chunk.contains(QwenParser.thinkEnd) {
      chunk = chunk.replacingOccurrences(of: QwenParser.thinkEnd, with: "")
    }
    if chunk.contains(QwenParser.thinkStart) {
      chunk = chunk.replacingOccurrences(of: QwenParser.thinkStart, with: "")
    }
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

  private struct ToolCallRegion {
    var jsonText: String
    var isComplete: Bool
    var endIdxAfterClose: String.Index?
  }

  private func extractToolCallRegions() -> [ToolCallRegion] {
    var results: [ToolCallRegion] = []
    // Tool-call scanning starts at the beginning of the normal-phase
    // slice so any `<tool_call>` literal that happened to appear inside
    // reasoning text isn't mistaken for a real tool call, while already
    // seen normal-phase regions keep stable indexes across chunks.
    guard let scanStart = buffer.index(buffer.startIndex, offsetBy: toolRegionScanIdx, limitedBy: buffer.endIndex) else {
      return results
    }
    var pos = scanStart
    while let startRange = buffer.range(of: QwenParser.toolCallStart, range: pos ..< buffer.endIndex) {
      let jsonStart = startRange.upperBound
      if let endRange = buffer.range(of: QwenParser.toolCallEnd, range: jsonStart ..< buffer.endIndex) {
        let inner = buffer[jsonStart ..< endRange.lowerBound]
        results.append(ToolCallRegion(
          jsonText: inner.trimmingCharacters(in: .whitespacesAndNewlines),
          isComplete: true,
          endIdxAfterClose: endRange.upperBound,
        ))
        pos = endRange.upperBound
      } else {
        var raw = String(buffer[jsonStart ..< buffer.endIndex])
        let endChars = Array(QwenParser.toolCallEnd)
        let overlap = partialOverlap(suffixOf: Array(raw), with: endChars)
        if overlap > 0 {
          raw = String(raw.dropLast(overlap))
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let complete = !trimmed.isEmpty && isValidJSON(trimmed)
        results.append(ToolCallRegion(
          jsonText: trimmed,
          isComplete: complete,
          endIdxAfterClose: nil,
        ))
        break
      }
    }
    return results
  }

  private mutating func processRegion(
    at index: Int,
    region: ToolCallRegion,
    isEnd: Bool,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var call = toolCalls[index]

    if openMessage != nil, call.name == nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }

    if call.name == nil {
      if let name = extractToolName(from: region.jsonText) {
        call.name = name
        let outputIndex = takeOutputIndex()
        call.outputIndex = outputIndex
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
      }
    }

    if let outputIndex = call.outputIndex, call.name != nil {
      if let argsSoFar = extractArgumentsText(from: region.jsonText, isComplete: region.isComplete) {
        if argsSoFar.count > call.argsEmitted.count {
          let diffStart = argsSoFar.index(argsSoFar.startIndex, offsetBy: call.argsEmitted.count)
          let diff = String(argsSoFar[diffStart...])
          call.argsEmitted = argsSoFar
          events.append(.functionCallArgumentsDelta(.init(
            itemId: call.id,
            outputIndex: outputIndex,
            delta: diff,
            sequenceNumber: takeSequence(),
          )))
        }
      }
    }

    toolCalls[index] = call

    let regionClosed = region.endIdxAfterClose != nil
    if !call.closed, regionClosed || (isEnd && region.isComplete) {
      events.append(contentsOf: closeToolCall(at: index, status: .completed))
      // Advance past `</tool_call>` so trailing content emits as a
      // fresh message on the next scan.
      if let endIdx = region.endIdxAfterClose {
        let endOffset = buffer.distance(from: buffer.startIndex, to: endIdx)
        if endOffset > sentContentIdx {
          sentContentIdx = endOffset
        }
      }
    }

    return events
  }

  /// Drop any buffer prefix that the scan has already emitted as reasoning,
  /// emitted as normal content, or consumed as closed tool-call structure.
  /// Active tool-call regions and marker holdback bytes stay in the buffer
  /// so later chunks can be diffed against existing parser state.
  private mutating func pruneConsumedPrefix() {
    let dropCount: Int = switch phase {
      case .reasoning:
        sentReasoningIdx
      case .normal:
        sentContentIdx
    }
    guard dropCount > 0 else { return }

    let regions = extractToolCallRegions()
    var completedRegionsToDrop = 0
    for (index, region) in regions.enumerated() {
      guard index < toolCalls.count,
            let endIdx = region.endIdxAfterClose
      else {
        break
      }
      let endOffset = buffer.distance(from: buffer.startIndex, to: endIdx)
      guard endOffset <= dropCount else { break }
      completedRegionsToDrop += 1
    }

    if completedRegionsToDrop > 0 {
      toolCalls.removeFirst(completedRegionsToDrop)
    }

    buffer.removeFirst(dropCount)
    rebase(&sentReasoningIdx, dropping: dropCount)
    rebase(&sentContentIdx, dropping: dropCount)
    rebase(&toolRegionScanIdx, dropping: dropCount)
  }

  private func rebase(_ cursor: inout Int, dropping dropCount: Int) {
    cursor = Swift.max(0, cursor - dropCount)
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

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
