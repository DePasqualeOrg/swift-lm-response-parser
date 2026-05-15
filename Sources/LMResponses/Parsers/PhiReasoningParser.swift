// Copyright © Anthony DePasquale

import Foundation

/// Parser for Microsoft's Phi-4 reasoning variants
/// (`Phi-4-reasoning`, `Phi-4-reasoning-plus`, `Phi-4-mini-reasoning`).
///
/// **Wire shape.** The model emits chain-of-thought reasoning between
/// literal `<think>` and `</think>` markers, then a final answer. The
/// model card explicitly documents the format as:
/// `<think> {Thought section} </think> {Solution section}`.
///
/// **No tool calls.** None of the reasoning variants have a tool-call
/// channel in their chat templates. Any `<tool_call>` literal that
/// happens to appear is content, not a marker.
///
/// **Continuation.** For a continuation request whose `priorOutput` ended
/// inside an unclosed `<think>` block, construct with
/// ``InitialState/reasoning`` so the parser starts already in reasoning
/// phase and emits the next chunk as reasoning text rather than as a
/// fresh message.
struct PhiReasoningParser: ResponseFormatParser {
  /// Initial reasoning phase. Set to ``reasoning`` when the parser
  /// should start already inside a `<think>` block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let thinkStart = "<think>"
  private static let thinkEnd = "</think>"

  /// Active parsing window. Completed prefixes are removed after each scan so
  /// long reasoning blocks do not keep being rescanned.
  private var buffer: String = ""

  private enum Phase { case reasoning, normal }
  private var phase: Phase

  /// In reasoning phase: index in `buffer` of the next character to
  /// classify as reasoning text. Advances past a leading `<think>`.
  private var sentReasoningIdx: Int = 0

  /// Tracks whether the optional leading `<think>` in reasoning phase has
  /// already been handled. This remains true across buffer pruning so a
  /// literal `<think>` later in reasoning is emitted as text.
  private var reasoningStartResolved = false

  /// In normal phase: index in `buffer` of the next character to emit
  /// as message content. Set when reasoning ends to the position right
  /// after `</think>`.
  private var sentContentIdx: Int = 0

  private var openReasoning: OpenReasoning?
  private var openMessage: OpenMessage?

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
    if openReasoning != nil {
      // Reasoning ran to EOS without seeing `</think>`. Surface as
      // truncated content rather than dropping the partial block.
      events.append(contentsOf: closeReasoning(status: .incomplete))
    }
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    return events
  }

  // MARK: Scan loop

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    defer { pruneConsumedPrefix() }

    var events: [ResponseStreamingEvent] = []
    // A single chunk may carry both a reasoning open and close, so
    // loop while the phase keeps changing.
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

  private mutating func pruneConsumedPrefix() {
    let consumedIdx = phase == .reasoning ? sentReasoningIdx : sentContentIdx
    let dropCount = Swift.min(consumedIdx, buffer.count)
    guard dropCount > 0 else { return }

    buffer.removeFirst(dropCount)
    sentReasoningIdx = Swift.max(0, sentReasoningIdx - dropCount)
    sentContentIdx = Swift.max(0, sentContentIdx - dropCount)
  }

  // MARK: Reasoning phase

  private mutating func scanReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)
    let thinkStartChars = Array(Self.thinkStart)
    let thinkEndChars = Array(Self.thinkEnd)

    if !reasoningStartResolved {
      // Strip a leading `<think>` if the model emitted one.
      if sentReasoningIdx == 0,
         bufChars.count >= thinkStartChars.count,
         Array(bufChars[0 ..< thinkStartChars.count]) == thinkStartChars
      {
        sentReasoningIdx = thinkStartChars.count
        reasoningStartResolved = true
      } else if sentReasoningIdx == 0, !isEnd {
        // Could still grow into `<think>` – hold back any partial
        // prefix so we don't accidentally emit `<thi` as reasoning
        // text and then realize it was a marker.
        let leadingOverlap = leadingPartialOverlap(of: bufChars, with: thinkStartChars)
        if leadingOverlap > 0, leadingOverlap == bufChars.count {
          return events
        }
        reasoningStartResolved = true
      } else {
        reasoningStartResolved = true
      }
    }

    // Find `</think>` to end reasoning.
    let endIdx = bufChars.firstIndexOf(substring: Self.thinkEnd, after: sentReasoningIdx)

    let safeEnd: Int
    if let endIdx {
      safeEnd = endIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let endOverlap = partialOverlap(suffixOf: bufChars, with: thinkEndChars)
      safeEnd = bufChars.count - endOverlap
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

    guard let endIdx, endIdx == safeEnd else {
      return events
    }

    // `</think>` found – close reasoning, advance content cursor past
    // the marker, transition to normal phase.
    sentContentIdx = endIdx + thinkEndChars.count
    events.append(contentsOf: closeReasoning(status: .completed))
    phase = .normal
    return events
  }

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

  // MARK: Normal phase

  private mutating func scanNormal(isEnd: Bool) -> [ResponseStreamingEvent] {
    // If the buffer at `sentContentIdx` starts with (or could complete
    // into) `<think>`, transition back to reasoning. Only honored when
    // no message content has been emitted yet – once content has been
    // sent, a stray `<think>` is content, not a marker.
    if openMessage == nil, openReasoning == nil {
      if let transitioned = transitionToReasoningIfMarkerPresent(isEnd: isEnd) {
        return transitioned
      }
    }

    // Past the marker check, the buffer at `sentContentIdx` is
    // guaranteed not to be a `<think>` prefix (the upstream gate
    // returned nil) or has already been confirmed as plain content
    // (`<thinkerrr…`). Everything in the buffer is therefore message
    // content, including any partial-marker bytes – emitting them as
    // content matches the Phi spec, which says reasoning is at the
    // start of the response and any later `<think>` literal is text.
    let bufChars = Array(buffer)
    let sendableEnd = bufChars.count
    guard sendableEnd > sentContentIdx else { return [] }

    let chunk = String(bufChars[sentContentIdx ..< sendableEnd])
    sentContentIdx = sendableEnd
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
    let thinkStartChars = Array(Self.thinkStart)
    let cursor = sentContentIdx
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
      if Array(thinkStartChars[0 ..< slice.count]) == slice {
        if !isEnd {
          return [] // hold and wait for more
        }
      }
    }
    return nil
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

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
