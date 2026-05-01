// Copyright © Anthony DePasquale

import Foundation

/// Extractor for a `<think>...</think>` reasoning preamble that may
/// optionally precede a tool-call body. Used by V3-family parsers
/// (DeepSeek V3 / V3.1 / V3.2) and GLM 4 when a chat template has
/// `chat_template_kwargs.thinking=True`.
///
/// **Wire shape.** The opener is optional – chat templates often inject
/// `<think>` into the prompt, in which case the model emits only
/// `</think>` to close. Mirrors vLLM's `DeepSeekR1ReasoningParser`,
/// which `DeepSeekV3ReasoningParser` delegates to when thinking is
/// enabled.
///
/// **Usage.** The host parser holds an instance, calls ``drain`` from
/// its own `process()` and `finalize()` before the regular tool-call
/// scan, and only proceeds with the scan when ``phase`` reaches
/// ``Phase/done``. The host's `nextSequence` and `nextOutputIndex`
/// counters are passed inout so the helper's emitted events are
/// consistent with the host's own emissions.
struct ThinkPreambleExtractor {
  /// Whether this extractor is active. When false, ``drain`` is a
  /// no-op pass-through and ``phase`` is permanently ``Phase/done``.
  enum InitialState: Equatable {
    /// Not currently in a reasoning block. The host parser proceeds
    /// directly to tool-call scanning. Used for continuation requests
    /// whose `priorOutput` already contained `</think>`.
    case normal
    /// In an implicit reasoning block (the chat template injected the
    /// `<think>` opener into the prompt; the model output starts
    /// mid-reasoning). The extractor still tolerates a leading
    /// `<think>` opener if the model re-emits it.
    case reasoning
  }

  enum Phase: Equatable {
    /// Either the extractor is disabled, or reasoning has finished
    /// (`</think>` consumed). Host parser proceeds with the regular
    /// scan.
    case done
    /// Currently inside a reasoning block. Bytes drain into the
    /// reasoning item until `</think>` arrives.
    case inThink
  }

  private static let thinkStart = "<think>"
  private static let thinkEnd = "</think>"
  private let implicitEndTokens: [String]

  private(set) var phase: Phase
  private var openReasoning: OpenReasoning?
  /// True until the first `drain` call has had a chance to skip a
  /// leading `<think>` opener. After that, leading `<think>` is just
  /// reasoning content (the chat template-injected opener case).
  private var checkedOpener: Bool = false

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  init(initialState: InitialState, implicitEndTokens: [String] = []) {
    self.implicitEndTokens = implicitEndTokens
    switch initialState {
      case .normal: phase = .done
      case .reasoning: phase = .inThink
    }
  }

  /// Drain the reasoning preamble from the head of `buffer`. Removes
  /// consumed bytes from the front of `buffer` and returns the
  /// emitted events. After the call, ``phase`` is ``Phase/done``
  /// when reasoning has fully closed (or was never present).
  ///
  /// Bytes are removed from the front (rather than tracked via a
  /// cursor) so the host parser sees the post-reasoning buffer
  /// cleanly and doesn't need to know about reasoning offsets.
  mutating func drain(
    buffer: inout String,
    isEnd: Bool,
    nextSequence: inout Int,
    nextOutputIndex: inout Int,
  ) -> [ResponseStreamingEvent] {
    if phase == .done { return [] }
    var events: [ResponseStreamingEvent] = []

    // First call: optionally skip a leading `<think>` opener (with
    // tolerance for leading whitespace). After this, any further
    // `<think>` text is just reasoning content.
    if !checkedOpener {
      let bufChars = Array(buffer)
      var cursor = 0
      while cursor < bufChars.count, bufChars[cursor].isWhitespace {
        cursor += 1
      }
      let thinkStartChars = Array(Self.thinkStart)
      let available = bufChars.count - cursor

      if available >= thinkStartChars.count {
        if Array(bufChars[cursor ..< cursor + thinkStartChars.count]) == thinkStartChars {
          // Skip the opener (and any leading whitespace before it).
          buffer.removeFirst(cursor + thinkStartChars.count)
        }
        checkedOpener = true
      } else if available > 0 {
        // Buffer ends mid-`<think>` after whitespace.
        let slice = Array(bufChars[cursor ..< bufChars.count])
        if Array(thinkStartChars[0 ..< slice.count]) == slice, !isEnd {
          // Hold for more bytes – don't emit anything yet.
          return events
        }
        // Not the opener; commit and proceed to inThink streaming.
        checkedOpener = true
      } else if available == 0, cursor > 0 {
        // Whitespace-only buffer.
        if !isEnd { return events }
        // At EOS with only whitespace – nothing to emit.
        buffer.removeAll()
        checkedOpener = true
      } else {
        // Empty buffer.
        if !isEnd { return events }
        checkedOpener = true
      }
    }

    // Now stream reasoning until `</think>` arrives, or until an
    // optional tool-call marker implicitly ends reasoning while leaving
    // the marker itself in the host parser buffer.
    if let boundary = firstReasoningBoundary(in: buffer) {
      let text = String(buffer[buffer.startIndex ..< boundary.range.lowerBound])
      if !text.isEmpty {
        events.append(contentsOf: emitReasoningDelta(
          text: text,
          nextSequence: &nextSequence,
          nextOutputIndex: &nextOutputIndex,
        ))
      }
      let consumedEnd = boundary.consumesMarker ? boundary.range.upperBound : boundary.range.lowerBound
      let consumed = buffer.distance(from: buffer.startIndex, to: consumedEnd)
      buffer.removeFirst(consumed)
      events.append(contentsOf: closeReasoning(
        status: .completed,
        nextSequence: &nextSequence,
      ))
      phase = .done
      return events
    }

    let bufChars = Array(buffer)
    let markers = [Self.thinkEnd] + implicitEndTokens
    let overlap = isEnd ? 0 : markers.reduce(0) { current, marker in
      max(current, partialOverlap(suffixOf: bufChars, with: Array(marker)))
    }
    let safeEnd = bufChars.count - overlap
    if safeEnd > 0 {
      let chunk = String(bufChars[0 ..< safeEnd])
      events.append(contentsOf: emitReasoningDelta(
        text: chunk,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
      buffer.removeFirst(safeEnd)
    }
    return events
  }

  private func firstReasoningBoundary(in buffer: String) -> (range: Range<String.Index>, consumesMarker: Bool)? {
    var best: (range: Range<String.Index>, consumesMarker: Bool)?
    if let endRange = buffer.range(of: Self.thinkEnd) {
      best = (endRange, true)
    }
    for token in implicitEndTokens {
      guard let range = buffer.range(of: token) else { continue }
      if best == nil || range.lowerBound < best!.range.lowerBound {
        best = (range, false)
      }
    }
    return best
  }

  /// Close any open reasoning item at finalize time. Called after the
  /// host parser's regular finalize logic when the stream ended
  /// mid-reasoning (no `</think>` ever arrived).
  mutating func finalizeIfOpen(
    nextSequence: inout Int,
  ) -> [ResponseStreamingEvent] {
    guard openReasoning != nil else { return [] }
    return closeReasoning(status: .incomplete, nextSequence: &nextSequence)
  }

  private mutating func emitReasoningDelta(
    text: String,
    nextSequence: inout Int,
    nextOutputIndex: inout Int,
  ) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var events: [ResponseStreamingEvent] = []
    if openReasoning == nil {
      events.append(contentsOf: openReasoningItem(
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
    }
    if var r = openReasoning {
      r.emittedText += text
      openReasoning = r
      let seq = nextSequence
      nextSequence += 1
      events.append(.reasoningTextDelta(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        delta: text,
        sequenceNumber: seq,
      )))
    }
    return events
  }

  private mutating func openReasoningItem(
    nextSequence: inout Int,
    nextOutputIndex: inout Int,
  ) -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.reasoning)
    let outputIndex = nextOutputIndex
    nextOutputIndex += 1
    openReasoning = OpenReasoning(id: id, outputIndex: outputIndex)
    let seq1 = nextSequence
    nextSequence += 1
    let seq2 = nextSequence
    nextSequence += 1
    return [
      .outputItemAdded(.init(
        item: .reasoning(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex,
        sequenceNumber: seq1,
      )),
      .contentPartAdded(.init(
        itemId: id,
        outputIndex: outputIndex,
        contentIndex: 0,
        part: .reasoningText(.init(text: "")),
        sequenceNumber: seq2,
      )),
    ]
  }

  private mutating func closeReasoning(
    status: ItemStatus,
    nextSequence: inout Int,
  ) -> [ResponseStreamingEvent] {
    guard let r = openReasoning else { return [] }
    openReasoning = nil
    let part = ReasoningTextContent(text: r.emittedText)
    let seq1 = nextSequence
    nextSequence += 1
    let seq2 = nextSequence
    nextSequence += 1
    let seq3 = nextSequence
    nextSequence += 1
    return [
      .reasoningTextDone(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        text: r.emittedText,
        sequenceNumber: seq1,
      )),
      .contentPartDone(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        part: .reasoningText(part),
        sequenceNumber: seq2,
      )),
      .outputItemDone(.init(
        item: .reasoning(.init(
          id: r.id,
          content: [.reasoningText(part)],
          status: status,
        )),
        outputIndex: r.outputIndex,
        sequenceNumber: seq3,
      )),
    ]
  }
}
