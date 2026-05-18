// Copyright © Anthony DePasquale

import Foundation
import LMResponses

/// Owns one Responses envelope across multiple generation passes within a
/// single turn, plus the synthesized `function_call_output` items that
/// surface tool-dispatch results back into the same envelope.
///
/// Mirrors vLLM's `responses_stream_generator` (`serving.py:1959-1985`):
/// the generator owns the `sequence_number` counter via a closure while
/// transient `StreamingState` instances handle per-pass content/output
/// indices. ``ResponseTurnEnvelope`` is the Swift-class equivalent, plus
/// output-index rebasing across passes.
///
/// Internal to the bridge – session machinery, not parser API.
final class ResponseTurnEnvelope {
  /// Stable response ID minted at `init` time and reused on every
  /// envelope event and as `Response.id` on every snapshot.
  let responseId: String

  private let config: ResponseStreamConfig
  private let createdAt: Int

  /// Turn-scoped sequence-number counter. Rewrites every event's
  /// `sequence_number` so the consumer sees a strictly monotonic stream
  /// across the whole turn including pass boundaries.
  private var nextSequence: Int = 0

  /// Turn-scoped output-index counter. Per-pass-local indices are
  /// rebased to turn-scoped indices using ``PassHandle``'s captured
  /// offset.
  private var nextOutputIndex: Int = 0

  /// Live snapshot of items emitted so far in this turn. Populated on
  /// `output_item.added` / `output_item.done` (after rebasing) so the
  /// terminal response event carries the canonical `Response.output[]`.
  private var accumulatedItems: [ResponseOutputItem] = []

  /// Phase tracking for assertions.
  private enum Phase {
    case beforeStart
    case streaming
    case finalized
  }

  private var phase: Phase = .beforeStart

  init(config: ResponseStreamConfig) {
    responseId = IDFactory.make(.response)
    self.config = config
    createdAt = config.createdAt ?? Int(Date().timeIntervalSince1970)
  }

  /// Mint `resp_…`, yield `response.created` and `response.in_progress`.
  /// Must be called exactly once before any pass.
  func start() -> [ResponseStreamingEvent] {
    precondition(phase == .beforeStart, "ResponseTurnEnvelope.start() called more than once")
    phase = .streaming
    let snapshot = makeResponseSnapshot(status: .inProgress)
    return [
      .responseCreated(.init(response: snapshot, sequenceNumber: takeSequence())),
      .responseInProgress(.init(response: snapshot, sequenceNumber: takeSequence())),
    ]
  }

  /// Open a generation pass. The returned handle captures the current
  /// `nextOutputIndex` as the pass's offset. The counter advances
  /// lazily as forwarded events are ingested via ``PassHandle/forward(_:)``
  /// – each rebased event whose `output_index >= nextOutputIndex`
  /// pushes `nextOutputIndex` to `index + 1`. So when the next pass
  /// calls `beginPass()`, it naturally lands past the previous pass's
  /// highest emitted index, and no two passes collide on the same slot.
  func beginPass() -> PassHandle {
    precondition(phase == .streaming, "ResponseTurnEnvelope.beginPass() called outside streaming phase")
    let offset = nextOutputIndex
    return PassHandle(envelope: self, baseOutputIndex: offset)
  }

  /// Synthesize a `function_call_output` item paired by `call_id` to
  /// the originating function call. Allocates the next turn-scoped
  /// output index. Must be called between passes.
  func emitToolResult(_ output: ResponseFunctionCallOutput) -> [ResponseStreamingEvent] {
    precondition(phase == .streaming, "ResponseTurnEnvelope.emitToolResult() called outside streaming phase")
    let index = nextOutputIndex
    nextOutputIndex += 1

    let item = ResponseOutputItem.functionCallOutput(output)
    accumulate(item: item, at: index)
    return [
      .outputItemAdded(.init(item: item, outputIndex: index, sequenceNumber: takeSequence())),
      .outputItemDone(.init(item: item, outputIndex: index, sequenceNumber: takeSequence())),
    ]
  }

  /// Yield the terminal response event. Idempotent in the sense that
  /// calling it advances `phase` to `.finalized`; calling again traps.
  func finalize(info: FinishInfo) -> [ResponseStreamingEvent] {
    precondition(phase == .streaming, "ResponseTurnEnvelope.finalize() called outside streaming phase")

    let status: ResponseStatus = switch info.finishReason {
      case .stop: .completed
      case .length: .incomplete
      case .cancelled: .cancelled
    }
    let incompleteDetails: IncompleteDetails? = switch info.finishReason {
      case .length: IncompleteDetails(reason: .maxOutputTokens)
      case .stop, .cancelled: nil
    }
    let usage = ResponseUsage(
      inputTokens: info.inputTokens,
      outputTokens: info.outputTokens,
      totalTokens: info.inputTokens + info.outputTokens,
      inputTokensDetails: .init(cachedTokens: info.cachedInputTokens),
      outputTokensDetails: .init(reasoningTokens: info.reasoningOutputTokens),
    )

    var snapshot = makeResponseSnapshot(status: status)
    snapshot.incompleteDetails = incompleteDetails
    snapshot.usage = usage

    phase = .finalized
    // Open Responses uses `response.incomplete` for max-output-token
    // exhaustion. vLLM and SGLang currently emit `response.completed`
    // with `status = incomplete`; the multi-pass MLX envelope follows
    // the Open Responses terminal discriminator.
    switch info.finishReason {
      case .length:
        return [
          .responseIncomplete(.init(response: snapshot, sequenceNumber: takeSequence())),
        ]
      case .stop, .cancelled:
        return [
          .responseCompleted(.init(response: snapshot, sequenceNumber: takeSequence())),
        ]
    }
  }

  // MARK: PassHandle callbacks

  /// Allocate the next turn-scoped sequence number.
  fileprivate func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }

  /// Apply this pass's local-index offset, rebase the event, advance
  /// `nextOutputIndex` past the highest rebased index seen, and ingest
  /// `output_item.added` / `output_item.done` items into the
  /// turn-scoped accumulator.
  fileprivate func ingest(rebased event: ResponseStreamingEvent) {
    switch event {
      case let .outputItemAdded(e):
        accumulate(item: e.item, at: e.outputIndex)
      case let .outputItemDone(e):
        accumulate(item: e.item, at: e.outputIndex)
      default:
        break
    }
    if let index = outputIndex(of: event) {
      if index >= nextOutputIndex {
        nextOutputIndex = index + 1
      }
    }
  }

  // MARK: Internals

  private func accumulate(item: ResponseOutputItem, at outputIndex: Int) {
    // Output indices are allocated densely: every parser's
    // pass-local index counter increments by 1 per item, and
    // `emitToolResult` and `beginPass`'s rebasing both advance the
    // turn-scoped counter monotonically. A skipped slot would mean
    // a contract violation upstream – hold the line here so the
    // failure surfaces at its source rather than silently producing
    // duplicate items in the snapshot.
    precondition(
      outputIndex <= accumulatedItems.count,
      "Output indices must be allocated densely; got \(outputIndex) with array length \(accumulatedItems.count)",
    )
    if outputIndex == accumulatedItems.count {
      accumulatedItems.append(item)
    } else {
      accumulatedItems[outputIndex] = item
    }
  }

  private func makeResponseSnapshot(status: ResponseStatus?) -> Response {
    // `output: accumulatedItems` shares the array's COW storage at
    // assignment time; the buffer is split on first mutation by
    // either side. Every item type in `ResponseOutputItem` is a
    // value type all the way down, so a snapshot retained by a
    // consumer (notably ``ResponseChatSession/lastResponse``)
    // remains stable even if the envelope kept mutating
    // `accumulatedItems` afterward. Today the
    // `phase == .finalized` precondition forbids further
    // mutation anyway, but the COW + value-semantics layer is
    // what makes the retained snapshot durable.
    Response(
      id: responseId,
      createdAt: createdAt,
      model: config.model,
      output: accumulatedItems,
      status: status,
      instructions: config.instructions,
      temperature: config.temperature,
      topP: config.topP,
      maxOutputTokens: config.maxOutputTokens,
    )
  }
}

/// Scope handle for one generation pass. Captures the per-pass
/// output-index offset at ``ResponseTurnEnvelope/beginPass()`` time and
/// applies it consistently to every batch of events forwarded through
/// ``forward(_:)``. Multiple `forward` calls per pass are normal – once
/// per chunk plus once for `parser.finalize()`'s events.
final class PassHandle {
  // Strong reference: `ResponseTurnEnvelope` holds no reference to
  // `PassHandle`, so there is no cycle to break, and the envelope's
  // lifetime is owned by the surrounding session/turn machinery for
  // longer than any pass needs it. Holding strong here turns a future
  // "envelope released early" refactor into a clear failure rather
  // than `forward()` silently dropping events.
  private let envelope: ResponseTurnEnvelope
  private let baseOutputIndex: Int
  private var ended: Bool = false

  fileprivate init(envelope: ResponseTurnEnvelope, baseOutputIndex: Int) {
    self.envelope = envelope
    self.baseOutputIndex = baseOutputIndex
  }

  /// Rebase per-pass-local `output_index` to turn-scoped, replace
  /// `sequence_number` with the envelope's next turn-scoped value, and
  /// ingest item add/done events into the envelope's turn-scoped
  /// accumulator.
  func forward(_ events: [ResponseStreamingEvent]) -> [ResponseStreamingEvent] {
    guard !ended else { return [] }
    var out: [ResponseStreamingEvent] = []
    out.reserveCapacity(events.count)
    for event in events {
      // Suppress envelope-level lifecycle events from per-pass
      // emitters: the turn envelope owns those.
      switch event {
        case .responseCreated, .responseInProgress, .responseCompleted, .responseIncomplete:
          continue
        default:
          break
      }
      var rebased = rebaseOutputIndex(event, by: baseOutputIndex)
      rebased.sequenceNumber = envelope.takeSequence()
      envelope.ingest(rebased: rebased)
      out.append(rebased)
    }
    return out
  }

  /// Mark the pass complete. Idempotent. Belt-and-suspenders against a
  /// future refactor that might invoke `forward` on this handle after
  /// the surrounding pass scope has ended – today every `forward` call
  /// is on the same task that calls `end()`, so the guard never fires
  /// in practice.
  func end() {
    ended = true
  }
}

// MARK: Output-index rebasing

/// Returns a copy of `event` with its `output_index` shifted by
/// `offset`. Lifecycle envelope events (which lack an `output_index`)
/// pass through unchanged.
private func rebaseOutputIndex(_ event: ResponseStreamingEvent, by offset: Int) -> ResponseStreamingEvent {
  switch event {
    case .responseCreated, .responseInProgress, .responseCompleted, .responseIncomplete:
      return event
    case var .outputItemAdded(e):
      e.outputIndex += offset
      return .outputItemAdded(e)
    case var .outputItemDone(e):
      e.outputIndex += offset
      return .outputItemDone(e)
    case var .contentPartAdded(e):
      e.outputIndex += offset
      return .contentPartAdded(e)
    case var .contentPartDone(e):
      e.outputIndex += offset
      return .contentPartDone(e)
    case var .outputTextDelta(e):
      e.outputIndex += offset
      return .outputTextDelta(e)
    case var .outputTextDone(e):
      e.outputIndex += offset
      return .outputTextDone(e)
    case var .functionCallArgumentsDelta(e):
      e.outputIndex += offset
      return .functionCallArgumentsDelta(e)
    case var .functionCallArgumentsDone(e):
      e.outputIndex += offset
      return .functionCallArgumentsDone(e)
    case var .reasoningDelta(e):
      e.outputIndex += offset
      return .reasoningDelta(e)
    case var .reasoningDone(e):
      e.outputIndex += offset
      return .reasoningDone(e)
    case var .outputTextAnnotationAdded(e):
      e.outputIndex += offset
      return .outputTextAnnotationAdded(e)
  }
}

private func outputIndex(of event: ResponseStreamingEvent) -> Int? {
  switch event {
    case .responseCreated, .responseInProgress, .responseCompleted, .responseIncomplete:
      nil
    case let .outputItemAdded(e): e.outputIndex
    case let .outputItemDone(e): e.outputIndex
    case let .contentPartAdded(e): e.outputIndex
    case let .contentPartDone(e): e.outputIndex
    case let .outputTextDelta(e): e.outputIndex
    case let .outputTextDone(e): e.outputIndex
    case let .functionCallArgumentsDelta(e): e.outputIndex
    case let .functionCallArgumentsDone(e): e.outputIndex
    case let .reasoningDelta(e): e.outputIndex
    case let .reasoningDone(e): e.outputIndex
    case let .outputTextAnnotationAdded(e): e.outputIndex
  }
}
