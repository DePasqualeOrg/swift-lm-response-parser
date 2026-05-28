// Copyright © Anthony DePasquale

import Foundation
import LMResponses

/// Owns one Responses envelope across multiple generation passes within a
/// single turn, plus the synthesized `function_call_output` items that
/// surface tool-dispatch results back into the same envelope.
///
/// Internal to the bridge — session machinery, not parser API.
///
/// **Concurrency.** `@unchecked Sendable` with an internal `NSLock`
/// taken on every public method and on the `fileprivate` callbacks
/// `PassHandle/forward(_:)` uses. The lock lives here (not in a
/// wrapper) so it covers the `PassHandle` callback path too —
/// otherwise a wrapper's lock would be bypassed by the callbacks.
final class ResponseTurnEnvelope: @unchecked Sendable {
  let responseId: String

  private let lock = NSLock()
  private let config: ResponseStreamConfig
  private let createdAt: Int

  /// Turn-scoped sequence-number counter.
  private var nextSequence: Int = 0

  /// Turn-scoped output-index counter.
  private var nextOutputIndex: Int = 0

  /// Sparse map of `outputIndex → item`. Stored sparsely so a
  /// parser anomaly (non-dense indices) doesn't force fabricating
  /// placeholder items into the terminal `Response.output` — those
  /// would leak into downstream consumers keying off message IDs.
  /// Densified on snapshot.
  private var accumulatedItems: [Int: ResponseOutputItem] = [:]

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

  func start() -> [ResponseStreamingEvent] {
    lock.lock(); defer { lock.unlock() }
    precondition(phase == .beforeStart, "ResponseTurnEnvelope.start() called more than once")
    phase = .streaming
    let snapshot = makeResponseSnapshotLocked(status: .inProgress)
    return [
      .responseCreated(.init(response: snapshot, sequenceNumber: takeSequenceLocked())),
      .responseInProgress(.init(response: snapshot, sequenceNumber: takeSequenceLocked())),
    ]
  }

  func beginPass() -> PassHandle {
    lock.lock(); defer { lock.unlock() }
    precondition(phase == .streaming, "ResponseTurnEnvelope.beginPass() called outside streaming phase")
    let offset = nextOutputIndex
    return PassHandle(envelope: self, baseOutputIndex: offset)
  }

  func emitToolResult(_ output: ResponseFunctionCallOutput) -> [ResponseStreamingEvent] {
    lock.lock(); defer { lock.unlock() }
    precondition(phase == .streaming, "ResponseTurnEnvelope.emitToolResult() called outside streaming phase")
    let index = nextOutputIndex
    nextOutputIndex += 1

    let item = ResponseOutputItem.functionCallOutput(output)
    accumulateLocked(item: item, at: index)
    return [
      .outputItemAdded(.init(item: item, outputIndex: index, sequenceNumber: takeSequenceLocked())),
      .outputItemDone(.init(item: item, outputIndex: index, sequenceNumber: takeSequenceLocked())),
    ]
  }

  func finalize(info: FinishInfo) -> [ResponseStreamingEvent] {
    lock.lock(); defer { lock.unlock() }
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

    var snapshot = makeResponseSnapshotLocked(status: status)
    snapshot.incompleteDetails = incompleteDetails
    snapshot.usage = usage

    phase = .finalized
    switch info.finishReason {
      case .length:
        return [
          .responseIncomplete(.init(response: snapshot, sequenceNumber: takeSequenceLocked())),
        ]
      case .stop, .cancelled:
        return [
          .responseCompleted(.init(response: snapshot, sequenceNumber: takeSequenceLocked())),
        ]
    }
  }

  // MARK: PassHandle callbacks (acquire lock per call)

  fileprivate func takeSequence() -> Int {
    lock.lock(); defer { lock.unlock() }
    return takeSequenceLocked()
  }

  fileprivate func ingest(rebased event: ResponseStreamingEvent) {
    lock.lock(); defer { lock.unlock() }
    switch event {
      case let .outputItemAdded(e):
        accumulateLocked(item: e.item, at: e.outputIndex)
      case let .outputItemDone(e):
        accumulateLocked(item: e.item, at: e.outputIndex)
      default:
        break
    }
    if let index = outputIndex(of: event) {
      if index >= nextOutputIndex {
        nextOutputIndex = index + 1
      }
    }
  }

  // MARK: Locked helpers (caller MUST hold `lock`)

  private func takeSequenceLocked() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }

  private func accumulateLocked(item: ResponseOutputItem, at outputIndex: Int) {
    accumulatedItems[outputIndex] = item
  }

  private func makeResponseSnapshotLocked(status: ResponseStatus?) -> Response {
    let densified = accumulatedItems.keys.sorted().compactMap { accumulatedItems[$0] }
    return Response(
      id: responseId,
      createdAt: createdAt,
      model: config.model,
      output: densified,
      status: status,
      instructions: config.instructions,
      temperature: config.temperature,
      topP: config.topP,
      maxOutputTokens: config.maxOutputTokens,
    )
  }
}

final class PassHandle {
  private let envelope: ResponseTurnEnvelope
  private let baseOutputIndex: Int
  private var ended: Bool = false

  fileprivate init(envelope: ResponseTurnEnvelope, baseOutputIndex: Int) {
    self.envelope = envelope
    self.baseOutputIndex = baseOutputIndex
  }

  func forward(_ events: [ResponseStreamingEvent]) -> [ResponseStreamingEvent] {
    guard !ended else { return [] }
    var out: [ResponseStreamingEvent] = []
    out.reserveCapacity(events.count)
    for event in events {
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

  func end() {
    ended = true
  }
}

// MARK: Output-index rebasing

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
