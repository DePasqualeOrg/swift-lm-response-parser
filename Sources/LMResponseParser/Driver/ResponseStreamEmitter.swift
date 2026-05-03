// Copyright © Anthony DePasquale

import Foundation

/// Wraps a per-format parser with the response-scoped envelope events,
/// response-scoped sequence numbering, and the `resp_…` ID so the emitted
/// stream is directly consumable by Responses-API SDKs.
///
/// The consumer owns its own model loop and streaming detokenizer; the
/// emitter only owns the response-scoped envelope concerns.
///
/// **Consumer pattern.**
///
/// ```swift
/// let parser = ResponseFormat
///     .infer(modelName: …, modelType: …, modelConfig: …)
///     .makeParser(tokenizer: tokenizer, tools: tools)
/// let emitter = ResponseStreamEmitter(parser: parser, config: ResponseStreamConfig(model: modelName))
///
/// for event in emitter.start() { handle(event) }
///
/// var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
/// var pendingTokenIds: [Int] = []
/// for token in tokens {
///     detokenizer.append(token: token)
///     pendingTokenIds.append(token)
///     if let chunk = detokenizer.next() {
///         for event in emitter.process(text: chunk, tokenIds: pendingTokenIds) { handle(event) }
///         pendingTokenIds.removeAll()
///     }
/// }
/// for event in emitter.finalize(info: finishInfo) { handle(event) }
/// ```
///
/// The emitter is single-use: one instance per response. It is `~Copyable`-ish
/// in spirit (call `start()` exactly once, then any number of `process(...)`,
/// then `finalize(...)` exactly once); the state machine asserts on the
/// happy-path ordering when DEBUG is enabled.
package final class ResponseStreamEmitter {
  private var parser: any ResponseFormatParser
  private let config: ResponseStreamConfig

  /// Response-scoped ID minted once per emitter and reused on every
  /// envelope event and as ``Response/id`` on every snapshot.
  package let responseId: String

  private let createdAt: Int

  /// Response-scoped sequence number assigned to the next event the
  /// emitter yields. Zero-based, monotonically increasing across the
  /// entire stream including the lifecycle envelope events.
  private var nextSequence: Int = 0

  /// Items the emitter has seen flow through. Populated on
  /// `output_item.added` / `output_item.done` so the terminal
  /// response event can carry the canonical
  /// ``Response/output`` array.
  private var accumulatedItems: [ResponseOutputItem] = []

  /// State machine for assertion / sanity-check purposes.
  private enum Phase {
    case beforeStart
    case streaming
    case finalized
  }

  private var phase: Phase = .beforeStart

  package init(parser: any ResponseFormatParser, config: ResponseStreamConfig) {
    self.parser = parser
    self.config = config
    responseId = IDFactory.make(.response)
    createdAt = config.createdAt ?? Int(Date().timeIntervalSince1970)
  }

  /// Yield the lifecycle-envelope events that must appear at the head of
  /// the stream: `response.created` (sequence 0) and `response.in_progress`
  /// (sequence 1). These must be emitted before any token processing so
  /// the consumer's snapshot is initialized even when generation produces
  /// zero tokens or is cancelled immediately.
  package func start() -> [ResponseStreamingEvent] {
    precondition(phase == .beforeStart, "ResponseStreamEmitter.start() called more than once")
    phase = .streaming

    let snapshot = makeResponseSnapshot(status: .inProgress)
    let created = ResponseCreatedEvent(response: snapshot, sequenceNumber: takeSequence())
    let inProgress = ResponseInProgressEvent(response: snapshot, sequenceNumber: takeSequence())
    return [.responseCreated(created), .responseInProgress(inProgress)]
  }

  /// Feed a chunk of detokenized text through the parser. When supplied,
  /// `tokenIds` must be the IDs whose detokenized form is exactly `text`.
  /// Parsers may use that alignment for token-aware marker handling.
  package func process(text: String, tokenIds: [Int]? = nil) -> [ResponseStreamingEvent] {
    precondition(phase == .streaming, "ResponseStreamEmitter.process() called outside the streaming phase")

    let parserEvents = parser.process(ParserInput(text: text, tokenIds: tokenIds))
    return resequenceAndTrack(parserEvents)
  }

  /// Flush parser state and yield the terminal response event.
  /// The parser is responsible for closing any open items with the right
  /// status; the emitter only translates the consumer's ``FinishInfo``
  /// into the response-level ``ResponseStatus`` and ``IncompleteDetails``
  /// that ride on the terminal event.
  package func finalize(info: FinishInfo) -> [ResponseStreamingEvent] {
    precondition(phase == .streaming, "ResponseStreamEmitter.finalize() called outside the streaming phase")

    var output = resequenceAndTrack(parser.finalize())

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

    // Open Responses uses `response.incomplete` for max-output-token
    // exhaustion. vLLM and SGLang currently emit `response.completed`
    // with `status = incomplete`; this emitter stays spec-first.
    switch info.finishReason {
      case .length:
        let incomplete = ResponseIncompleteEvent(response: snapshot, sequenceNumber: takeSequence())
        output.append(.responseIncomplete(incomplete))
      case .stop, .cancelled:
        let completed = ResponseCompletedEvent(response: snapshot, sequenceNumber: takeSequence())
        output.append(.responseCompleted(completed))
    }

    phase = .finalized
    return output
  }

  // MARK: Internals

  private func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }

  private func makeResponseSnapshot(status: ResponseStatus?) -> Response {
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

  /// Substitute response-scoped sequence numbers in place of the parser's
  /// parser-local ones, and track item add/done events so the terminal
  /// response snapshot has the right `output[]` content.
  private func resequenceAndTrack(_ events: [ResponseStreamingEvent]) -> [ResponseStreamingEvent] {
    var out: [ResponseStreamingEvent] = []
    out.reserveCapacity(events.count)
    for var event in events {
      event.sequenceNumber = takeSequence()
      switch event {
        case let .outputItemAdded(e):
          accumulate(item: e.item, at: e.outputIndex)
        case let .outputItemDone(e):
          accumulate(item: e.item, at: e.outputIndex)
        default:
          break
      }
      out.append(event)
    }
    return out
  }

  private func accumulate(item: ResponseOutputItem, at outputIndex: Int) {
    guard outputIndex <= accumulatedItems.count else {
      return
    }
    if outputIndex == accumulatedItems.count {
      accumulatedItems.append(item)
    } else {
      accumulatedItems[outputIndex] = item
    }
  }
}
