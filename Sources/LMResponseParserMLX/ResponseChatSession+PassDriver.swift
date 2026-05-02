// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import MLXLMCommon

/// Drive one generation pass through the parser and turn envelope. Each
/// chunk's events are forwarded through the pass scope (which rebases them
/// onto the turn-scoped envelope) and ingested into the turn-scoped
/// `itemsBox` so the caller can read pending tool calls from
/// `itemsBox.snapshot` after this call returns. Returns the pass-finish
/// info; the caller composes the final `FinishInfo` from the per-pass
/// counts accumulated in `usageBox`.
///
/// Mirrors `ChatSession.streamMap`'s pattern of running generation
/// alongside (not inside) `ModelContainer.perform`: the perform call
/// returns the `ModelContext` via `SendableBox`, generation runs against
/// that captured reference. The cache lock above is what guarantees
/// exclusive access; the model weights themselves are not being mutated
/// across the call.
func runOnePass<Element: Sendable>(
  on model: ModelContainer,
  input: LMInput,
  kvCache: [KVCache],
  generateParameters: GenerateParameters,
  modelType: String?,
  modelConfig: [String: any Sendable]?,
  tools: [ToolSpec],
  envelopeBox: EnvelopeBox,
  transform: @Sendable ([ResponseStreamingEvent]) -> [Element],
  continuation: AsyncThrowingStream<Element, Error>.Continuation,
  usageBox: UsageBox,
  itemsBox: TurnItemsBox,
  format: ResponseFormat?,
) async throws -> PassFinishInfo {
  let contextBox = await model.perform { context in
    SendableBox(context)
  }
  let context = contextBox.consume()

  let adapter = MLXTokenizerAdapter(context.tokenizer)
  let resolvedFormat = format
    ?? ResponseFormat.infer(
      modelName: context.configuration.name,
      modelType: modelType ?? "",
      modelConfig: modelConfig ?? [:],
    )
    ?? .json
  // The parser sees only generated suffix text. If the rendered prompt
  // leaves a parser marker open at that suffix boundary, pass the prompt
  // tail through the same `priorOutput` contract used by continuation
  // streams.
  let effectivePriorOutput = resolvedFormat.combinedPriorOutput(
    fromPreparedPrompt: input,
    tokenizer: adapter,
    generatedPriorOutput: nil,
  )
  var parser = resolvedFormat.makeParser(
    tokenizer: adapter,
    tools: tools,
    priorOutput: effectivePriorOutput,
  )

  let pass = try runPass(
    on: context,
    input: input,
    cache: kvCache,
    parameters: generateParameters,
    format: resolvedFormat,
    adapter: adapter,
  )

  let scope = envelopeBox.beginPass()
  var lastInfo: PassFinishInfo?

  do {
    for await event in pass {
      if Task.isCancelled { break }
      switch event {
        case let .chunk(text, tokenIds):
          let parserEvents = parser.process(.init(text: text, tokenIds: tokenIds))
          usageBox.observe(events: parserEvents, tokenCount: tokenIds.count)
          let forwarded = scope.forward(parserEvents)
          itemsBox.ingest(forwarded)
          for batch in transform(forwarded) {
            continuation.yield(batch)
          }

        case let .finished(info):
          lastInfo = info
          usageBox.addPassInput(info.inputTokens)
          usageBox.addPassOutput(info.outputTokens)
      }
    }

    // Skip `parser.finalize()` on cancellation. A force-finalized
    // parser would synthesize `output_item.done` events for items the
    // model never finished (e.g., a half-emitted function call
    // closed at its current state), which would then land in
    // `itemsBox` and surface as "completed" in any items snapshot
    // the consumer reads after the cancelled turn. Letting the
    // parser go out of scope unfinalized is safe – parsers that
    // hold non-trivial resources are expected to release them in
    // `deinit`, since the session does not call `finalize()` on the
    // throw path either.
    if Task.isCancelled {
      await pass.awaitCleanup()
      scope.end()
      // If MLX delivered a `.finished` record before the
      // cancellation was observed at the for-loop boundary, the
      // per-pass counts are already in `usageBox` (see the
      // `.finished` case above). Mirror them in the returned
      // `PassFinishInfo` so the local invariant "the returned
      // info's per-pass counts match what was accumulated into
      // usageBox" holds even on the cancel path. The session-
      // level cancel path silent-closes without reading this
      // value today, but a future logging or telemetry hook
      // would otherwise see contradictory numbers.
      return PassFinishInfo(
        inputTokens: lastInfo?.inputTokens ?? 0,
        outputTokens: lastInfo?.outputTokens ?? 0,
        finishReason: .cancelled,
      )
    }

    let finalEvents = parser.finalize()
    let forwardedFinal = scope.forward(finalEvents)
    itemsBox.ingest(forwardedFinal)
    // `usageBox.inReasoning` is shared across passes within a turn.
    // `parser.finalize()` can emit `output_item.done` for a
    // reasoning item that was still open at end-of-stream (e.g., a
    // half-open `<think>` block); without observing that close, a
    // following pass would inherit `inReasoning = true` and
    // mis-attribute its content tokens as reasoning. Pass
    // `tokenCount: 0` so the close lands as a state transition
    // without inflating reasoning counts.
    usageBox.observe(events: forwardedFinal, tokenCount: 0)
    for batch in transform(forwardedFinal) {
      continuation.yield(batch)
    }

    await pass.awaitCleanup()
    scope.end()

    if let info = lastInfo {
      return info
    }
    throw ResponseChatSessionError.passDidNotFinish
  } catch {
    await pass.awaitCleanup()
    scope.end()
    throw error
  }
}

/// Box for the turn envelope so it can cross task boundaries within the
/// session. The envelope is single-task-mutated; this just satisfies the
/// data-race checker.
final class EnvelopeBox: @unchecked Sendable {
  private let envelope: ResponseTurnEnvelope

  init(_ envelope: ResponseTurnEnvelope) {
    self.envelope = envelope
  }

  func start() -> [ResponseStreamingEvent] {
    envelope.start()
  }

  func beginPass() -> PassHandle {
    envelope.beginPass()
  }

  func emitToolResult(_ output: ResponseFunctionCallOutput) -> [ResponseStreamingEvent] {
    envelope.emitToolResult(output)
  }

  func finalize(info: FinishInfo) -> [ResponseStreamingEvent] {
    envelope.finalize(info: info)
  }
}

final class UsageBox: @unchecked Sendable {
  private var usage = UsageAccumulator()

  func observe(events: [ResponseStreamingEvent], tokenCount: Int) {
    usage.observe(events: events, tokenCount: tokenCount)
  }

  func addPassInput(_ tokens: Int) {
    usage.addPassInput(tokens)
  }

  func addPassOutput(_ tokens: Int) {
    usage.addPassOutput(tokens)
  }

  func finalInfo(finishReason: FinishReason) -> FinishInfo {
    usage.finalInfo(finishReason: finishReason)
  }
}

final class TurnItemsBox: @unchecked Sendable {
  private var inner = ResponseItemsAccumulator()
  var snapshot: [ResponseOutputItem] {
    inner.items
  }

  func ingest(_ events: [ResponseStreamingEvent]) {
    inner.ingest(events)
  }
}

/// Lock-protected holder for the most recently finalized turn's
/// terminal `Response` snapshot. Reads from outside the session task
/// race writes from inside the producer task; an `NSLock` keeps the
/// read/write atomic and `@unchecked Sendable` is acceptable because
/// the lock provides the synchronization the data-race checker can't
/// see.
final class LastResponseBox: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Response?
  func set(_ response: Response) {
    lock.lock(); defer { lock.unlock() }
    value = response
  }

  var current: Response? {
    lock.lock(); defer { lock.unlock() }
    return value
  }
}

public enum ResponseChatSessionError: LocalizedError {
  case noCacheAvailable
  case passDidNotFinish

  public var errorDescription: String? {
    switch self {
      case .noCacheAvailable:
        "No KV cache is available. Call streamResponseEvents() before saveCache(to:)."
      case .passDidNotFinish:
        "The model's generation task ended without reporting a stop reason. This indicates an internal contract violation in the underlying inference loop. Try the request again; if the error persists, please file a bug report."
    }
  }
}
