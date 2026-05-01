// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import MLX
import MLXLMCommon
import os

private let log = Logger(
  subsystem: "org.depasquale.lm-response-parser",
  category: "bridge",
)

/// Stream Responses-API events from one MLX generation pass.
///
/// Drives the parser, the streaming detokenizer, and the response-scoped
/// envelope through a single helper. Callers that own KV-cache lifecycle
/// themselves pass a `cache` and reuse it after awaiting the returned
/// handle's ``ResponseStreamHandle/awaitCleanup()``.
///
/// Setup failures (bad input shape, model/format vocab mismatch) surface as
/// synchronous throws; the returned `events` stream is non-throwing.
///
/// `modelType` and `modelConfig` are explicit because mlx-swift-lm's
/// `LLMModelFactory._load` decodes `model_type` and the raw `config.json`
/// dict, uses them for dispatch, and discards both. The bridge needs both
/// to drive ``/LMResponseParser/ResponseFormat/infer(modelName:modelType:modelConfig:)``.
///
/// - TODO: Drop both parameters if `ModelConfiguration` ever exposes the
///   decoded `model_type` and raw config dict directly – they become
///   derivable from the passed-in model context.
///
/// Tools live on `config.tools` so the parser (which needs them to recognize
/// function-call grammar and halt tokens) and the response envelope (which
/// echoes them back as `response.tools`) read from the same source.
public func streamResponseEvents(
  input: LMInput,
  cache: [KVCache]? = nil,
  parameters: GenerateParameters,
  context: ModelContext,
  modelType: String? = nil,
  modelConfig: [String: any Sendable]? = nil,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  priorOutput: String? = nil,
  wiredMemoryTicket: WiredMemoryTicket? = nil,
) throws -> ResponseStreamHandle<ResponseStreamingEvent> {
  try runSinglePass(
    input: input,
    cache: cache,
    parameters: parameters,
    context: context,
    modelType: modelType,
    modelConfig: modelConfig,
    format: format,
    config: config,
    priorOutput: priorOutput,
    wiredMemoryTicket: wiredMemoryTicket,
    mapStream: { events in events },
  )
}

/// Stream Responses-API items from one MLX generation pass.
///
/// Delivers a fresh `[ResponseOutputItem]` snapshot on every chunk. Backed
/// by the package-level `ResponseItemsAccumulator` – each yield is the
/// cumulative state at that point in the stream. Callers that want events
/// instead of items should use
/// ``streamResponseEvents(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``.
public func streamResponseItems(
  input: LMInput,
  cache: [KVCache]? = nil,
  parameters: GenerateParameters,
  context: ModelContext,
  modelType: String? = nil,
  modelConfig: [String: any Sendable]? = nil,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  priorOutput: String? = nil,
  wiredMemoryTicket: WiredMemoryTicket? = nil,
) throws -> ResponseStreamHandle<[ResponseOutputItem]> {
  let accumulator = ItemsAccumulatorBox()
  return try runSinglePass(
    input: input,
    cache: cache,
    parameters: parameters,
    context: context,
    modelType: modelType,
    modelConfig: modelConfig,
    format: format,
    config: config,
    priorOutput: priorOutput,
    wiredMemoryTicket: wiredMemoryTicket,
    mapStream: { events in
      accumulator.ingest(events)
      return [accumulator.snapshot]
    },
  )
}

public extension ModelContainer {
  /// `ModelContainer`-based variant of
  /// ``streamResponseEvents(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``.
  /// Pulls the `ModelContext` out of the container so callers don't have
  /// to manage the `perform { … }` boundary and the non-`Sendable` context
  /// themselves.
  ///
  /// Stateless one-shot only: there is no `cache` parameter, since cache
  /// reuse implies caller-owned cache lifecycle. For multi-turn or
  /// cache-reuse, use ``ResponseChatSession`` (which owns the cache across
  /// turns) or drop down to the context-based free function.
  func streamResponseEvents(
    input: consuming sending LMInput,
    parameters: GenerateParameters,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    format: ResponseFormat? = nil,
    config: ResponseStreamConfig,
    priorOutput: String? = nil,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
  ) async throws -> ResponseStreamHandle<ResponseStreamingEvent> {
    let inputBox = SendableBox(input)
    let context = await perform { SendableBox($0) }.consume()
    return try LMResponseParserMLX.streamResponseEvents(
      input: inputBox.consume(),
      parameters: parameters,
      context: context,
      modelType: modelType,
      modelConfig: modelConfig,
      format: format,
      config: config,
      priorOutput: priorOutput,
      wiredMemoryTicket: wiredMemoryTicket,
    )
  }

  /// `ModelContainer`-based variant of
  /// ``streamResponseItems(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``.
  /// See ``streamResponseEvents(input:parameters:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``
  /// for the stateless one-shot rationale.
  func streamResponseItems(
    input: consuming sending LMInput,
    parameters: GenerateParameters,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    format: ResponseFormat? = nil,
    config: ResponseStreamConfig,
    priorOutput: String? = nil,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
  ) async throws -> ResponseStreamHandle<[ResponseOutputItem]> {
    let inputBox = SendableBox(input)
    let context = await perform { SendableBox($0) }.consume()
    return try LMResponseParserMLX.streamResponseItems(
      input: inputBox.consume(),
      parameters: parameters,
      context: context,
      modelType: modelType,
      modelConfig: modelConfig,
      format: format,
      config: config,
      priorOutput: priorOutput,
      wiredMemoryTicket: wiredMemoryTicket,
    )
  }
}

/// Reference wrapper so the items snapshot survives across the
/// `mapStream` closure invocations inside `runSinglePass`.
private final class ItemsAccumulatorBox: @unchecked Sendable {
  private var inner = ResponseItemsAccumulator()
  var snapshot: [ResponseOutputItem] {
    inner.items
  }

  func ingest(_ events: [ResponseStreamingEvent]) {
    inner.ingest(events)
  }
}

/// Internal implementation shared by both helpers. The two differ only in
/// how they map a batch of `ResponseStreamingEvent`s onto the output
/// stream's element type.
private func runSinglePass<Element: Sendable>(
  input: LMInput,
  cache: [KVCache]?,
  parameters: GenerateParameters,
  context: ModelContext,
  modelType: String?,
  modelConfig: [String: any Sendable]?,
  format: ResponseFormat?,
  config: ResponseStreamConfig,
  priorOutput: String?,
  wiredMemoryTicket: WiredMemoryTicket?,
  mapStream: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
) throws -> ResponseStreamHandle<Element> {
  let adapter = MLXTokenizerAdapter(context.tokenizer)
  let resolvedFormat = format
    ?? ResponseFormat.infer(
      modelName: context.configuration.name,
      modelType: modelType ?? "",
      modelConfig: modelConfig ?? [:],
    )
    ?? .json

  let parser = resolvedFormat.makeParser(
    tokenizer: adapter,
    tools: config.tools,
    priorOutput: priorOutput,
  )
  let emitterBox = EmitterBox(emitter: ResponseStreamEmitter(parser: parser, config: config))

  // Throw before minting the stream so a setup failure (vocab mismatch
  // from `validateTokenizerVocabulary`, or `TokenIterator.init` failure)
  // surfaces synchronously rather than as the first iteration event.
  let pass = try runPass(
    on: context,
    input: input,
    cache: cache,
    parameters: parameters,
    format: resolvedFormat,
    adapter: adapter,
    wiredMemoryTicket: wiredMemoryTicket,
  )

  let (outStream, continuation) = AsyncStream<Element>.makeStream()

  let startEvents = emitterBox.start()
  for batch in mapStream(startEvents) {
    continuation.yield(batch)
  }

  let responseBox = ResponseBox()

  let processor = Task {
    var usage = UsageAccumulator()
    var lastFinishInfo: PassFinishInfo?

    for await event in pass {
      if Task.isCancelled { break }
      switch event {
        case let .chunk(text, tokenIds):
          let parserEvents = emitterBox.process(text: text, tokenIds: tokenIds)
          usage.observe(events: parserEvents, tokenCount: tokenIds.count)
          for batch in mapStream(parserEvents) {
            continuation.yield(batch)
          }

        case let .finished(info):
          lastFinishInfo = info
          usage.addPassInput(info.inputTokens)
          usage.addPassOutput(info.outputTokens)
      }
    }

    await pass.awaitCleanup()

    // The terminal response event is only synthesized when MLX
    // delivered a `.finished` record. A consumer-cancelled iterator
    // (processor cancelled → `pass` ends without `.finished`)
    // closes silently. The non-cancelled "no `.finished`" branch is
    // a contract violation by the underlying generation pass –
    // surface it via Logger.fault and close the stream silently
    // since the public surface is non-throwing.
    guard let info = lastFinishInfo else {
      if !Task.isCancelled {
        log.fault(
          "MLX pass ended without a .finished record; closing stream silently",
        )
      }
      continuation.finish()
      return
    }

    let terminal = emitterBox.finalize(info: usage.finalInfo(finishReason: info.finishReason))
    for event in terminal {
      if let response = event.terminalResponse {
        responseBox.set(response)
      }
    }
    for batch in mapStream(terminal) {
      continuation.yield(batch)
    }
    continuation.finish()
  }

  continuation.onTermination = { _ in
    processor.cancel()
  }

  return ResponseStreamHandle(
    stream: outStream,
    // The processor body unconditionally awaits `pass.awaitCleanup()`
    // on every exit path before exiting; awaiting `processor.value`
    // therefore implies the pass has drained.
    awaitCleanup: {
      await processor.value
    },
    finalResponse: {
      await processor.value
      return responseBox.get()
    },
  )
}

/// Single-writer / single-reader holding pen for the terminal ``Response``
/// snapshot. The processor task writes once before `continuation.finish()`;
/// readers go through `await processor.value` first, so the read happens
/// after the write. `@unchecked Sendable` is acceptable because that
/// synchronization is provided externally by the task barrier.
private final class ResponseBox: @unchecked Sendable {
  private var response: Response?
  func set(_ response: Response) {
    self.response = response
  }

  func get() -> Response? {
    response
  }
}

/// Reference wrapper so the emitter survives across the `mapStream` and
/// processor-task boundary. The emitter is single-task-mutated; this
/// just satisfies Swift's data-race checker.
private final class EmitterBox: @unchecked Sendable {
  private let emitter: ResponseStreamEmitter

  init(emitter: ResponseStreamEmitter) {
    self.emitter = emitter
  }

  func start() -> [ResponseStreamingEvent] {
    emitter.start()
  }

  func process(text: String, tokenIds: [Int]) -> [ResponseStreamingEvent] {
    emitter.process(text: text, tokenIds: tokenIds)
  }

  func finalize(info: FinishInfo) -> [ResponseStreamingEvent] {
    emitter.finalize(info: info)
  }
}
