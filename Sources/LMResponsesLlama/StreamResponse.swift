// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses
import os

private let log = Logger(
  subsystem: "org.depasquale.lm-responses",
  category: "bridge.Llama",
)

/// Stream Responses-API events from one Llama generation pass.
///
/// Drives the parser, the streaming detokenizer, and the response-scoped
/// envelope through a single helper. Callers that own KV-cache lifecycle
/// themselves manage it on the `LlamaContext` directly (the cache is
/// implicit in the context).
///
/// Setup failures (vocab mismatch) surface as synchronous throws; the
/// returned `events` stream is non-throwing.
///
/// `modelName` is explicit for the response envelope. Format inference
/// (`ResponseFormat.infer`) reads `architecture` and `vocabSize` from the
/// context's `LlamaModel` — the GGUF carries everything the bridge needs,
/// so callers don't pass `modelType` or a `config.json` dict.
///
/// Tools live on `config.tools` so the parser (which needs them to recognize
/// function-call grammar and halt tokens) and the response envelope (which
/// echoes them back as `response.tools`) read from the same source.
public func streamResponseEvents(
  on context: LlamaContext,
  prompt: [Int32],
  parameters: GenerateParameters = .init(),
  modelName: String,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String> = [],
  priorOutput: String? = nil,
) throws -> ResponseStreamHandle<ResponseStreamingEvent> {
  try runSinglePass(
    on: context,
    prompt: prompt,
    parameters: parameters,
    modelName: modelName,
    format: format,
    config: config,
    extraEOSTokens: extraEOSTokens,
    priorOutput: priorOutput,
    mapStream: { events in events },
  )
}

/// Stream Responses-API items from one Llama generation pass.
///
/// Delivers a fresh `[ResponseOutputItem]` snapshot on every chunk. Backed
/// by the package-level `ResponseItemsAccumulator` — each yield is the
/// cumulative state at that point in the stream.
public func streamResponseItems(
  on context: LlamaContext,
  prompt: [Int32],
  parameters: GenerateParameters = .init(),
  modelName: String,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String> = [],
  priorOutput: String? = nil,
) throws -> ResponseStreamHandle<[ResponseOutputItem]> {
  let accumulator = TurnItemsBox()
  return try runSinglePass(
    on: context,
    prompt: prompt,
    parameters: parameters,
    modelName: modelName,
    format: format,
    config: config,
    extraEOSTokens: extraEOSTokens,
    priorOutput: priorOutput,
    mapStream: { events in
      accumulator.ingest(events)
      return [accumulator.snapshot]
    },
  )
}

/// Stream plain assistant text from one Llama generation pass – the
/// text-delta projection of
/// ``streamResponseEvents(on:prompt:parameters:modelName:format:config:extraEOSTokens:priorOutput:)``.
public func streamText(
  on context: LlamaContext,
  prompt: [Int32],
  parameters: GenerateParameters = .init(),
  modelName: String,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String> = [],
  priorOutput: String? = nil,
) throws -> ResponseStreamHandle<String> {
  try runSinglePass(
    on: context,
    prompt: prompt,
    parameters: parameters,
    modelName: modelName,
    format: format,
    config: config,
    extraEOSTokens: extraEOSTokens,
    priorOutput: priorOutput,
    mapStream: { events in
      events.compactMap { event in
        if case let .outputTextDelta(e) = event { return e.delta }
        return nil
      }
    },
  )
}

// MARK: Multimodal overloads

/// Multimodal variant. Same shape as the text version, but the prompt is a
/// ``/Llama/MultimodalInput`` (text + ordered media chunks) and the engine is a
/// ``/Llama/LlamaMtmdContext``. mtmd handles tokenization internally — the
/// adapter doesn't pre-tokenize the prompt.
public func streamResponseEvents(
  on context: LlamaMtmdContext,
  input: MultimodalInput,
  parameters: GenerateParameters = .init(),
  modelName: String,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String> = [],
  priorOutput: String? = nil,
) throws -> ResponseStreamHandle<ResponseStreamingEvent> {
  try runMtmdSinglePass(
    on: context,
    input: input,
    parameters: parameters,
    modelName: modelName,
    format: format,
    config: config,
    extraEOSTokens: extraEOSTokens,
    priorOutput: priorOutput,
    mapStream: { events in events },
  )
}

public func streamResponseItems(
  on context: LlamaMtmdContext,
  input: MultimodalInput,
  parameters: GenerateParameters = .init(),
  modelName: String,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String> = [],
  priorOutput: String? = nil,
) throws -> ResponseStreamHandle<[ResponseOutputItem]> {
  let accumulator = TurnItemsBox()
  return try runMtmdSinglePass(
    on: context,
    input: input,
    parameters: parameters,
    modelName: modelName,
    format: format,
    config: config,
    extraEOSTokens: extraEOSTokens,
    priorOutput: priorOutput,
    mapStream: { events in
      accumulator.ingest(events)
      return [accumulator.snapshot]
    },
  )
}

/// Multimodal text projection – plain assistant text from one mtmd pass.
public func streamText(
  on context: LlamaMtmdContext,
  input: MultimodalInput,
  parameters: GenerateParameters = .init(),
  modelName: String,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String> = [],
  priorOutput: String? = nil,
) throws -> ResponseStreamHandle<String> {
  try runMtmdSinglePass(
    on: context,
    input: input,
    parameters: parameters,
    modelName: modelName,
    format: format,
    config: config,
    extraEOSTokens: extraEOSTokens,
    priorOutput: priorOutput,
    mapStream: { events in
      events.compactMap { event in
        if case let .outputTextDelta(e) = event { return e.delta }
        return nil
      }
    },
  )
}

/// Multimodal variant taking a pre-prepared input + the start of
/// evaluation (for KV prefix reuse). Used by ``MultimodalResponseChatSession``
/// to skip re-encoding chunks (typically image embeddings) that the
/// previous turn already left in the cache.
public func streamResponseEvents(
  on context: LlamaMtmdContext,
  prepared: MtmdPreparedInput,
  startingAtChunk: Int,
  nPast: LlamaPosition,
  promptText: String,
  parameters: GenerateParameters = .init(),
  modelName: String,
  format: ResponseFormat? = nil,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String> = [],
  priorOutput: String? = nil,
) throws -> ResponseStreamHandle<ResponseStreamingEvent> {
  try runMtmdPreparedPass(
    on: context,
    prepared: prepared,
    startingAtChunk: startingAtChunk,
    nPast: nPast,
    promptText: promptText,
    parameters: parameters,
    modelName: modelName,
    format: format,
    config: config,
    extraEOSTokens: extraEOSTokens,
    priorOutput: priorOutput,
    mapStream: { events in events },
  )
}

private func runMtmdSinglePass<Element: Sendable>(
  on context: LlamaMtmdContext,
  input: MultimodalInput,
  parameters: GenerateParameters,
  modelName: String,
  format: ResponseFormat?,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String>,
  priorOutput: String?,
  mapStream: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
) throws -> ResponseStreamHandle<Element> {
  let tokenizer = LibLlamaTokenizer(model: context.textContext.model)
  // For prompt-boundary inspection we approximate by tokenizing the
  // prompt's text portion via libllama. The media markers tokenize as
  // ordinary text — close enough for parsers that look at the trailing
  // characters of the rendered prompt rather than at marker positions.
  let promptTextIds: [Int] = (try? tokenizer.encode(text: input.prompt, addSpecialTokens: false)) ?? []
  let (resolvedFormat, emitterBox) = try makePassEmitter(
    promptTextIds: promptTextIds,
    format: format,
    modelName: modelName,
    config: config,
    priorOutput: priorOutput,
    tokenizer: tokenizer,
  )

  let pass = try runPass(
    on: context,
    input: input,
    parameters: parameters,
    format: resolvedFormat,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
  )

  return driveStreamHandle(
    pass: pass,
    emitterBox: emitterBox,
    logLabel: "Llama-mtmd",
    mapStream: mapStream,
  )
}

private func runSinglePass<Element: Sendable>(
  on context: LlamaContext,
  prompt: [Int32],
  parameters: GenerateParameters,
  modelName: String,
  format: ResponseFormat?,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String>,
  priorOutput: String?,
  mapStream: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
) throws -> ResponseStreamHandle<Element> {
  let tokenizer = LibLlamaTokenizer(model: context.model)
  let promptInts = prompt.map(Int.init)
  let (resolvedFormat, emitterBox) = try makePassEmitter(
    promptTextIds: promptInts,
    format: format,
    modelName: modelName,
    config: config,
    priorOutput: priorOutput,
    tokenizer: tokenizer,
  )

  // Throw before minting the stream so setup failures (vocab mismatch)
  // surface synchronously rather than as the first iteration event.
  let pass = try runPass(
    on: context,
    promptTokens: prompt,
    parameters: parameters,
    format: resolvedFormat,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
  )

  return driveStreamHandle(
    pass: pass,
    emitterBox: emitterBox,
    logLabel: "Llama",
    mapStream: mapStream,
  )
}

private func runMtmdPreparedPass<Element: Sendable>(
  on context: LlamaMtmdContext,
  prepared: MtmdPreparedInput,
  startingAtChunk: Int,
  nPast: LlamaPosition,
  promptText: String,
  parameters: GenerateParameters,
  modelName: String,
  format: ResponseFormat?,
  config: ResponseStreamConfig,
  extraEOSTokens: Set<String>,
  priorOutput: String?,
  mapStream: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
) throws -> ResponseStreamHandle<Element> {
  let tokenizer = LibLlamaTokenizer(model: context.textContext.model)
  let promptTextIds: [Int] = (try? tokenizer.encode(text: promptText, addSpecialTokens: false)) ?? []
  let (resolvedFormat, emitterBox) = try makePassEmitter(
    promptTextIds: promptTextIds,
    format: format,
    modelName: modelName,
    config: config,
    priorOutput: priorOutput,
    tokenizer: tokenizer,
  )
  let promptSeedIds = Array(promptTextIds.suffix(7))

  let pass = try runPass(
    on: context,
    prepared: prepared,
    startingAtChunk: startingAtChunk,
    nPast: nPast,
    promptSeedIds: promptSeedIds,
    parameters: parameters,
    format: resolvedFormat,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
  )

  return driveStreamHandle(
    pass: pass,
    emitterBox: emitterBox,
    logLabel: "Llama-mtmd (prepared)",
    mapStream: mapStream,
  )
}

// MARK: Shared pass-driver helpers

/// Builds the parser + `EmitterBox` from the parameters all three pass
/// variants share. `promptTextIds` varies per caller (raw prompt for the
/// text path; tokenized rendered text for the mtmd paths).
private func makePassEmitter(
  promptTextIds: [Int],
  format: ResponseFormat?,
  modelName: String,
  config: ResponseStreamConfig,
  priorOutput: String?,
  tokenizer: LibLlamaTokenizer,
) throws -> (resolvedFormat: ResponseFormat, emitterBox: EmitterBox) {
  let resolvedFormat = format
    ?? ResponseFormat.infer(
      modelName: modelName,
      modelType: tokenizer.model.architecture ?? "",
      modelConfig: ["vocab_size": Int(tokenizer.model.vocabSize)],
    )
    ?? .json
  let effectivePriorOutput = try resolvedFormat.combinedPriorOutput(
    fromPromptTokens: promptTextIds,
    tokenizer: tokenizer,
    generatedPriorOutput: priorOutput,
  )
  let parser = resolvedFormat.makeParser(
    tokenizer: tokenizer,
    tools: config.tools,
    priorOutput: effectivePriorOutput,
  )
  let emitterBox = EmitterBox(emitter: ResponseStreamEmitter(parser: parser, config: config))
  return (resolvedFormat, emitterBox)
}

/// Shared post-stream driver. Each pass-stream variant constructs its
/// own `pass` via the appropriate `runPass` overload (text, mtmd,
/// mtmd-prepared) and feeds it here for the common processor task +
/// finalize + handle plumbing. `logLabel` is what shows up in the OS
/// log on engine error or silent finish, so callers pass the variant
/// name (`"Llama"`, `"Llama-mtmd"`, `"Llama-mtmd (prepared)"`).
private func driveStreamHandle<Element: Sendable>(
  pass: ResponseStreamHandle<PassOutput>,
  emitterBox: EmitterBox,
  logLabel: String,
  mapStream: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
) -> ResponseStreamHandle<Element> {
  let (outStream, continuation) = AsyncStream<Element>.makeStream()

  let startEvents = emitterBox.start()
  for batch in mapStream(startEvents) {
    continuation.yield(batch)
  }

  let responseBox = LastResponseBox()

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

        case let .failed(engineError):
          // The low-level API's stream is non-throwing by design
          // (see ResponseStreamHandle's doc), so we can't propagate
          // the engine error to the consumer here. Log it and let
          // the stream finish silently — the silent-finish fault log
          // below would have fired anyway, so this strictly improves
          // diagnostics by logging the real cause first.
          log.error(
            "\(logLabel) stream failed: \(engineError.localizedDescription, privacy: .public)",
          )
          lastFinishInfo = nil
      }
    }

    await pass.awaitCleanup()

    guard let info = lastFinishInfo else {
      if !Task.isCancelled {
        log.fault(
          "\(logLabel) pass ended without a .finished record; closing stream silently",
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
    awaitCleanup: { await processor.value },
    finalResponse: {
      await processor.value
      return responseBox.current
    },
  )
}

private final class EmitterBox: @unchecked Sendable {
  private let lock = NSLock()
  private let emitter: ResponseStreamEmitter

  init(emitter: ResponseStreamEmitter) {
    self.emitter = emitter
  }

  func start() -> [ResponseStreamingEvent] {
    lock.lock(); defer { lock.unlock() }
    return emitter.start()
  }

  func process(text: String, tokenIds: [Int]) -> [ResponseStreamingEvent] {
    lock.lock(); defer { lock.unlock() }
    return emitter.process(text: text, tokenIds: tokenIds)
  }

  func finalize(info: FinishInfo) -> [ResponseStreamingEvent] {
    lock.lock(); defer { lock.unlock() }
    return emitter.finalize(info: info)
  }
}
