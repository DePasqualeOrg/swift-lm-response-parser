// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses

/// Shared loop body for text and multimodal pass runners: iterates
/// the engine's chunk stream, threads each chunk through the parser,
/// rebases events onto the turn envelope's pass scope, ingests into
/// the items box, and emits batches via `transform`. Caller owns
/// parser construction + pass setup.
func drivePassLoop<Element: Sendable>(
  pass: ResponseStreamHandle<PassOutput>,
  parser: inout any ResponseFormatParser,
  envelope: ResponseTurnEnvelope,
  transform: @Sendable ([ResponseStreamingEvent]) -> [Element],
  continuation: AsyncThrowingStream<Element, Error>.Continuation,
  usageBox: UsageBox,
  itemsBox: TurnItemsBox,
) async throws -> PassFinishInfo {
  let scope = envelope.beginPass()
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

        case let .failed(engineError):
          // Surface the engine error; cleanup runs in the outer
          // catch. The alternative — silent finish + ambiguous
          // `passDidNotFinish` — loses the root cause.
          throw engineError
      }
    }

    if Task.isCancelled {
      await pass.awaitCleanup()
      scope.end()
      return PassFinishInfo(
        inputTokens: lastInfo?.inputTokens ?? 0,
        outputTokens: lastInfo?.outputTokens ?? 0,
        finishReason: .cancelled,
      )
    }

    let finalEvents = parser.finalize()
    let forwardedFinal = scope.forward(finalEvents)
    itemsBox.ingest(forwardedFinal)
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

/// Drive one generation pass through the parser and turn envelope. Each
/// chunk's events are forwarded through the pass scope (which rebases them
/// onto the turn-scoped envelope) and ingested into the turn-scoped
/// `itemsBox` so the caller can read pending tool calls from
/// `itemsBox.snapshot` after this call returns.
func runOnePass<Element: Sendable>(
  on context: LlamaContext,
  prompt: [Int32],
  tokenizer: LibLlamaTokenizer,
  generateParameters: GenerateParameters,
  modelName: String,
  tools: [ToolSpec],
  envelope: ResponseTurnEnvelope,
  transform: @Sendable ([ResponseStreamingEvent]) -> [Element],
  continuation: AsyncThrowingStream<Element, Error>.Continuation,
  usageBox: UsageBox,
  itemsBox: TurnItemsBox,
  format: ResponseFormat?,
  extraEOSTokens: Set<String>,
  cachedPrefixLength: Int = 0,
  tokenSink: (@Sendable (Int) -> Void)? = nil,
) async throws -> PassFinishInfo {
  let resolvedFormat = format
    ?? ResponseFormat.infer(
      modelName: modelName,
      modelType: context.model.architecture ?? "",
      modelConfig: ["vocab_size": Int(context.model.vocabSize)],
    )
    ?? .json

  let promptInts = prompt.map(Int.init)
  let effectivePriorOutput = try resolvedFormat.combinedPriorOutput(
    fromPromptTokens: promptInts,
    tokenizer: tokenizer,
    generatedPriorOutput: nil,
  )
  var parser = resolvedFormat.makeParser(
    tokenizer: tokenizer,
    tools: tools,
    priorOutput: effectivePriorOutput,
  )

  // Per-token sink keeps KV-tracking independent of the detokenizer's
  // chunk boundaries (see RunPass.swift::makeProcessorHandle).
  let pass = try runPass(
    on: context,
    promptTokens: prompt,
    parameters: generateParameters,
    format: resolvedFormat,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
    cachedPrefixLength: cachedPrefixLength,
    tokenIdsSink: tokenSink,
  )

  return try await drivePassLoop(
    pass: pass,
    parser: &parser,
    envelope: envelope,
    transform: transform,
    continuation: continuation,
    usageBox: usageBox,
    itemsBox: itemsBox,
  )
}

// MARK: Sendable boxes

//
// These boxes hold per-turn mutable state shared between the session
// driver task and the inner pass-processor task. Each box takes an
// `NSLock` per method so the `@unchecked Sendable` claim is provable
// from the box's own body, independent of the surrounding mutex.
// (`ResponseTurnEnvelope` carries its own internal lock — no wrapper
// box needed.)

final class UsageBox: @unchecked Sendable {
  private let lock = NSLock()
  private var usage = UsageAccumulator()

  func observe(events: [ResponseStreamingEvent], tokenCount: Int) {
    lock.lock(); defer { lock.unlock() }
    usage.observe(events: events, tokenCount: tokenCount)
  }

  func addPassInput(_ tokens: Int) {
    lock.lock(); defer { lock.unlock() }
    usage.addPassInput(tokens)
  }

  func addPassOutput(_ tokens: Int) {
    lock.lock(); defer { lock.unlock() }
    usage.addPassOutput(tokens)
  }

  func finalInfo(finishReason: FinishReason) -> FinishInfo {
    lock.lock(); defer { lock.unlock() }
    return usage.finalInfo(finishReason: finishReason)
  }
}

final class TurnItemsBox: @unchecked Sendable {
  private let lock = NSLock()
  private var inner = ResponseItemsAccumulator()
  var snapshot: [ResponseOutputItem] {
    lock.lock(); defer { lock.unlock() }
    return inner.items
  }

  func ingest(_ events: [ResponseStreamingEvent]) {
    lock.lock(); defer { lock.unlock() }
    inner.ingest(events)
  }
}

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
  case passDidNotFinish
  /// The turn's restart loop exited without any pass populating
  /// `PassFinishInfo`. Unreachable in the current flow; thrown as a
  /// typed error rather than synthesizing `.stop` so a regression
  /// that allows a non-cancellation early-break surfaces with
  /// diagnostic detail instead of zeroed-token telemetry.
  case passInfoMissing
  /// The rendered prompt — even after dropping all conversation
  /// history — exceeds the model context's available window. The
  /// associated values report the rendered token count and the
  /// budget the session was trying to keep it under (i.e.
  /// `contextLength - generateParameters.maxTokens`).
  case promptExceedsContext(promptTokens: Int, budget: Int)
  /// The chunk-signature sidecar file accompanying a multimodal KV
  /// cache was written by an incompatible library version. Callers
  /// should catch this and either re-render from scratch (drop the
  /// cache + sigs files) or refuse to load. Only thrown from
  /// ``MultimodalResponseChatSession/init(context:modelName:instructions:history:cache:generateParameters:additionalContext:tools:toolDispatch:format:extraEOSTokens:)``
  /// — the text session has no equivalent sidecar.
  case cacheVersionMismatch(found: Int, expected: Int)

  public var errorDescription: String? {
    switch self {
      case .passDidNotFinish:
        "The model's generation task ended without reporting a stop reason. Try again; if it persists, file a bug report."
      case .passInfoMissing:
        "The turn ended without any generation pass reporting back. Internal bridge invariant violated; file a bug report."
      case let .promptExceedsContext(promptTokens, budget):
        "The user's prompt + system instructions tokenize to \(promptTokens) tokens, which exceeds the \(budget)-token budget left after reserving room for generation. Trim the new message or raise the context window."
      case let .cacheVersionMismatch(found, expected):
        "Multimodal cache sidecar version mismatch: file is version \(found), this build expects version \(expected). Delete the cache files and re-render from history."
    }
  }
}
