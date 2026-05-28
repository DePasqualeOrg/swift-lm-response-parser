// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses
import os

/// One detokenized chunk plus its underlying token IDs, the terminal
/// completion record, or a fatal error for a single Llama generation
/// pass. `.failed` carries any error the engine stream threw — the
/// consumer is expected to re-throw it so the user sees the root
/// cause instead of a generic "did not finish" downstream.
enum PassOutput {
  case chunk(text: String, tokenIds: [Int])
  case finished(PassFinishInfo)
  case failed(any Error & Sendable)
}

/// Per-pass Llama completion data. Distinct from the parser library's
/// ``FinishInfo`` (which the bridge constructs by aggregating multiple
/// `PassFinishInfo`s for the terminal response event).
struct PassFinishInfo {
  let inputTokens: Int
  let outputTokens: Int
  let finishReason: FinishReason
}

public enum BridgeError: LocalizedError, Equatable {
  case tokenizerMissingRequiredTokens(format: ResponseFormat, missing: [String])

  public var errorDescription: String? {
    switch self {
      case let .tokenizerMissingRequiredTokens(format, missing):
        let list = missing.sorted().joined(separator: ", ")
        let caseName = String(describing: format)
        return """
        The loaded tokenizer's vocabulary does not contain \(list), \
        which \(caseName) requires. The loaded checkpoint or tokenizer \
        source is likely not a \(caseName) match.
        """
    }
  }
}

/// Drive one Llama generation pass through the parser-friendly chunk shape.
///
/// Owns the streaming detokenizer, the pending-token buffer, the stop-token
/// drop set, and the cleanup barrier. Knows nothing about parsers — parser
/// ownership belongs to the caller (low-level helpers via
/// `ResponseStreamEmitter`, session via per-pass parser construction).
///
/// Auto-injects the format's `requiredExtraEOSTokens` into the engine's
/// `extraStopTokens` set, mirroring vLLM and SGLang's unconditional
/// injection for `gpt_oss`. Callers don't need to pre-register
/// format-specific halt tokens. Throws
/// ``BridgeError/tokenizerMissingRequiredTokens(format:missing:)`` only when
/// the loaded tokenizer's vocabulary doesn't contain those tokens — a
/// model-format mismatch the caller can't fix by changing config.
func runPass(
  on context: LlamaContext,
  promptTokens: [Int32],
  parameters: GenerateParameters,
  format: ResponseFormat,
  tokenizer: LibLlamaTokenizer,
  extraEOSTokens: Set<String> = [],
  cachedPrefixLength: Int = 0,
  tokenIdsSink: (@Sendable (Int) -> Void)? = nil,
) throws -> ResponseStreamHandle<PassOutput> {
  let policy = format.stopTokenPolicy
  try validateTokenizerVocabulary(policy: policy, format: format, tokenizer: tokenizer)

  let stopConfig = buildStopConfig(
    policy: policy,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
  )

  let llamaStream = context.generate(
    prompt: promptTokens,
    parameters: parameters,
    extraStopTokens: stopConfig.extraStopAsInt32,
    includeStopTokenInStream: policy.includeStopToken,
    cachedPrefixLength: cachedPrefixLength,
  )

  let promptSeedIds: [Int] = Array(promptTokens.suffix(7)).map(Int.init)
  return makeProcessorHandle(
    stream: llamaStream,
    promptSeedIds: promptSeedIds,
    tokenizer: tokenizer,
    dropIds: stopConfig.dropIds,
    yieldedStopIds: stopConfig.yieldedStopIds,
    tokenIdsSink: tokenIdsSink,
  )
}

/// Mtmd variant — same processor pipeline, just sourced from
/// `LlamaMtmdContext.generate(input:)` instead of the text engine. The
/// engine's `input.media` doesn't carry literal text tokens at our level
/// (mtmd handles tokenization internally), so the prompt-seed buffer is
/// derived from the text portion of the prompt for detokenizer warm-up.
func runPass(
  on context: LlamaMtmdContext,
  input: MultimodalInput,
  parameters: GenerateParameters,
  format: ResponseFormat,
  tokenizer: LibLlamaTokenizer,
  extraEOSTokens: Set<String> = [],
) throws -> ResponseStreamHandle<PassOutput> {
  let policy = format.stopTokenPolicy
  try validateTokenizerVocabulary(policy: policy, format: format, tokenizer: tokenizer)

  let stopConfig = buildStopConfig(
    policy: policy,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
  )

  let mtmdStream = context.generate(
    input: input,
    parameters: parameters,
    extraStopTokens: stopConfig.extraStopAsInt32,
    includeStopTokenInStream: policy.includeStopToken,
  )

  // Detokenizer warm-up from the text-only tail of the prompt — what
  // libllama would emit for the text spans around media markers.
  let promptSeedIds: [Int] = if let tail = try? tokenizer.encode(text: input.prompt, addSpecialTokens: false).suffix(7) {
    Array(tail)
  } else {
    []
  }

  return makeProcessorHandle(
    stream: mtmdStream,
    promptSeedIds: promptSeedIds,
    tokenizer: tokenizer,
    dropIds: stopConfig.dropIds,
    yieldedStopIds: stopConfig.yieldedStopIds,
  )
}

/// Mtmd variant taking a pre-prepared input. Used by the multimodal
/// session for KV prefix reuse — it tokenizes once via
/// ``/Llama/LlamaMtmdContext/prepare(input:)``, matches the chunk signatures
/// against last turn's, and starts evaluation at the first divergent
/// chunk.
func runPass(
  on context: LlamaMtmdContext,
  prepared: MtmdPreparedInput,
  startingAtChunk: Int,
  nPast: LlamaPosition,
  promptSeedIds: [Int],
  parameters: GenerateParameters,
  format: ResponseFormat,
  tokenizer: LibLlamaTokenizer,
  extraEOSTokens: Set<String> = [],
) throws -> ResponseStreamHandle<PassOutput> {
  let policy = format.stopTokenPolicy
  try validateTokenizerVocabulary(policy: policy, format: format, tokenizer: tokenizer)

  let stopConfig = buildStopConfig(
    policy: policy,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
  )

  let mtmdStream = context.generate(
    prepared: prepared,
    startingAtChunk: startingAtChunk,
    nPast: nPast,
    parameters: parameters,
    extraStopTokens: stopConfig.extraStopAsInt32,
    includeStopTokenInStream: policy.includeStopToken,
  )

  return makeProcessorHandle(
    stream: mtmdStream,
    promptSeedIds: promptSeedIds,
    tokenizer: tokenizer,
    dropIds: stopConfig.dropIds,
    yieldedStopIds: stopConfig.yieldedStopIds,
  )
}

private struct StopConfig {
  let extraStopAsInt32: Set<Int32>
  let yieldedStopIds: Set<Int32>
  let dropIds: Set<Int>
}

private func buildStopConfig(
  policy: ResponseFormatStopTokenPolicy,
  tokenizer: LibLlamaTokenizer,
  extraEOSTokens: Set<String>,
) -> StopConfig {
  var allExtra = extraEOSTokens
  allExtra.formUnion(policy.requiredExtraEOSTokens)

  // libllama stops on either EOS or EOT and yields whichever fired when
  // includeStopTokenInStream=true. Llama 3.x's chat terminator is EOT,
  // not EOS, so both need to be in the drop set.
  var effectiveStopIds: Set<Int> = []
  if let eos = tokenizer.eosTokenId { effectiveStopIds.insert(eos) }
  if let eot = tokenizer.eotTokenId { effectiveStopIds.insert(eot) }
  for token in allExtra {
    if let id = tokenizer.convertTokenToId(token) {
      effectiveStopIds.insert(id)
    }
  }

  let includedIds: Set<Int> = Set(
    policy.includedStopTokens.compactMap(tokenizer.convertTokenToId),
  )

  return StopConfig(
    extraStopAsInt32: Set(effectiveStopIds.map(Int32.init)),
    yieldedStopIds: Set(includedIds.map(Int32.init)),
    dropIds: effectiveStopIds.subtracting(includedIds),
  )
}

/// Shared post-stream pipeline: token-drop filter → streaming detokenizer
/// → `PassOutput` continuation. Used by both the text and mtmd `runPass`
/// helpers.
///
/// `tokenIdsSink` fires once per token that survives the drop-ids filter,
/// before any detokenizer machinery runs. This is the source of truth for
/// "what the engine just decoded" — independent of whether the streaming
/// detokenizer accepted the token or had to reset its prefix buffer.
/// Callers tracking KV-cache contents (sessions) hook this; callers
/// that only care about text chunks don't pass it.
private func makeProcessorHandle(
  stream: AsyncThrowingStream<TokenEvent, Error>,
  promptSeedIds: [Int],
  tokenizer: LibLlamaTokenizer,
  dropIds: Set<Int>,
  yieldedStopIds: Set<Int32>,
  tokenIdsSink: (@Sendable (Int) -> Void)? = nil,
) -> ResponseStreamHandle<PassOutput> {
  let (outStream, continuation) = AsyncStream<PassOutput>.makeStream()
  let sourceTask = LlamaPassTask(stream: stream)

  let processor = Task { [dropIds, tokenizer, promptSeedIds, yieldedStopIds, tokenIdsSink] in
    var detokenizer = tokenizer.streamingDetokenizer(initialTokenIds: promptSeedIds)
    var pendingTokenIds: [Int] = []

    do {
      for try await event in sourceTask.stream {
        switch event {
          case let .token(rawId):
            let id = Int(rawId)
            if dropIds.contains(id), !yieldedStopIds.contains(rawId) {
              continue
            }
            // Fire the KV-tracking sink before pendingTokenIds /
            // detokenizer logic. The detokenizer can reset its prefix
            // buffer mid-stream (StreamingDetokenizerError recovery)
            // and discard tokens it had accumulated for a future chunk
            // — those tokens were already decoded into KV by the engine,
            // so anything tracking KV state needs to see them here, not
            // when (or whether) they reach a yielded chunk.
            tokenIdsSink?(id)
            pendingTokenIds.append(id)

            let chunk: String?
            // Tokens dropped by the detokenizer below are still in
            // KV and were already routed to `tokenIdsSink`; emit them
            // to the parser as chunks with empty text so multi-token
            // marker sequences (Qwen, Llama 3.x) still trigger state
            // transitions even when no character payload reaches the
            // detokenizer.
            var dropForParser: [Int] = []
            do {
              chunk = try detokenizer.consume(id)
            } catch let error as StreamingDetokenizerError {
              runPassLogger.warning(
                "Streaming prefix violated, resetting detokenizer: \(error.localizedDescription, privacy: .public)",
              )
              detokenizer = tokenizer.streamingDetokenizer()
              do {
                chunk = try detokenizer.consume(id)
                pendingTokenIds = [id]
              } catch {
                runPassLogger.error(
                  "Detokenizer retry failed; dropping \(pendingTokenIds.count) token(s): \(error.localizedDescription, privacy: .public)",
                )
                dropForParser = pendingTokenIds
                pendingTokenIds.removeAll(keepingCapacity: true)
                chunk = nil
              }
            } catch {
              runPassLogger.error(
                "Detokenizer failed; dropping token \(id): \(error.localizedDescription, privacy: .public)",
              )
              pendingTokenIds.removeLast()
              dropForParser = [id]
              chunk = nil
            }

            if !dropForParser.isEmpty {
              continuation.yield(.chunk(text: "", tokenIds: dropForParser))
            }
            if let chunk {
              continuation.yield(.chunk(text: chunk, tokenIds: pendingTokenIds))
              pendingTokenIds.removeAll(keepingCapacity: true)
            }

          case let .info(info):
            // Flush any bytes the detokenizer was holding back (last
            // token ended mid-multibyte-UTF8 scalar) so the partial
            // bytes appear as a replacement-char chunk instead of being
            // silently dropped. Failure here drops the trailing bytes
            // but doesn't abort the pass — the stream still terminates
            // cleanly via `.finished`.
            do {
              if let leftover = try detokenizer.flush(), !leftover.isEmpty {
                continuation.yield(.chunk(text: leftover, tokenIds: pendingTokenIds))
                pendingTokenIds.removeAll(keepingCapacity: true)
              }
            } catch {
              runPassLogger.error(
                "Detokenizer flush failed; dropping trailing bytes: \(error.localizedDescription, privacy: .public)",
              )
            }
            continuation.yield(.finished(PassFinishInfo(
              inputTokens: info.promptTokenCount,
              outputTokens: info.generationTokenCount,
              finishReason: translate(info.stopReason),
            )))
        }
      }
    } catch {
      runPassLogger.error(
        "Generation stream failed: \(error.localizedDescription, privacy: .public)",
      )
      // Surface the error through the stream so the consumer re-throws
      // it instead of seeing a silent finish + downstream
      // `passDidNotFinish`. `any Error` in a catch context satisfies
      // `Sendable` under Swift 6 implicit-conformance, so the upcast
      // to `any Error & Sendable` is always valid.
      continuation.yield(.failed(error as any Error & Sendable))
    }

    continuation.finish()
  }

  continuation.onTermination = { _ in
    processor.cancel()
  }

  return ResponseStreamHandle<PassOutput>(stream: outStream) {
    await processor.value
  }
}

/// Verify the loaded tokenizer's vocabulary contains every token the
/// format requires — both halt tokens and parser-observation tokens.
func validateTokenizerVocabulary(
  policy: ResponseFormatStopTokenPolicy,
  format: ResponseFormat,
  tokenizer: LibLlamaTokenizer,
) throws {
  let allRequired = policy.requiredExtraEOSTokens.union(policy.includedStopTokens)
  if allRequired.isEmpty { return }
  let missing = allRequired.filter { token in
    tokenizer.convertTokenToId(token) == nil
  }
  if !missing.isEmpty {
    throw BridgeError.tokenizerMissingRequiredTokens(
      format: format,
      missing: Array(missing),
    )
  }
}

private func translate(_ reason: StopReason) -> FinishReason {
  switch reason {
    case .endOfSequence: .stop
    case .extraStopToken: .stop
    case .maxTokensReached: .length
    case .cancelled: .cancelled
  }
}

/// Holds the engine's stream so the processor task can iterate it.
/// Wrapped in a class so the closure captures don't need to box the
/// non-Sendable stream itself.
private final class LlamaPassTask: @unchecked Sendable {
  let stream: AsyncThrowingStream<TokenEvent, Error>
  init(stream: AsyncThrowingStream<TokenEvent, Error>) {
    self.stream = stream
  }
}

private let runPassLogger = Logger(
  subsystem: "org.depasquale.lm-responses",
  category: "RunPass.Llama",
)
