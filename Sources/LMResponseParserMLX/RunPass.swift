// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import MLX
import MLXLMCommon
import os

/// One detokenized chunk plus its underlying token IDs, or the terminal
/// completion record for a single MLX generation pass.
///
/// Internal to the bridge – consumers see ``ResponseStreamingEvent`` or
/// ``ResponseOutputItem``, not raw token-level output.
enum PassOutput {
  /// One detokenized chunk produced by `LMResponseParser.StreamingDetokenizer`.
  /// `tokenIds` are the IDs whose decoded form is exactly `text`.
  case chunk(text: String, tokenIds: [Int])

  /// The pass finished. Carries per-pass MLX completion data, distinct
  /// from the parser library's turn-aggregated ``FinishInfo``.
  case finished(PassFinishInfo)
}

/// Per-pass MLX completion data. Distinct from the parser library's
/// ``FinishInfo`` (which the bridge constructs by aggregating multiple
/// `PassFinishInfo`s for the terminal response event).
struct PassFinishInfo {
  /// This pass's prompt token count.
  let inputTokens: Int

  /// This pass's MLX-reported generation token count. Authoritative
  /// for total output tokens – counts every token MLX produced,
  /// including any stop tokens the bridge dropped before yielding.
  let outputTokens: Int

  /// Why generation stopped.
  let finishReason: FinishReason
}

/// Errors produced by the bridge's single-pass driver during configuration validation.
public enum BridgeError: LocalizedError, Equatable {
  /// The loaded tokenizer's vocabulary doesn't contain a token the
  /// format requires — either a halt token (e.g. `<|call|>` for Harmony)
  /// or a parser-observation token. The bridge auto-injects each format's
  /// halt tokens into the model's effective stop set, so the only way to
  /// reach this error is to load a tokenizer that doesn't recognize them —
  /// typically a model-format mismatch where the loaded checkpoint
  /// doesn't actually speak the selected format.
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

/// Drive one MLX generation pass through the parser-friendly chunk shape.
///
/// Owns the streaming detokenizer, the pending-token buffer, the stop-token
/// drop set, and the cleanup barrier. Knows nothing about parsers – parser
/// ownership belongs to the caller (low-level helpers via
/// `ResponseStreamEmitter`, session via per-pass parser construction).
///
/// Auto-injects the format's `requiredExtraEOSTokens` into a local copy of
/// the model configuration before generation, mirroring vLLM and SGLang's
/// unconditional injection for `gpt_oss`. Callers don't need to pre-register
/// format-specific halt tokens at model-load time. Throws
/// ``BridgeError/tokenizerMissingRequiredTokens(format:missing:)`` only when
/// the loaded tokenizer's vocabulary doesn't contain those tokens — a
/// model-format mismatch the caller can't fix by changing config.
func runPass(
  on context: ModelContext,
  input: LMInput,
  cache: [KVCache]?,
  parameters: GenerateParameters,
  format: ResponseFormat,
  adapter: MLXTokenizerAdapter,
  wiredMemoryTicket: WiredMemoryTicket? = nil,
) throws -> ResponseStreamHandle<PassOutput> {
  let policy = format.stopTokenPolicy

  try validateTokenizerVocabulary(
    policy: policy,
    format: format,
    tokenizer: context.tokenizer,
  )

  // Auto-inject the format's required halt tokens into a local copy of
  // the configuration. Mirrors vLLM and SGLang, which unconditionally
  // append Harmony's stop tokens to the sampler's stop set whenever
  // `model_type == "gpt_oss"` — the upstream HF conversion script lists
  // only `<|return|>` in `eos_token_id` and omits `<|call|>`, so without
  // this every Harmony runtime has to add it back. `ModelContext` and
  // `ModelConfiguration` are both value types, and `extraEOSTokens` is
  // a stored `Set<String>`, so the mutation is local to this pass; the
  // caller's loaded model is unaffected. (`model: any LanguageModel`
  // is a class existential — a future refactor mutating it would break
  // this isolation.)
  var augmented = context
  augmented.configuration.extraEOSTokens.formUnion(policy.requiredExtraEOSTokens)

  let mlxStopSet = effectiveStopTokenIds(
    modelConfiguration: augmented.configuration,
    tokenizer: augmented.tokenizer,
  )

  let includedIds: Set<Int> = Set(
    policy.includedStopTokens.compactMap(augmented.tokenizer.convertTokenToId),
  )
  // Drop every stop token MLX would otherwise yield except those the
  // format declares the parser needs to observe. Mirrors the contract
  // on `ResponseFormatStopTokenPolicy`: any token a parser depends on
  // must be in `includedStopTokens`, otherwise it is silently consumed
  // here. Only takes effect when `policy.includeStopToken` is true
  // (i.e. `includedIds` non-empty); for formats whose parsers operate
  // on pre-stop text only, MLX never yields stop tokens in the first
  // place and `dropIds` has no observable effect.
  let dropIds = mlxStopSet.subtracting(includedIds)

  let (mlxStream, mlxTask) = try generateTokensTask(
    input: input,
    cache: cache,
    parameters: parameters,
    context: augmented,
    includeStopToken: policy.includeStopToken,
    wiredMemoryTicket: wiredMemoryTicket,
  )

  let (outStream, continuation) = AsyncStream<PassOutput>.makeStream()

  // Seed with the prompt tail — enough to defeat decoder cleanup at the
  // prompt/generated boundary (vLLM uses `prompt_ids[-7:]`).
  let promptSeedIds = Array(input.text.tokens.asArray(Int.self).suffix(7))

  let processor = Task { [dropIds, adapter, promptSeedIds] in
    // Use the parser-library `StreamingDetokenizer` (which takes
    // `ParserTokenizer`) rather than `MLXLMCommon`'s version, so the
    // bridge sees the parser-local `StreamingDetokenizerError` for
    // recovery and the parser layer stays decoupled from any
    // particular tokenizer ecosystem.
    var detokenizer = adapter.streamingDetokenizer(initialTokenIds: promptSeedIds)
    var pendingTokenIds: [Int] = []

    do {
      for try await event in mlxStream {
        switch event {
          case let .token(id):
            if dropIds.contains(id) {
              continue
            }
            pendingTokenIds.append(id)

            // vLLM-style recovery contract for the streaming detokenizer.
            // `pendingTokenIds` tracks the token IDs whose bytes will
            // appear in the next emitted chunk; each `consume` outcome
            // demands a different mutation:
            //
            //   * Returns String  → cleared after emit.
            //   * Returns nil     → token stays pending (bytes buffered
            //                       in the detokenizer for a future chunk).
            //   * Throws prefix invariant, retry succeeds → reset
            //                       destroyed the detokenizer's buffered
            //                       bytes; prior pending tokens' bytes
            //                       are lost. Pin to [id] so the chunk's
            //                       `tokenIds` align.
            //   * Throws prefix invariant, retry also throws → reset
            //                       already lost prior pendings; the
            //                       failing token's bytes are
            //                       unrecoverable. Clear fully.
            //   * Throws other (decode pass-through) → transactional
            //                       `consume` rolled back; detokenizer
            //                       state unchanged. Drop just the
            //                       failing token.
            let chunk: String?
            do {
              chunk = try detokenizer.consume(id)
            } catch let error as StreamingDetokenizerError {
              runPassLogger.warning(
                "Streaming prefix violated, resetting detokenizer: \(error.localizedDescription, privacy: .public)",
              )
              detokenizer = adapter.streamingDetokenizer()
              do {
                chunk = try detokenizer.consume(id)
                pendingTokenIds = [id]
              } catch {
                runPassLogger.error(
                  "Detokenizer retry failed; dropping token \(id): \(error.localizedDescription, privacy: .public)",
                )
                pendingTokenIds.removeAll(keepingCapacity: true)
                chunk = nil
              }
            } catch {
              runPassLogger.error(
                "Detokenizer failed; dropping token \(id): \(error.localizedDescription, privacy: .public)",
              )
              pendingTokenIds.removeLast()
              chunk = nil
            }

            if let chunk {
              continuation.yield(.chunk(text: chunk, tokenIds: pendingTokenIds))
              pendingTokenIds.removeAll(keepingCapacity: true)
            }

          case let .info(info):
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
    }

    // If generation ended mid-multi-byte UTF-8 scalar, the
    // detokenizer is still holding the leading bytes. Those bytes
    // are dropped at end-of-stream by design – this matches
    // `MLXLMCommon`'s text token loop handler, which doesn't flush
    // its detokenizer either. Well-behaved models stop on a
    // whole-scalar boundary; this only bites pathological models
    // or sharp `length` truncation.
    await mlxTask.value
    continuation.finish()
  }

  continuation.onTermination = { _ in
    mlxTask.cancel()
    processor.cancel()
  }

  return ResponseStreamHandle<PassOutput>(stream: outStream) {
    await processor.value
    await mlxTask.value
  }
}

/// Verify the loaded tokenizer's vocabulary contains every token the
/// format requires — both halt tokens (``ResponseFormatStopTokenPolicy/requiredExtraEOSTokens``)
/// and parser-observation tokens (``ResponseFormatStopTokenPolicy/includedStopTokens``).
/// Throws ``BridgeError/tokenizerMissingRequiredTokens(format:missing:)`` when
/// any token doesn't resolve to an ID. Validating the union (rather than
/// just the halt set) covers the case where a future format declares
/// observation-only tokens disjoint from its halt set: without this check
/// such a token would be silently dropped from `includedIds` while its ID
/// might still appear in the engine's stop set via `eosTokenIds`,
/// producing a halt that the parser never observes.
func validateTokenizerVocabulary(
  policy: ResponseFormatStopTokenPolicy,
  format: ResponseFormat,
  tokenizer: any MLXLMCommon.Tokenizer,
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

/// Reconstructs the complete set of token IDs that MLX's raw-token loop
/// treats as terminal. `MLXLMCommon.buildStopTokenIds` is private and
/// covers model/tokenizer EOS plus extra EOS tokens; the loop also stops
/// on `unknownTokenId`, so include it here too. The bridge's stop-token
/// drop filter and defensive checks must see the same effective set MLX
/// sees.
func effectiveStopTokenIds(
  modelConfiguration: ModelConfiguration,
  tokenizer: any MLXLMCommon.Tokenizer,
) -> Set<Int> {
  var stopTokenIds = modelConfiguration.eosTokenIds
  if let tokenizerEOS = tokenizer.eosTokenId {
    stopTokenIds.insert(tokenizerEOS)
  }
  if let unknownId = tokenizer.unknownTokenId {
    stopTokenIds.insert(unknownId)
  }
  for token in modelConfiguration.extraEOSTokens {
    if let id = tokenizer.convertTokenToId(token) {
      stopTokenIds.insert(id)
    }
  }
  return stopTokenIds
}

private func translate(_ reason: GenerateStopReason) -> FinishReason {
  switch reason {
    case .stop: .stop
    case .length: .length
    case .cancelled: .cancelled
  }
}

private let runPassLogger = Logger(
  subsystem: "org.depasquale.lm-response-parser",
  category: "RunPass",
)
