// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import MLX
import MLXLMCommon

/// One detokenized chunk plus its underlying token IDs, or the terminal
/// completion record for a single MLX generation pass.
///
/// Internal to the bridge – consumers see ``ResponseStreamingEvent`` or
/// ``ResponseOutputItem``, not raw token-level output.
enum PassOutput {
  /// One detokenized chunk produced by `NaiveStreamingDetokenizer`.
  /// `tokenIds` are the IDs whose decoded form is exactly `text`.
  case chunk(text: String, tokenIds: [Int])

  /// The pass finished. Carries per-pass MLX completion data, distinct
  /// from the parser library's turn-aggregated ``FinishInfo``.
  case finished(PassFinishInfo)
}

/// Per-pass MLX completion data. Distinct from the parser library's
/// ``FinishInfo`` (which the bridge constructs by aggregating multiple
/// `PassFinishInfo`s for the terminal `response.completed` event).
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
        let caseName = format.swiftCaseName
        return """
        The loaded tokenizer's vocabulary does not contain \(list), \
        which \(caseName) requires. The loaded checkpoint or tokenizer \
        source is likely not a \(caseName) match.
        """
    }
  }
}

private extension ResponseFormat {
  /// Stable user-facing case name. Computed via an explicit switch so the
  /// rendering is independent of `String(describing:)` /
  /// `CustomStringConvertible`, and so any future addition of an
  /// associated value to a `ResponseFormat` case forces this site to
  /// be revisited rather than producing a noisy `harmony(profile: …)`
  /// rendering in error output.
  var swiftCaseName: String {
    switch self {
      case .hermes: "hermes"
      case .qwen: "qwen"
      case .qwen3Xml: "qwen3Xml"
      case .deepseekR1: "deepseekR1"
      case .deepseekV3: "deepseekV3"
      case .deepseekV31: "deepseekV31"
      case .deepseekV32: "deepseekV32"
      case .mistral: "mistral"
      case .llama3: "llama3"
      case .pythonic: "pythonic"
      case .lfm2: "lfm2"
      case .olmo3: "olmo3"
      case .olmo3Thinking: "olmo3Thinking"
      case .phi4Mini: "phi4Mini"
      case .phiReasoning: "phiReasoning"
      case .gemmaFunctionCall: "gemmaFunctionCall"
      case .harmony: "harmony"
      case .gemma4: "gemma4"
      case .kimiK2: "kimiK2"
      case .kimiK2Thinking: "kimiK2Thinking"
      case .miniMaxM2: "miniMaxM2"
      case .miniMax: "miniMax"
      case .glm4: "glm4"
      case .glm4Thinking: "glm4Thinking"
      case .longcat: "longcat"
      case .granite: "granite"
      case .granite20bFc: "granite20bFc"
      case .granite4: "granite4"
      case .internlm: "internlm"
      case .jamba: "jamba"
      case .hunyuanA13B: "hunyuanA13B"
      case .magistral: "magistral"
      case .xlam: "xlam"
      case .seedOss: "seedOss"
      case .step3p5: "step3p5"
      case .ernie: "ernie"
      case .ernieThinking: "ernieThinking"
      case .json: "json"
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

  let processor = Task { [dropIds, adapter] in
    // Use the parser-library `NaiveStreamingDetokenizer` (which takes
    // `ParserTokenizer`) rather than `MLXLMCommon`'s version. The
    // parser-library detokenizer explicitly passes
    // `skipSpecialTokens: false` and guards against negative diffs
    // produced by HF post-processors (e.g., `Lstrip`/`Rstrip`
    // re-applied once a following token is present), which would
    // trap `String.suffix(_:)` in the MLXLMCommon variant.
    var detokenizer = LMResponseParser.NaiveStreamingDetokenizer(tokenizer: adapter)
    var pendingTokenIds: [Int] = []

    for await event in mlxStream {
      switch event {
        case let .token(id):
          if dropIds.contains(id) {
            continue
          }
          detokenizer.append(token: id)
          pendingTokenIds.append(id)
          if let chunk = detokenizer.next() {
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

    // If generation ended mid-multi-byte UTF-8 scalar, the
    // detokenizer is still holding the leading bytes (it withholds
    // anything ending in U+FFFD until the trailing byte arrives).
    // Those bytes are dropped at end-of-stream by design – this
    // matches `MLXLMCommon.TextToolTokenLoopHandler.onGenerationEnd`,
    // which doesn't flush its detokenizer either. Well-behaved
    // models stop on a whole-scalar boundary; this only bites
    // pathological models or sharp `length` truncation.
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
