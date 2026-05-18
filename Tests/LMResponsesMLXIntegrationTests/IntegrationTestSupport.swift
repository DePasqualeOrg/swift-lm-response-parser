// Copyright © Anthony DePasquale

import Foundation
import HFAPI
import LMResponses
import LMResponsesMLX
import MLXLMCommon
@preconcurrency import Tokenizers

// Integration-test infrastructure for `LMResponsesMLX`. Mirrors the
// pattern used by the `LMResponsesMLXTests` target's swift-tokenizers
// dependency: a small downloader bridge over `HFAPI.HubClient`, a
// tokenizer loader that returns `swift-tokenizers` `AutoTokenizer` results
// wrapped as `MLXLMCommon.Tokenizer`, and an actor that caches loaded
// `ModelContainer`s across tests within a run.
//
// Snapshots land in `HFAPI.HubCache.default` (the standard
// `~/.cache/huggingface/hub` location), so models you've already
// downloaded with `hf` or any other HF tool are reused without
// re-downloading.

// MARK: - Errors

enum IntegrationTestError: LocalizedError {
  case invalidRepositoryID(String)

  var errorDescription: String? {
    switch self {
      case let .invalidRepositoryID(id):
        "Invalid Hugging Face repository ID '\(id)' (expected 'namespace/name')."
    }
  }
}

// MARK: - Downloader bridge

/// Conforms `HFAPI.HubClient` to `MLXLMCommon.Downloader` so
/// `loadModelContainer(from:using:configuration:...)` can drive
/// HFAPI-backed downloads without the consumer wiring this up by hand.
///
/// Mirrors the bridge that `MLXHuggingFace`'s `#hubDownloader()` macro
/// generates (which targets `huggingface/swift-huggingface`'s `HubClient`
/// instead of the swift-hf-api fork).
struct HFAPIDownloader: MLXLMCommon.Downloader {
  let client: HubClient

  init(client: HubClient = HubClient()) {
    self.client = client
  }

  func download(
    id: String,
    revision: String?,
    matching patterns: [String],
    useLatest _: Bool,
    progressHandler: @Sendable @escaping (Foundation.Progress) -> Void,
  ) async throws -> URL {
    guard let repoID = HFAPI.Repo.ID(rawValue: id) else {
      throw IntegrationTestError.invalidRepositoryID(id)
    }
    let revision = revision ?? "main"
    return try await client.downloadSnapshot(
      of: repoID,
      revision: revision,
      matching: patterns,
      progressHandler: progressHandler,
    )
  }
}

// MARK: - Tokenizer loader bridge

/// Loads a tokenizer from a local directory via `Tokenizers.AutoTokenizer`
/// and wraps the result in a `MLXLMCommon.Tokenizer` adapter.
struct SwiftTokenizersLoader: MLXLMCommon.TokenizerLoader {
  func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
    let upstream = try await Tokenizers.AutoTokenizer.from(directory: directory)
    return SwiftTokenizersBridge(upstream)
  }
}

/// Forwards every `MLXLMCommon.Tokenizer` call to a
/// `Tokenizers.Tokenizer`. Translates `applyChatTemplate` between the
/// two protocols' parameter shapes (swift-tokenizers takes more knobs
/// than MLXLMCommon exposes; we pin the rest to their MLXLMCommon
/// defaults: generation prompt on, no truncation, no max length).
struct SwiftTokenizersBridge: MLXLMCommon.Tokenizer {
  private let upstream: any Tokenizers.Tokenizer

  init(_ upstream: any Tokenizers.Tokenizer) {
    self.upstream = upstream
  }

  func encode(text: String, addSpecialTokens: Bool) throws -> [Int] {
    try upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
  }

  func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
    try upstream.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
  }

  func convertTokenToId(_ token: String) -> Int? {
    upstream.convertTokenToId(token)
  }

  func convertIdToToken(_ id: Int) -> String? {
    upstream.convertIdToToken(id)
  }

  var bosToken: String? {
    upstream.bosToken
  }

  var eosToken: String? {
    upstream.eosToken
  }

  var unknownToken: String? {
    upstream.unknownToken
  }

  func applyChatTemplate(
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    additionalContext: [String: any Sendable]?,
  ) throws -> [Int] {
    do {
      return try upstream.applyChatTemplate(
        messages: messages,
        chatTemplate: nil,
        addGenerationPrompt: true,
        truncation: false,
        maxLength: nil,
        tools: tools,
        additionalContext: additionalContext,
      )
    } catch Tokenizers.TokenizerError.missingChatTemplate {
      throw MLXLMCommon.TokenizerError.missingChatTemplate
    }
  }
}

// MARK: - Model fixture loader

/// Identifier + the bridge inputs (`modelType`, `modelConfig`) needed to
/// drive `ResponseFormat.infer(modelName:modelType:modelConfig:)` for that
/// model. `extraEOSTokens` is for fixtures that need extra halt tokens
/// beyond what the format auto-injects (typically empty — the bridge
/// auto-injects format-specific halts itself).
struct ModelFixture {
  let id: String
  let modelType: String
  let modelConfigPatterns: [String]
  let extraEOSTokens: Set<String>

  init(
    id: String,
    modelType: String,
    modelConfigPatterns: [String] = ["*.json"],
    extraEOSTokens: Set<String> = [],
  ) {
    self.id = id
    self.modelType = modelType
    self.modelConfigPatterns = modelConfigPatterns
    self.extraEOSTokens = extraEOSTokens
  }
}

extension ModelFixture {
  /// Qwen3 0.6B 4-bit. Smallest model that exercises:
  /// - The `.qwen` parser format (Hermes-style `<tool_call>{json}</tool_call>`).
  /// - Reasoning extraction via `<think>` markers.
  /// - End-to-end generation under ~400 MB on disk.
  static let qwen3_0_6b = ModelFixture(
    id: "mlx-community/Qwen3-0.6B-4bit",
    modelType: "qwen3",
  )

  /// Llama 3.2 1B 4-bit. Exercises `.llama3` parser. ~700 MB on disk.
  static let llama3_2_1b = ModelFixture(
    id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
    modelType: "llama",
  )

  /// Cohere Command R7B 12-2024, 4-bit. The only published Cohere2
  /// checkpoint that's small enough for integration testing
  /// (~4.5 GB on disk). Exercises the `.cohereCmd3` parser:
  /// `<|START_THINKING|>` / `<|START_ACTION|>` / `<|START_RESPONSE|>`
  /// marker state machine, JSON-array tool calls, and `<co>…</co: …>`
  /// citations.
  ///
  /// **TODO: cmd4 path.** When a smaller `cohere2_vision` (Cohere2
  /// Vision) checkpoint becomes available — or a 4-bit MLX conversion
  /// of `CohereLabs/command-a-vision-07-2025` lands — add a sibling
  /// fixture with `modelType: "cohere2_vision"` to exercise the
  /// `.cohereCmd4` dispatch path end-to-end (start-in-reasoning via
  /// `<|START_THINKING|>` prompt boundary, `<|START_TEXT|>` markers,
  /// `Cohere2VisionForConditionalGeneration` arch). `cohere2_moe`
  /// (`Cohere2MoeForCausalLM`) is registered in vLLM as a future
  /// architecture; no checkpoint is published yet.
  static let cohereR7B = ModelFixture(
    id: "mlx-community/c4ai-command-r7b-12-2024-4bit",
    modelType: "cohere2",
  )
}

/// Loads a `ModelContainer` from `IntegrationTestModels` and reads
/// `config.json` from the snapshot directory so the bridge has the
/// raw config dict it needs for format inference.
struct LoadedFixture {
  let container: ModelContainer
  let modelType: String
  let modelConfig: [String: any Sendable]
}

/// Caches loaded `ModelContainer`s and their config dicts across tests
/// within a run. Loading is expensive (snapshot download + GPU memory
/// upload), so each fixture is loaded at most once.
actor IntegrationTestModels {
  private let downloader: any MLXLMCommon.Downloader
  private let tokenizerLoader: any MLXLMCommon.TokenizerLoader

  private var loaded: [String: Task<LoadedFixture, Error>] = [:]

  init(
    downloader: any MLXLMCommon.Downloader = HFAPIDownloader(),
    tokenizerLoader: any MLXLMCommon.TokenizerLoader = SwiftTokenizersLoader(),
  ) {
    self.downloader = downloader
    self.tokenizerLoader = tokenizerLoader
  }

  func fixture(_ spec: ModelFixture) async throws -> LoadedFixture {
    if let task = loaded[spec.id] {
      return try await task.value
    }
    let downloader = downloader
    let tokenizerLoader = tokenizerLoader
    let task = Task {
      try await Self.load(
        spec: spec,
        downloader: downloader,
        tokenizerLoader: tokenizerLoader,
      )
    }
    loaded[spec.id] = task
    return try await task.value
  }

  private static func load(
    spec: ModelFixture,
    downloader: any MLXLMCommon.Downloader,
    tokenizerLoader: any MLXLMCommon.TokenizerLoader,
  ) async throws -> LoadedFixture {
    print("Loading \(spec.id)…")
    let configuration = ModelConfiguration(
      id: spec.id,
      extraEOSTokens: spec.extraEOSTokens,
    )
    let container = try await loadModelContainer(
      from: downloader,
      using: tokenizerLoader,
      configuration: configuration,
      progressHandler: logProgress(spec.id),
    )
    // Re-fetch the snapshot to read config.json. The snapshot is
    // already on disk from the loader call above, so this hits the
    // cache and returns immediately.
    let snapshot = try await downloader.download(
      id: spec.id,
      revision: nil,
      matching: spec.modelConfigPatterns,
      useLatest: false,
      progressHandler: { _ in },
    )
    let configURL = snapshot.appending(path: "config.json")
    let modelConfig = try Self.readConfigJSON(at: configURL)
    print("Loaded \(spec.id)")
    return LoadedFixture(
      container: container,
      modelType: spec.modelType,
      modelConfig: modelConfig,
    )
  }

  private static func readConfigJSON(at url: URL) throws -> [String: any Sendable] {
    let data = try Data(contentsOf: url)
    let json = try JSONSerialization.jsonObject(with: data)
    guard let dict = json as? [String: any Sendable] else { return [:] }
    return dict
  }
}

// MARK: - Tokenizer-only fixtures

/// Identifier for a chat-template round-trip fixture. Downloads only the
/// tokenizer files needed by `Tokenizers.AutoTokenizer.from(directory:)`,
/// not model weights.
struct TokenizerFixture {
  let id: String

  static let mistral7BInstructV0_3 = TokenizerFixture(id: "mistralai/Mistral-7B-Instruct-v0.3")
  static let ministral8B = TokenizerFixture(id: "mistralai/Ministral-8B-Instruct-2410")
  static let hermes3Llama31 = TokenizerFixture(id: "NousResearch/Hermes-3-Llama-3.1-8B")
  static let llama3_2_1bInstruct = TokenizerFixture(id: "meta-llama/Llama-3.2-1B-Instruct")
}

/// Loads tokenizers via `Tokenizers.AutoTokenizer` from a snapshot of just
/// the tokenizer files. The snapshot lands in
/// `HFAPI.HubCache.default` so repeated runs hit the cache. Each fixture
/// is loaded at most once per actor instance.
actor IntegrationTokenizers {
  // Patterns broad enough to cover BPE / Unigram / WordPiece / SentencePiece
  // tokenizer payloads plus the chat-template sidecars.
  private static let tokenizerPatterns: [String] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
  ]

  private let client: HubClient
  private var loaded: [String: Task<any Tokenizers.Tokenizer, Error>] = [:]

  init(client: HubClient = HubClient()) {
    self.client = client
  }

  func tokenizer(_ spec: TokenizerFixture) async throws -> any Tokenizers.Tokenizer {
    if let task = loaded[spec.id] {
      return try await task.value
    }
    let client = client
    let task = Task {
      try await Self.load(spec: spec, client: client)
    }
    loaded[spec.id] = task
    return try await task.value
  }

  private static func load(
    spec: TokenizerFixture,
    client: HubClient,
  ) async throws -> any Tokenizers.Tokenizer {
    print("Loading tokenizer for \(spec.id)…")
    guard let repoID = HFAPI.Repo.ID(rawValue: spec.id) else {
      throw IntegrationTestError.invalidRepositoryID(spec.id)
    }
    let snapshot = try await client.downloadSnapshot(
      of: repoID,
      revision: "main",
      matching: tokenizerPatterns,
      progressHandler: logProgress(spec.id),
    )
    let tokenizer = try await Tokenizers.AutoTokenizer.from(directory: snapshot)
    print("Loaded tokenizer for \(spec.id)")
    return tokenizer
  }
}

// MARK: - Progress logging

func logProgress(_ label: String) -> @Sendable (Foundation.Progress) -> Void {
  let lock = NSLock()
  nonisolated(unsafe) var lastThreshold = -1
  return { progress in
    let pct = Int(progress.fractionCompleted * 100)
    let threshold = pct / 10
    lock.lock()
    let shouldPrint = threshold > lastThreshold
    if shouldPrint { lastThreshold = threshold }
    lock.unlock()
    if shouldPrint {
      print("  \(label): \(pct)%")
    }
  }
}

// MARK: - Stream helpers

/// Collect every event from a `streamResponseEvents(prompt:)` call into an
/// array. Most assertions look at the full event sequence, so collecting
/// up front is the more readable shape.
func collectEvents(
  _ stream: AsyncThrowingStream<ResponseStreamingEvent, Error>,
) async throws -> [ResponseStreamingEvent] {
  var collected: [ResponseStreamingEvent] = []
  for try await event in stream {
    collected.append(event)
  }
  return collected
}

/// Collect output items + concatenated text from a stream. Useful for
/// "did the model say anything sensible?" assertions.
func collectMessageText(
  _ stream: AsyncThrowingStream<ResponseStreamingEvent, Error>,
) async throws -> (items: [ResponseOutputItem], messageText: String) {
  var accumulator = ResponseItemsAccumulator()
  var text = ""
  for try await event in stream {
    accumulator.ingest(event)
    if case let .outputTextDelta(e) = event {
      text += e.delta
    }
  }
  return (accumulator.items, text)
}
