// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import LMResponseParserMLX
import MLXLMCommon
import Testing

// End-to-end integration tests that download a small MLX model and drive
// it through the bridge. Each test uses `IntegrationTestModels` to load
// the container at most once per test run; subsequent tests against the
// same model reuse the cached container.
//
// **Running these tests.** MLX's Metal shader library
// (`default.metallib`) is built by Xcode-specific build phases, not by
// `swift test`, so plain `swift test` cannot invoke MLX kernels even on
// macOS. The suite is therefore gated behind the
// `LMRESPONSE_PARSER_INTEGRATION_TESTS=1` environment variable, and is
// expected to be driven through Xcode (Cmd-U or
// `xcodebuild test -scheme LMResponseParserMLX`) where the metallib
// gets bundled correctly. swift-tokenizers' Benchmarks target uses the
// same gate-then-run-from-Xcode pattern.
//
// First-run cost: ~400 MB download for `mlx-community/Qwen3-0.6B-4bit`,
// stored in the standard Hugging Face cache (`~/.cache/huggingface/hub`)
// so it's reused by the `hf` CLI and other tools. Set `HF_OFFLINE=1` to
// run against an already-cached snapshot without checking the hub.
//
// The suite is `.serialized` so the model is loaded once at most and
// generation passes don't compete for GPU memory.

private let models = IntegrationTestModels()

@Suite("Integration — Qwen3-0.6B", .serialized)
struct Qwen3IntegrationTests {
  // MARK: Lifecycle envelope shape

  @Test
  func `streamResponseEvents emits exactly one response.created and one response.completed`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let session = makeSession(fixture, instructions: "Be brief.", thinking: false)

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "What is 2+2? Answer with just the number."),
    )

    let createdCount = events.count(where: {
      if case .responseCreated = $0 { return true }
      return false
    })
    let completedCount = events.count(where: {
      if case .responseCompleted = $0 { return true }
      return false
    })
    let inProgressCount = events.count(where: {
      if case .responseInProgress = $0 { return true }
      return false
    })

    #expect(createdCount == 1, "Expected exactly one response.created, got \(createdCount)")
    #expect(inProgressCount == 1, "Expected exactly one response.in_progress, got \(inProgressCount)")
    #expect(completedCount == 1, "Expected exactly one response.completed, got \(completedCount)")
  }

  @Test
  func `resp_… ID is stable across all events of a turn`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let session = makeSession(fixture, instructions: "Be brief.", thinking: false)

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "Say 'ok' once."),
    )

    let envelopeIds = events.compactMap { event -> String? in
      switch event {
        case let .responseCreated(e): return e.response.id
        case let .responseInProgress(e): return e.response.id
        case let .responseCompleted(e): return e.response.id
        default: return nil
      }
    }
    #expect(!envelopeIds.isEmpty)
    #expect(Set(envelopeIds).count == 1, "All envelope events should share one resp_… ID; got \(Set(envelopeIds))")
    if let first = envelopeIds.first {
      #expect(first.hasPrefix("resp_"), "Envelope ID should have resp_ prefix; got \(first)")
    }
  }

  @Test
  func `sequence_number is strictly monotonic across the whole turn`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let session = makeSession(fixture, instructions: "Be brief.", thinking: false)

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "Reply with 'ok'."),
    )

    let sequences = events.map(\.sequenceNumber)
    for (i, n) in sequences.enumerated() where i > 0 {
      #expect(n == sequences[i - 1] + 1, "sequence #\(i) (\(n)) does not follow \(sequences[i - 1])")
    }
  }

  // MARK: Content shape

  @Test
  func `Model emits at least one message item with non-empty text`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let session = makeSession(fixture, instructions: "Be brief.", thinking: false)

    let (items, text) = try await collectMessageText(
      session.streamResponseEvents(prompt: "Reply with the single word 'ok'."),
    )

    let messageItems = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m }
      return nil
    }
    #expect(!messageItems.isEmpty, "Expected at least one message item, got items: \(items)")
    #expect(!text.isEmpty, "Expected non-empty assistant text, got items: \(items)")
  }

  @Test
  func `Qwen3 thinking-mode prompt produces a reasoning item before the message`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    // Qwen3's chat template injects a `<think>` opener when thinking
    // mode is enabled, so the model's output begins inside a
    // reasoning block. We bump maxTokens so the model has budget to
    // close the reasoning and emit at least the start of the message.
    let session = makeSession(
      fixture,
      instructions: "Think step by step, then answer.",
      thinking: true,
      maxTokens: 800,
    )

    let (items, _) = try await collectMessageText(
      session.streamResponseEvents(prompt: "What is 7 times 8?"),
    )

    let kinds = items.map { item in
      switch item {
        case .message: "message"
        case .functionCall: "functionCall"
        case .reasoning: "reasoning"
        case .functionCallOutput: "functionCallOutput"
      }
    }
    #expect(kinds.contains("reasoning"), "Expected at least one reasoning item; got \(kinds)")
    if let firstNonReasoning = kinds.firstIndex(where: { $0 != "reasoning" }),
       let firstReasoning = kinds.firstIndex(of: "reasoning")
    {
      #expect(firstReasoning < firstNonReasoning, "Reasoning should precede non-reasoning items; got \(kinds)")
    }
  }

  // MARK: Final response shape

  @Test
  func `Clean-stop turn carries status=completed and a non-empty output[]`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let session = makeSession(fixture, instructions: "Be brief.", thinking: false)

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "Reply with one word."),
    )
    guard case let .responseCompleted(completed) = events.last else {
      Issue.record("Expected responseCompleted as last event"); return
    }
    // With thinking off and a one-word reply, the model should emit
    // its EOS within the default token budget, mapping to
    // `.completed` on the bridge side. `.incomplete` would mean the
    // model hit the cap, which is a bug in the test fixture (token
    // budget too tight) rather than the bridge.
    #expect(
      completed.response.status == .completed,
      "Expected status=.completed; got \(String(describing: completed.response.status))",
    )
    #expect(!completed.response.output.isEmpty, "Expected non-empty output[]; got \(completed.response.output)")
  }

  // MARK: Multi-turn KV-cache reuse

  @Test
  func `Multi-turn conversation reuses KV cache across turns`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    // Thinking mode off so the model answers directly within the
    // token budget instead of burning it on `<think>`.
    let session = makeSession(
      fixture,
      instructions: "Be brief. Remember context across turns.",
      thinking: false,
    )

    // Turn 1: establish a fact.
    _ = try await collectMessageText(
      session.streamResponseEvents(prompt: "My name is Alice. Reply with 'noted'."),
    )

    // Turn 2: should be able to recall the fact (which means the
    // cache is carrying state from turn 1; if we'd thrown it away,
    // the model would have no idea).
    let (_, text) = try await collectMessageText(
      session.streamResponseEvents(prompt: "What is my name? Answer with just the name."),
    )
    #expect(
      text.lowercased().contains("alice"),
      "Expected 'alice' in turn-2 response (KV-cache reuse), got: \(text)",
    )
  }

  // MARK: Helpers

  private func makeSession(
    _ fixture: LoadedFixture,
    instructions: String? = nil,
    thinking: Bool,
    maxTokens: Int = 256,
  ) -> ResponseChatSession {
    ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: instructions,
      generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
      additionalContext: ["enable_thinking": thinking],
    )
  }
}
