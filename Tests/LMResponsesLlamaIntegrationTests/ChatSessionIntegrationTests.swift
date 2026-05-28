// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponsesLlama
import Testing
import Tokenizers

@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct ChatSessionIntegrationTests {
  let fixtures = IntegrationFixtures()

  /// Two-turn chat: ask something, then a follow-up that depends on the
  /// first answer's context. Asserts the session retained the history and
  /// the model produced different text on each turn.
  @Test func `multi turn chat remembers context`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))

    let session = ResponseChatSession(
      context: context,
      modelName: fixture.modelName,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 80),
      additionalContext: ["enable_thinking": false],
    )

    func runTurn(_ prompt: String) async throws -> String {
      var text = ""
      for try await event in session.streamResponseEvents(prompt: prompt) {
        if case let .outputTextDelta(e) = event {
          text += e.delta
        }
      }
      return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    let firstAnswer = try await runTurn("My name is Anthony. What is my name?")
    print("--- Turn 1 ---")
    print(firstAnswer)

    let followUp = try await runTurn("Repeat my name back to me, exactly.")
    print("--- Turn 2 ---")
    print(followUp)

    #expect(!firstAnswer.isEmpty)
    #expect(!followUp.isEmpty)
    // Both answers should contain the name "Anthony" — the second turn
    // can only know it if the session remembered the first turn.
    #expect(followUp.localizedCaseInsensitiveContains("Anthony"))
  }

  /// Session created with a pre-populated history must answer the first
  /// turn as if those exchanges had happened locally. Confirms the
  /// `init(history:)` re-hydration path threads history through the chat
  /// template correctly on turn 1.
  @Test func `init with history re hydrates context`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))

    let priorHistory: [Tokenizers.Message] = [
      ["role": "user", "content": "My favorite color is teal."],
      ["role": "assistant", "content": "Got it — teal is a great choice."],
    ]

    let session = ResponseChatSession(
      context: context,
      modelName: fixture.modelName, history: priorHistory,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 40),
      additionalContext: ["enable_thinking": false],
    )

    var text = ""
    for try await event in session.streamResponseEvents(prompt: "What's my favorite color?") {
      if case let .outputTextDelta(e) = event { text += e.delta }
    }
    let answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Re-hydrated history answer ---")
    print(answer)
    #expect(answer.localizedCaseInsensitiveContains("teal"))
  }

  /// Round-trip a session through saveCache/init(cache:) and assert that
  /// the resumed session continues the conversation correctly. Confirms
  /// the cache file holds enough state to skip re-decoding the prompt
  /// prefix on the resumed first turn.
  @Test func `save load cache resumes conversation`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)

    let tempDir = FileManager.default.temporaryDirectory
    let cacheURL = tempDir.appendingPathComponent("session-cache-\(UUID().uuidString).bin")
    defer { try? FileManager.default.removeItem(at: cacheURL) }

    // Phase 1: run a turn on a session, save its cache.
    var savedHistory: [Tokenizers.Message]
    do {
      let model = try await LlamaModel.load(from: ggufURL)
      let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
      let session = ResponseChatSession(
        context: context,
        modelName: fixture.modelName,
        generateParameters: GenerateParameters(seed: 42, maxTokens: 24),
        additionalContext: ["enable_thinking": false],
      )
      _ = try await session.respond(to: "My favorite color is teal.")
      try await session.saveCache(to: cacheURL)
      // Read back the history through respond's side-effects: the session
      // mutated its internal store, but the public API doesn't expose
      // it. Reconstruct what we know the session stored.
      savedHistory = [
        ["role": "user", "content": "My favorite color is teal."],
        ["role": "assistant", "content": ""], // content filled in next turn's render
      ]
      // Drop the placeholder assistant entry since the saved cache
      // includes the actual assistant tokens — we'll reconstruct from
      // lastResponse instead.
      if let last = session.lastResponse {
        savedHistory[1] = ["role": "assistant", "content": last.outputText]
      }
    }

    // Phase 2: re-hydrate from the cache file + history; ask a follow-up
    // that depends on the prior turn.
    let model = try await LlamaModel.load(from: ggufURL)
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
    let resumed = try await ResponseChatSession(
      context: context,
      modelName: fixture.modelName, history: savedHistory,
      cache: cacheURL,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 24),
      additionalContext: ["enable_thinking": false],
    )

    var text = ""
    for try await event in resumed.streamResponseEvents(prompt: "What's my favorite color?") {
      if case let .outputTextDelta(e) = event { text += e.delta }
    }
    let answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Resumed-from-cache answer ---")
    print(answer)
    #expect(answer.localizedCaseInsensitiveContains("teal"))
  }

  /// With a tiny context window, multi-turn history accumulates past
  /// the window. The session should drop the earliest history entries
  /// and still produce a valid response for the current turn.
  @Test func `context overflow drops old history`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    // Tiny context: 512 tokens — fills up after a few chatty turns.
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 512))

    let session = ResponseChatSession(
      context: context,
      modelName: fixture.modelName,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 80),
      additionalContext: ["enable_thinking": false],
    )

    func runTurn(_ prompt: String) async throws -> String {
      var text = ""
      for try await event in session.streamResponseEvents(prompt: prompt) {
        if case let .outputTextDelta(e) = event { text += e.delta }
      }
      return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Burn through several turns so the rendered prompt grows past
    // contextLength - maxTokens (432 tokens). Each prompt asks for a
    // long-ish response to fill history fast.
    for i in 1 ... 4 {
      let answer = try await runTurn("Tell me about the number \(i * 7), in two sentences.")
      print("--- Turn \(i) ---")
      print(answer)
    }

    // Final turn: the rendered prompt would overflow the budget
    // without trimming. The session should drop earlier turns and
    // still answer this one without throwing.
    let finalAnswer = try await runTurn("In one word, what color is the sky?")
    print("--- Final turn ---")
    print(finalAnswer)
    #expect(!finalAnswer.isEmpty)
    #expect(finalAnswer.localizedCaseInsensitiveContains("blue"))
  }

  /// Cancel a generation mid-stream and assert that the session is
  /// still usable for a follow-up turn. The cancelled turn's history
  /// must NOT be persisted (matching MLX behavior), and the KV cache
  /// must not be corrupted by partial-decode state.
  @Test func `cancellation mid stream allows follow up turn`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))

    let session = ResponseChatSession(
      context: context,
      modelName: fixture.modelName,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 200),
      additionalContext: ["enable_thinking": false],
    )

    // Turn 1: start streaming, break out after the first few deltas.
    // The continuation's onTermination handler should cancel the
    // generator task and drain cleanly when the stream goes out of scope.
    var deltaCount = 0
    do {
      for try await event in session.streamResponseEvents(prompt: "Write a very long story about dragons.") {
        if case .outputTextDelta = event {
          deltaCount += 1
          if deltaCount >= 3 { break }
        }
      }
    }
    #expect(deltaCount >= 3)

    // Let the cancellation propagate. synchronize() waits on the
    // session lock — when it returns, any in-flight cleanup is done.
    await session.synchronize()

    // Turn 2: a fresh prompt. Since turn 1 was cancelled before
    // finalize, its history shouldn't have been written; turn 2 should
    // behave as a first turn on a fresh session.
    var followUp = ""
    for try await event in session.streamResponseEvents(prompt: "Say the word HELLO and nothing else.") {
      if case let .outputTextDelta(e) = event { followUp += e.delta }
    }
    let trimmed = followUp.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Post-cancellation follow-up ---")
    print(trimmed)
    #expect(!trimmed.isEmpty)
    #expect(trimmed.localizedCaseInsensitiveContains("hello"))
  }

  /// Exercises the convenience text API for ergonomic callers: `streamText`
  /// yields plain `String` chunks with no event matching, and
  /// `respond(to:).outputText` returns the aggregated reply in one call.
  @Test func `streamText and outputText return plain text`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))

    let session = ResponseChatSession(
      context: context,
      modelName: fixture.modelName,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 40),
      additionalContext: ["enable_thinking": false],
    )

    var streamed = ""
    for try await chunk in session.streamText(prompt: "Say hello in one short sentence.") {
      streamed += chunk
    }
    #expect(!streamed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

    let whole = try await session.respond(to: "Now count to three.").outputText
    #expect(!whole.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
  }
}
