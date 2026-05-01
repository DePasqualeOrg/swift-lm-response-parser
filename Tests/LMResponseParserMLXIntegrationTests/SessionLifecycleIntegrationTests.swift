// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import LMResponseParserMLX
import MLXLMCommon
import Testing

// Session-lifecycle integration tests: history-restoring init, KV-cache
// save/restore round-trip, cancellation, and concurrent-call
// serialization. Together they pin the parity surface that
// `ResponseChatSession` shares with `MLXLMCommon.ChatSession`. Multi-turn
// cache reuse and the envelope-shape invariants are covered by
// `IntegrationTests.swift`; the helper-layer cache reuse is covered by
// `LowLevelHelpersIntegrationTests.swift`.

private let models = IntegrationTestModels()

@Suite("Integration — session lifecycle", .serialized)
struct SessionLifecycleIntegrationTests {
  @Test
  func `History-restoring init recalls a fact seeded via history:`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)

    let history: [Chat.Message] = [
      .user("My favorite color is teal. Reply with 'noted'."),
      .assistant("noted"),
    ]

    let session = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: "Be brief. Remember context across turns.",
      history: history,
      generateParameters: GenerateParameters(maxTokens: 128, temperature: 0),
      additionalContext: ["enable_thinking": false],
    )

    let (_, text) = try await collectMessageText(
      session.streamResponseEvents(
        prompt: "What is my favorite color? Answer with just the color.",
      ),
    )

    #expect(
      text.lowercased().contains("teal"),
      "Expected 'teal' in turn-1 response (history re-hydration), got: \(text)",
    )
  }

  @Test
  func `saveCache(to:) round-trip recovers conversation state in a new session`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)

    let session1 = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: "Be brief. Remember context across turns.",
      generateParameters: GenerateParameters(maxTokens: 128, temperature: 0),
      additionalContext: ["enable_thinking": false],
    )

    // Turn 1 to seed cache state.
    _ = try await collectMessageText(
      session1.streamResponseEvents(
        prompt: "My favorite city is Lisbon. Reply with 'noted'.",
      ),
    )

    let cacheURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("response-session-cache-\(UUID().uuidString).safetensors")
    try await session1.saveCache(to: cacheURL)
    defer { try? FileManager.default.removeItem(at: cacheURL) }

    let (loadedCache, _) = try loadPromptCache(url: cacheURL)

    // The cache encodes both the system prompt and turn 1, so the new
    // session must omit `instructions:` to avoid double-prefixing.
    let session2 = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      cache: loadedCache,
      generateParameters: GenerateParameters(maxTokens: 128, temperature: 0),
      additionalContext: ["enable_thinking": false],
    )

    let (_, text) = try await collectMessageText(
      session2.streamResponseEvents(
        prompt: "What is my favorite city? Answer with just the city.",
      ),
    )

    #expect(
      text.contains("Lisbon"),
      "Expected 'Lisbon' in restored-cache response, got: \(text)",
    )
  }

  @Test
  func `Breaking out of iteration ends the stream without responseCompleted and leaves the session usable`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)

    let session = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: "Be brief.",
      // Big enough budget that the model is still streaming when we
      // break. If it managed to finish before our break, the test
      // would be vacuous.
      generateParameters: GenerateParameters(maxTokens: 512, temperature: 0),
      additionalContext: ["enable_thinking": false],
    )

    var observedDelta = false
    var observedCompleted = false
    var observedError: Error?
    do {
      for try await event in session.streamResponseEvents(
        prompt: "List the first thirty prime numbers, one per line.",
      ) {
        switch event {
          case .outputTextDelta:
            observedDelta = true
          case .responseCompleted:
            observedCompleted = true
          default:
            break
        }
        if observedDelta {
          break
        }
      }
    } catch {
      observedError = error
    }

    // Breaking the iterator triggers the producer's `onTermination`,
    // which `task.cancel()`s the outer turn task. The bridge treats
    // that as a clean stream end (no error), so the consumer should
    // fall out of the for-loop without entering the catch block.
    #expect(observedError == nil, "Expected clean stream end on cancel; got \(String(describing: observedError))")

    #expect(observedDelta, "Test only meaningful if the consumer saw at least one delta")
    #expect(!observedCompleted, "Expected no responseCompleted before the break")

    // Cache lock must release after the producer's pass cleanup. If
    // it leaked, this synchronize() (or the second turn below) would
    // deadlock and the test runner would time out.
    await session.synchronize()

    let (_, text) = try await collectMessageText(
      session.streamResponseEvents(prompt: "Reply with the single word 'recovered'."),
    )
    #expect(!text.isEmpty, "Session unusable after early-break cancel; got empty turn-2 text")
  }

  @Test
  func `Concurrent streamResponseEvents calls serialize through the cache lock`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let session = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: "Be brief.",
      generateParameters: GenerateParameters(maxTokens: 64, temperature: 0),
      additionalContext: ["enable_thinking": false],
    )

    // Construct both streams on the test's isolation domain before
    // handing them to concurrent collectors. AsyncThrowingStream is
    // Sendable when its element is, so this avoids reading the
    // (non-Sendable) `session` from two child tasks. The two producer
    // Tasks were already created by the calls below — they queue on
    // the cache lock the moment they reach `cache.update`.
    let firstStream = session.streamResponseEvents(prompt: "Reply with the single word 'first'.")
    let secondStream = session.streamResponseEvents(prompt: "Reply with the single word 'second'.")

    async let firstEvents = collectEvents(firstStream)
    async let secondEvents = collectEvents(secondStream)

    let (a, b) = try await (firstEvents, secondEvents)

    for (label, events) in [("first", a), ("second", b)] {
      let createds = events.count(where: {
        if case .responseCreated = $0 { return true }
        return false
      })
      let completeds = events.count(where: {
        if case .responseCompleted = $0 { return true }
        return false
      })

      #expect(createds == 1, "\(label) turn must have exactly one responseCreated; got \(createds)")
      #expect(completeds == 1, "\(label) turn must have exactly one responseCompleted; got \(completeds)")

      // Sequence numbers within a single turn are strictly monotonic.
      // If the lock leaked and turns interleaved, we'd see a
      // sequence-number gap between events of one turn (because the
      // other turn's events would consume intervening counter values
      // in the shared envelope — but each turn gets its own envelope,
      // so non-monotonic also serves as a turn-isolation check).
      let sequences = events.map(\.sequenceNumber)
      for i in 1 ..< sequences.count {
        #expect(
          sequences[i] == sequences[i - 1] + 1,
          "\(label) sequence #\(i) (\(sequences[i])) does not follow \(sequences[i - 1])",
        )
      }
    }
  }
}
