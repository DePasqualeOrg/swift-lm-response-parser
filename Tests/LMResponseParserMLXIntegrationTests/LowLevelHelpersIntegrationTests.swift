// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import LMResponseParserMLX
import MLXLMCommon
import Testing

// Integration tests for the low-level helpers (`streamResponseEvents` /
// `streamResponseItems`) — the layer below `ResponseChatSession` that
// callers use when they own KV-cache lifecycle themselves.
//
// These exercise:
// - The `(input: LMInput, cache: [KVCache]?, parameters:, context: ModelContext, ...)`
//   shape of the helpers, with a manually-constructed cache and an
//   `LMInput` prepared via `context.processor.prepare(input:)`.
// - `ResponseStream<ResponseStreamingEvent>` and
//   `ResponseStream<[ResponseOutputItem]>` end-to-end against a
//   real model.
// - `ResponseStream.awaitCleanup()` resolving cleanly after
//   normal iteration completes.
//
// The session-layer tests in `IntegrationTests.swift` cover the
// multi-turn restart loop and tool dispatch — which are session-only
// concerns — so the two suites are complementary.

private let models = IntegrationTestModels()

private let parameters = GenerateParameters(maxTokens: 128, temperature: 0)

@Suite("Integration — low-level helpers", .serialized)
struct LowLevelHelpersIntegrationTests {
  @Test
  func `streamResponseEvents drives a manual ModelContext to a complete envelope`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let modelType = fixture.modelType
    let modelConfig = fixture.modelConfig

    try await fixture.container.perform { context in
      let userInput = UserInput(
        chat: [.user("Reply with the single word 'ok'.")],
        additionalContext: ["enable_thinking": false],
      )
      let input = try await context.processor.prepare(input: userInput)
      let cache = context.model.newCache(parameters: parameters)

      let handle = try streamResponseEvents(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        modelType: modelType,
        modelConfig: modelConfig,
        config: ResponseStreamConfig(model: context.configuration.name),
      )

      var events: [ResponseStreamingEvent] = []
      for await event in handle {
        events.append(event)
      }

      // Cleanup barrier resolves immediately on a clean finish
      // because the helper already awaited the producer task
      // before yielding the terminal event.
      await handle.awaitCleanup()

      // Lifecycle envelope: created → in_progress → … → completed.
      #expect(!events.isEmpty, "Expected non-empty event stream")
      guard case .responseCreated = events.first else {
        Issue.record("Expected responseCreated first; got \(String(describing: events.first))")
        return
      }
      guard case let .responseCompleted(completed) = events.last else {
        Issue.record("Expected responseCompleted last; got \(String(describing: events.last))")
        return
      }
      #expect(
        completed.response.status == .completed,
        "Expected status=.completed; got \(String(describing: completed.response.status))",
      )
      #expect(
        !completed.response.output.isEmpty,
        "Expected non-empty output[]; got \(completed.response.output)",
      )
    }
  }

  @Test
  func `streamResponseItems yields a monotonically growing snapshot`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let modelType = fixture.modelType
    let modelConfig = fixture.modelConfig

    try await fixture.container.perform { context in
      let userInput = UserInput(
        chat: [.user("Count from one to three.")],
        additionalContext: ["enable_thinking": false],
      )
      let input = try await context.processor.prepare(input: userInput)
      let cache = context.model.newCache(parameters: parameters)

      let handle = try streamResponseItems(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        modelType: modelType,
        modelConfig: modelConfig,
        config: ResponseStreamConfig(model: context.configuration.name),
      )

      var snapshots: [[ResponseOutputItem]] = []
      for await snapshot in handle {
        snapshots.append(snapshot)
      }
      await handle.awaitCleanup()

      // A live snapshot stream means the consumer can render the
      // current state at any point and only see additions over
      // time: the item count must not regress, and the first
      // delta in any item slot must be a prefix of subsequent
      // states. We assert the count invariant directly; deeper
      // delta-prefix invariants are pinned by the parser-library
      // unit tests.
      #expect(!snapshots.isEmpty, "Expected at least one snapshot yield")
      for i in 1 ..< snapshots.count {
        #expect(
          snapshots[i].count >= snapshots[i - 1].count,
          "Snapshot count regressed at index \(i): \(snapshots[i - 1].count) -> \(snapshots[i].count)",
        )
      }
      // Final snapshot has at least one item with content.
      guard let final = snapshots.last, !final.isEmpty else {
        Issue.record("Expected non-empty final snapshot; got \(snapshots.last ?? [])")
        return
      }
    }
  }

  @Test
  func `finalResponse() exposes the terminal Response to streamResponseItems consumers`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let modelType = fixture.modelType
    let modelConfig = fixture.modelConfig

    try await fixture.container.perform { context in
      let userInput = UserInput(
        chat: [.user("Reply with the single word 'ok'.")],
        additionalContext: ["enable_thinking": false],
      )
      let input = try await context.processor.prepare(input: userInput)
      let cache = context.model.newCache(parameters: parameters)

      let handle = try streamResponseItems(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        modelType: modelType,
        modelConfig: modelConfig,
        config: ResponseStreamConfig(model: context.configuration.name),
      )

      for await _ in handle {}

      // The items helper never yields the lifecycle envelope through
      // its snapshot stream, so without `finalResponse()` a consumer
      // has no way to read terminal usage / status. Pin both here.
      guard let response = await handle.finalResponse() else {
        Issue.record("finalResponse() returned nil after clean iteration")
        return
      }
      #expect(
        response.status == .completed,
        "Expected status=.completed; got \(String(describing: response.status))",
      )
      guard let usage = response.usage else {
        Issue.record("Expected non-nil usage on terminal Response")
        return
      }
      #expect(usage.inputTokens > 0, "Expected non-zero inputTokens; got \(usage.inputTokens)")
      #expect(usage.outputTokens > 0, "Expected non-zero outputTokens; got \(usage.outputTokens)")
    }
  }

  @Test
  func `ModelContainer overloads of streamResponseEvents / streamResponseItems work end-to-end`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let modelType = fixture.modelType
    let modelConfig = fixture.modelConfig
    let modelName = await fixture.container.configuration.name

    // Events overload — exercises the ModelContainer entry point and
    // verifies the same lifecycle envelope shape as the manual-context
    // tests above. `UserInput` is non-`Sendable`, so each `prepare`
    // call gets its own freshly-constructed value rather than reusing
    // one across the two `await` boundaries.
    let input = try await fixture.container.prepare(
      input: UserInput(
        chat: [.user("Reply with the single word 'ok'.")],
        additionalContext: ["enable_thinking": false],
      ),
    )

    let eventsHandle = try await fixture.container.streamResponseEvents(
      input: input,
      parameters: parameters,
      modelType: modelType,
      modelConfig: modelConfig,
      config: ResponseStreamConfig(model: modelName),
    )
    var events: [ResponseStreamingEvent] = []
    for await event in eventsHandle {
      events.append(event)
    }
    await eventsHandle.awaitCleanup()
    guard case .responseCompleted = events.last else {
      Issue.record("Expected responseCompleted last from container overload; got \(String(describing: events.last))")
      return
    }

    // Items overload — re-prepares input (the events call above
    // generated a turn from the same prompt, but no cache was reused
    // so this runs against a fresh KV cache too).
    let secondInput = try await fixture.container.prepare(
      input: UserInput(
        chat: [.user("Reply with the single word 'ok'.")],
        additionalContext: ["enable_thinking": false],
      ),
    )
    let itemsHandle = try await fixture.container.streamResponseItems(
      input: secondInput,
      parameters: parameters,
      modelType: modelType,
      modelConfig: modelConfig,
      config: ResponseStreamConfig(model: modelName),
    )
    var lastSnapshot: [ResponseOutputItem] = []
    for await snapshot in itemsHandle {
      lastSnapshot = snapshot
    }
    guard let response = await itemsHandle.finalResponse() else {
      Issue.record("finalResponse() returned nil from container-overload items helper")
      return
    }
    #expect(response.status == .completed)
    #expect(!lastSnapshot.isEmpty, "Expected non-empty final snapshot from container overload")
  }

  @Test
  func `Two consecutive helper calls with the same cache reuse KV state`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)
    let modelType = fixture.modelType
    let modelConfig = fixture.modelConfig

    try await fixture.container.perform { context in
      let cache = context.model.newCache(parameters: parameters)
      let config = ResponseStreamConfig(model: context.configuration.name)

      // Turn 1: state a fact.
      let firstInput = try await context.processor.prepare(
        input: UserInput(
          chat: [.user("My favorite number is 42. Reply with 'noted'.")],
          additionalContext: ["enable_thinking": false],
        ),
      )
      let firstHandle = try streamResponseEvents(
        input: firstInput,
        cache: cache,
        parameters: parameters,
        context: context,
        modelType: modelType,
        modelConfig: modelConfig,
        config: config,
      )
      for await _ in firstHandle {}
      await firstHandle.awaitCleanup()

      // Turn 2: ask about the fact. The cache from turn 1 carries
      // the assistant's "noted" reply, so for turn 2 we re-prepare
      // input from the full conversation. The KV cache holds the
      // tokenized prefix from turn 1, so the model should recall
      // the number.
      let secondInput = try await context.processor.prepare(
        input: UserInput(
          chat: [
            .user("My favorite number is 42. Reply with 'noted'."),
            .assistant("noted"),
            .user("What is my favorite number? Answer with just the digits."),
          ],
          additionalContext: ["enable_thinking": false],
        ),
      )
      let secondHandle = try streamResponseItems(
        input: secondInput,
        cache: cache,
        parameters: parameters,
        context: context,
        modelType: modelType,
        modelConfig: modelConfig,
        config: config,
      )
      var lastSnapshot: [ResponseOutputItem] = []
      for await snapshot in secondHandle {
        lastSnapshot = snapshot
      }
      await secondHandle.awaitCleanup()

      let combinedText = lastSnapshot.compactMap { item -> String? in
        guard case let .message(m) = item else { return nil }
        return m.content.compactMap {
          if case let .outputText(t) = $0 { return t.text }
          return nil
        }.joined()
      }.joined()

      #expect(
        combinedText.contains("42"),
        "Expected '42' in turn-2 response (cache reuse), got: \(combinedText)",
      )
    }
  }
}
