// Copyright © Anthony DePasquale

import Foundation
import LMResponses
import LMResponsesMLX
import MLXLMCommon
import Testing

// End-to-end integration tests for the Cohere parser path. Mirrors the
// Qwen3 suite but exercises the `.cohereCmd3` dispatch route: `model_type:
// "cohere2"` from a Command R7B `config.json` resolves to
// `.cohereCmd3`, which constructs `CohereParser(variant: .cmd3)`.
//
// **Running these tests.** Same gating as the rest of the integration
// suite: requires `LMRESPONSES_INTEGRATION_TESTS=1` and is intended to be
// driven from Xcode so MLX's `default.metallib` is bundled. First-run
// cost: ~4.5 GB download of `mlx-community/c4ai-command-r7b-12-2024-4bit`
// to the standard Hugging Face cache.
//
// **cmd4 path not covered.** The cmd4 marker set (`<|START_TEXT|>` /
// `<|END_TEXT|>`) and reasoning-default initial state are unit-tested but
// not exercised end-to-end here. The only published `cohere2_vision`
// checkpoint (`CohereLabs/command-a-vision-07-2025`) is 224 GB and
// gated; `cohere2_moe` has no published checkpoint at all. See the
// `cohereR7B` fixture's TODO for the verification steps to run when a
// smaller cmd4 model becomes available.

private let models = IntegrationTestModels()

@Suite("Integration — Cohere Command R7B", .serialized)
struct CohereIntegrationTests {
  // MARK: Format dispatch

  @Test
  func `model_type cohere2 routes to cohereCmd3`() async throws {
    let fixture = try await models.fixture(.cohereR7B)
    // Verifies the same inference our `ResponseChatSession` runs
    // internally when no explicit `format` override is supplied. If
    // upstream renames the `model_type` string in `config.json` (or
    // ships a new arch), this assertion catches it before the
    // generation tests below fail in a less obvious way.
    let resolved = ResponseFormat.infer(
      modelName: "",
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
    )
    #expect(
      resolved == .cohereCmd3,
      "Expected .cohereCmd3 for model_type=\(fixture.modelType); got \(String(describing: resolved))",
    )
  }

  // MARK: Lifecycle envelope shape

  @Test
  func `streamResponseEvents emits exactly one response.created and one response.completed`() async throws {
    let fixture = try await models.fixture(.cohereR7B)
    let session = makeSession(fixture, instructions: "Be brief.")

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
  func `sequence_number is strictly monotonic across the whole turn`() async throws {
    let fixture = try await models.fixture(.cohereR7B)
    let session = makeSession(fixture, instructions: "Be brief.")

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "Reply with 'ok'."),
    )

    let sequences = events.map(\.sequenceNumber)
    for (i, n) in sequences.enumerated() where i > 0 {
      #expect(n == sequences[i - 1] + 1, "sequence #\(i) (\(n)) does not follow \(sequences[i - 1])")
    }
  }

  // MARK: Content shape — cmd3 markers

  @Test
  func `Cohere wire format produces a message item with non-empty text`() async throws {
    let fixture = try await models.fixture(.cohereR7B)
    let session = makeSession(fixture, instructions: "Be brief.")

    let (items, text) = try await collectMessageText(
      session.streamResponseEvents(prompt: "Reply with the single word 'ok'."),
    )

    let messageItems = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m }
      return nil
    }
    #expect(!messageItems.isEmpty, "Expected at least one message item, got items: \(items)")
    #expect(!text.isEmpty, "Expected non-empty assistant text, got items: \(items)")
    // Marker tokens must not leak into the message text — they're
    // structural delimiters consumed by the parser.
    #expect(
      !text.contains("<|START_RESPONSE|>") && !text.contains("<|END_RESPONSE|>"),
      "Cohere marker tokens leaked into message text: \(text)",
    )
    #expect(
      !text.contains("<|START_THINKING|>") && !text.contains("<|END_THINKING|>"),
      "Cohere reasoning markers leaked into message text: \(text)",
    )
  }

  @Test
  func `Reasoning region routes to a reasoning item, not a message`() async throws {
    let fixture = try await models.fixture(.cohereR7B)
    // Command R7B emits `<|START_THINKING|>…<|END_THINKING|>` when the
    // instruction nudges it toward step-by-step reasoning. The parser
    // must route that block to a `reasoning` item and only then open a
    // `message`. Token budget is generous because reasoning + response
    // can be long.
    let session = makeSession(
      fixture,
      instructions: "Think step by step inside <|START_THINKING|> markers, then answer.",
      maxTokens: 800,
    )

    let (items, _) = try await collectMessageText(
      session.streamResponseEvents(prompt: "What is 7 times 8? Show your work."),
    )

    let kinds = items.map { item in
      switch item {
        case .message: "message"
        case .functionCall: "functionCall"
        case .reasoning: "reasoning"
        case .functionCallOutput: "functionCallOutput"
      }
    }
    // The model is small and sometimes skips the reasoning block.
    // When it does emit one, it must come before any message item —
    // that's the cmd3 wire-format contract.
    if let firstReasoning = kinds.firstIndex(of: "reasoning"),
       let firstMessage = kinds.firstIndex(of: "message")
    {
      #expect(
        firstReasoning < firstMessage,
        "Reasoning should precede message; got \(kinds)",
      )
    }
  }

  // MARK: Final response shape

  @Test
  func `Clean-stop turn carries status=completed and a non-empty output[]`() async throws {
    let fixture = try await models.fixture(.cohereR7B)
    let session = makeSession(fixture, instructions: "Be brief.")

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "Reply with one word."),
    )
    guard case let .responseCompleted(completed) = events.last else {
      Issue.record("Expected responseCompleted as last event"); return
    }
    #expect(
      completed.response.status == .completed,
      "Expected status=.completed; got \(String(describing: completed.response.status))",
    )
    #expect(!completed.response.output.isEmpty, "Expected non-empty output[]; got \(completed.response.output)")
  }

  // MARK: Helpers

  private func makeSession(
    _ fixture: LoadedFixture,
    instructions: String? = nil,
    maxTokens: Int = 256,
  ) -> ResponseChatSession {
    // Cohere2 chat templates don't expose an `enable_thinking`
    // toggle the way Qwen3 does; reasoning is steered by prompt
    // content instead, so we leave `additionalContext` empty.
    ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: instructions,
      generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
    )
  }
}

@Suite("Integration — Cohere tool dispatch", .serialized)
struct CohereToolDispatchIntegrationTests {
  /// Hermes-style weather tool, same shape as the Qwen3 suite's. The
  /// chat template for Command R7B reformats `tools` into Cohere's
  /// own JSON-schema-with-tool_call_id wire shape on the prompt side;
  /// the parser side just needs to decode the resulting
  /// `<|START_ACTION|>[…]<|END_ACTION|>` JSON array.
  private static let weatherTool: ToolSpec = [
    "type": "function",
    "function": [
      "name": "get_current_weather",
      "description": "Get the current weather in a given location.",
      "parameters": [
        "type": "object",
        "properties": [
          "city": [
            "type": "string",
            "description": "The city to find the weather for, e.g. 'San Francisco'.",
          ] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["city"] as [String],
      ] as [String: any Sendable],
    ] as [String: any Sendable],
  ]

  @Test
  func `Tool round-trip: model emits function_call, dispatch fires, second pass uses the result`() async throws {
    let fixture = try await models.fixture(.cohereR7B)

    actor DispatchRecorder {
      var calls: [(name: String, callId: String, arguments: String)] = []
      func record(_ call: ResponseFunctionToolCall) {
        calls.append((name: call.name, callId: call.callId, arguments: call.arguments))
      }
    }
    let recorder = DispatchRecorder()
    let cannedResult = #"{"city":"Dallas","temp_f":98,"conditions":"sunny"}"#

    let session = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: """
      You are a weather assistant. When the user asks about the \
      weather, call the get_current_weather tool with the city \
      they ask about. After the tool returns, give a one-sentence \
      summary using the returned values.
      """,
      generateParameters: GenerateParameters(maxTokens: 512, temperature: 0),
      tools: [Self.weatherTool],
      toolDispatch: { call in
        await recorder.record(call)
        return .string(cannedResult)
      },
    )

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "What is the weather in Dallas?"),
    )

    let createds = events.count(where: {
      if case .responseCreated = $0 { return true }
      return false
    })
    let completeds = events.count(where: {
      if case .responseCompleted = $0 { return true }
      return false
    })
    #expect(createds == 1, "Expected one responseCreated; got \(createds)")
    #expect(completeds == 1, "Expected one responseCompleted; got \(completeds)")

    var accumulator = ResponseItemsAccumulator()
    for event in events {
      accumulator.ingest(event)
    }
    let items = accumulator.items

    let functionCalls: [ResponseFunctionToolCall] = items.compactMap {
      if case let .functionCall(c) = $0 { return c }
      return nil
    }
    let functionOutputs: [ResponseFunctionCallOutput] = items.compactMap {
      if case let .functionCallOutput(o) = $0 { return o }
      return nil
    }

    try #require(
      !functionCalls.isEmpty,
      """
      Expected the model to emit at least one function_call. \
      Command R7B is small and may not always tool-call reliably; \
      if this assertion is the only failure mode, swap the fixture \
      for a larger Cohere2 checkpoint. Items: \(items)
      """,
    )

    let firstCall = functionCalls[0]
    #expect(firstCall.name == "get_current_weather", "Expected get_current_weather; got \(firstCall.name)")
    #expect(firstCall.status == .completed, "Expected function_call status=.completed; got \(firstCall.status)")

    let argsData = Data(firstCall.arguments.utf8)
    let parsedArgs = try JSONSerialization.jsonObject(with: argsData) as? [String: Any]
    try #require(parsedArgs != nil, "Function call arguments did not parse as JSON object: \(firstCall.arguments)")
    #expect(parsedArgs?["city"] is String, "Expected `city` string in arguments; got \(firstCall.arguments)")

    let recordedCalls = await recorder.calls
    #expect(recordedCalls.count == 1, "Expected exactly one dispatch invocation; got \(recordedCalls.count)")
    #expect(recordedCalls.first?.callId == firstCall.callId, "Dispatch fired with mismatched call_id")

    try #require(!functionOutputs.isEmpty, "Expected at least one function_call_output item; got \(items)")
    #expect(functionOutputs[0].callId == firstCall.callId, "function_call_output call_id must match function_call")
    #expect(functionOutputs[0].output == .string(cannedResult), "Tool output mismatch")
  }
}
