// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import LMResponseParserMLX
import MLXLMCommon
import Testing

// Tool-dispatch integration tests: drive the multi-pass restart loop in
// `ResponseChatSession` end-to-end against a real MLX model. One test
// exercises the full round-trip (model → function_call → toolDispatch →
// function_call_output synthesized into the envelope → restart pass), the
// other pins the `toolDispatch == nil` branch (calls accumulate in
// `output[]` and the turn finalizes after one pass without auto-restart).
//
// We try Qwen3-0.6B because it ships the `.qwen` (Hermes-style) parser
// path and is already the workhorse for the rest of the integration
// suite. If a future regression makes Qwen3-0.6B unreliable here, the
// straightforward swap is `.llama3_2_1b` (already defined as a fixture)
// or a larger Qwen3 / Hermes variant.

private let models = IntegrationTestModels()

/// Hermes-style weather tool. Mirrors the canonical `WEATHER_TOOL` used
/// across vLLM's `tests/tool_use` suite, kept narrow on purpose: a single
/// required-ish parameter and a simple description so a small model has
/// the best chance of emitting a well-formed call.
private let weatherTool: ToolSpec = [
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

@Suite("Integration — tool dispatch", .serialized)
struct ToolDispatchIntegrationTests {
  @Test
  func `Tool round-trip: model emits function_call, dispatch fires, second pass uses the result`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)

    // Capture what the dispatcher saw, so the test can assert on the
    // call shape after the stream completes.
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
      additionalContext: ["enable_thinking": false],
      tools: [weatherTool],
      toolDispatch: { call in
        await recorder.record(call)
        return .string(cannedResult)
      },
    )

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "What is the weather in Dallas?"),
    )

    // Lifecycle: exactly one created and one completed across the
    // whole turn (regardless of pass count).
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

    // Reconstruct the final output[] from `output_item.done` events.
    // `responseCompleted.response.output` carries the same items, but
    // walking the events makes the assertion failure messages clearer.
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
      Expected the model to emit at least one function_call. Qwen3-0.6B \
      does not always tool-call reliably; if this assertion is the only \
      failure mode, swap the fixture for a larger model. Items: \(items)
      """,
    )

    let firstCall = functionCalls[0]
    #expect(firstCall.name == "get_current_weather", "Expected get_current_weather; got \(firstCall.name)")
    #expect(firstCall.status == .completed, "Expected function_call status=.completed; got \(firstCall.status)")

    // Arguments must parse as a JSON object containing `city`.
    let argsData = Data(firstCall.arguments.utf8)
    let parsedArgs = try JSONSerialization.jsonObject(with: argsData) as? [String: Any]
    try #require(parsedArgs != nil, "Function call arguments did not parse as JSON object: \(firstCall.arguments)")
    #expect(parsedArgs?["city"] is String, "Expected `city` string in arguments; got \(firstCall.arguments)")

    // Dispatch must have fired exactly once for that call.
    let recordedCalls = await recorder.calls
    #expect(recordedCalls.count == 1, "Expected exactly one dispatch invocation; got \(recordedCalls.count)")
    #expect(recordedCalls.first?.callId == firstCall.callId, "Dispatch fired with mismatched call_id")

    // Output must include the synthesized function_call_output paired
    // by call_id, and it must follow the function_call in the items
    // sequence (parser.emit pairing invariant).
    try #require(!functionOutputs.isEmpty, "Expected at least one function_call_output item; got \(items)")
    #expect(functionOutputs[0].callId == firstCall.callId, "function_call_output call_id must match function_call")
    #expect(functionOutputs[0].output == .string(cannedResult), "Tool output mismatch")

    if let callIdx = items.firstIndex(where: {
      if case let .functionCall(c) = $0, c.callId == firstCall.callId { return true }
      return false
    }), let outIdx = items.firstIndex(where: {
      if case let .functionCallOutput(o) = $0, o.callId == firstCall.callId { return true }
      return false
    }) {
      #expect(outIdx > callIdx, "function_call_output must come after its function_call; got call=\(callIdx) out=\(outIdx)")
    }
  }

  @Test
  func `toolDispatch == nil leaves function_call in output[] and finalizes the turn after one pass`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)

    let session = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: """
      You are a weather assistant. When the user asks about the \
      weather, call the get_current_weather tool with the city \
      they ask about.
      """,
      generateParameters: GenerateParameters(maxTokens: 256, temperature: 0),
      additionalContext: ["enable_thinking": false],
      tools: [weatherTool],
      // toolDispatch deliberately omitted.
    )

    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "What is the weather in Dallas?"),
    )

    guard case let .responseCompleted(completed) = events.last else {
      Issue.record("Expected responseCompleted as last event; got \(String(describing: events.last))")
      return
    }
    #expect(
      completed.response.status == .completed,
      "Expected status=.completed; got \(String(describing: completed.response.status))",
    )

    let functionCalls: [ResponseFunctionToolCall] = completed.response.output.compactMap {
      if case let .functionCall(c) = $0 { return c }
      return nil
    }
    let functionOutputs: [ResponseFunctionCallOutput] = completed.response.output.compactMap {
      if case let .functionCallOutput(o) = $0 { return o }
      return nil
    }

    try #require(
      !functionCalls.isEmpty,
      """
      Expected at least one function_call in output[]. Qwen3-0.6B \
      may not always tool-call; same fixture-swap caveat as the \
      round-trip test. Output: \(completed.response.output)
      """,
    )

    // No tool dispatcher means no synthesized function_call_output and
    // no second pass.
    #expect(functionOutputs.isEmpty, "Expected no function_call_output without toolDispatch; got \(functionOutputs)")
  }

  @Test
  func `toolDispatch throwing surfaces the error and leaves the session usable`() async throws {
    let fixture = try await models.fixture(.qwen3_0_6b)

    struct DispatchFailed: LocalizedError {
      var errorDescription: String? {
        "intentional dispatch failure"
      }
    }

    let session = ResponseChatSession(
      fixture.container,
      modelType: fixture.modelType,
      modelConfig: fixture.modelConfig,
      instructions: """
      You are a weather assistant. When the user asks about the \
      weather, call the get_current_weather tool with the city \
      they ask about.
      """,
      generateParameters: GenerateParameters(maxTokens: 256, temperature: 0),
      additionalContext: ["enable_thinking": false],
      tools: [weatherTool],
      toolDispatch: { _ in throw DispatchFailed() },
    )

    // First turn: dispatcher throws after the first pass emits a
    // function_call. The error must reach the consumer.
    var thrownError: Error?
    do {
      for try await _ in session.streamResponseEvents(prompt: "What is the weather in Dallas?") {
        // Drain — we only care about whether the stream throws.
      }
    } catch {
      thrownError = error
    }

    try #require(
      thrownError != nil,
      """
      Expected the stream to throw when toolDispatch fails. \
      (If Qwen3-0.6B didn't emit a tool call, dispatch wouldn't have \
      run at all and the stream would have completed cleanly — \
      same fixture-swap caveat as the round-trip test.)
      """,
    )
    #expect(thrownError is DispatchFailed, "Expected DispatchFailed; got \(String(describing: thrownError))")

    // Cache lock must release after the throw. If it leaked,
    // synchronize() (or the second turn below) would deadlock and
    // the test runner would time out.
    await session.synchronize()

    // Second turn: session must still be usable. Swap to a
    // dispatcher that succeeds so we can drive a clean turn.
    session.toolDispatch = { _ in
      .string(#"{"city":"Dallas","temp_f":98}"#)
    }
    let events = try await collectEvents(
      session.streamResponseEvents(prompt: "What is the weather in Dallas?"),
    )
    guard case .responseCompleted = events.last else {
      Issue.record("Session unusable after dispatcher-throw: expected responseCompleted as last event; got \(String(describing: events.last))")
      return
    }
  }
}
