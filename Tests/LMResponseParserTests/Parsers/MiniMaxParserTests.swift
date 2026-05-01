// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// MiniMax-Text-01 / M1 (the non-M2 line) wraps tool calls in
// `<tool_calls>...</tool_calls>` containing NDJSON – one JSON object per
// line. Distinct from `.miniMaxM2` (XML invoke/parameter shape) and
// from `.jamba` (which wraps a JSON array). Tool calls inside
// `<think>...</think>` are filtered out so reasoning-block "examples"
// don't surface as real tool calls.

@Suite("MiniMaxParser — envelope and NDJSON")
struct MiniMaxEnvelopeTests {
  @Test
  func `Single call inside <tool_calls>...</tool_calls>`() throws {
    var parser = MiniMaxParser()
    let input = """
    <tool_calls>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
  }

  @Test
  func `Multiple parallel calls one per line`() {
    var parser = MiniMaxParser()
    let input = """
    <tool_calls>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    {"name": "get_time", "arguments": {"timezone": "UTC"}}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "get_weather")
    #expect(b.name == "get_time")
  }

  @Test
  func `JSON array inside envelope is NOT recognized (NDJSON only)`() {
    // vLLM's parser splits by newline and parses each line as a single
    // object. A JSON array `[{...}, {...}]` doesn't pass the
    // hasPrefix("{") gate, so the envelope falls back to message text.
    var parser = MiniMaxParser()
    let input = #"<tool_calls>[{"name": "f", "arguments": {}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Plain text without envelope passes through`() {
    var parser = MiniMaxParser()
    let events = parser.process(ParserInput(text: "How can I help you today?")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "How can I help you today?")
  }

  @Test
  func `Leading text before envelope emits as message then tool call`() {
    var parser = MiniMaxParser()
    let input = """
    Sure, calling now: <tool_calls>
    {"name": "f", "arguments": {}}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(t.text == "Sure, calling now: ")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }

  @Test
  func `Unclosed envelope at end of stream still emits complete NDJSON calls`() {
    var parser = MiniMaxParser()
    let events = parser.process(ParserInput(text: #"<tool_calls>{"name": "f", "arguments": {}}"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
    #expect(f.arguments == "{}")
  }

  @Test
  func `Malformed truncated envelope at end of stream falls back to message`() {
    var parser = MiniMaxParser()
    let events = parser.process(ParserInput(text: #"<tool_calls>{"name": "f", "arguments": {"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message fallback"); return
    }
    #expect(t.text == #"<tool_calls>{"name": "f", "arguments": {"#)
  }

  @Test
  func `Argument key alias parameters is accepted`() throws {
    var parser = MiniMaxParser()
    let input = """
    <tool_calls>
    {"name": "f", "parameters": {"x": 1}}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Non-object argument values serialize as JSON fragments`() {
    // vLLM applies json.dumps to the raw `arguments` JSON value, not
    // only dictionaries. Arrays and strings therefore stay visible
    // instead of being rewritten to `{}`.
    var parser = MiniMaxParser()
    let input = """
    <tool_calls>
    {"name": "array_args", "arguments": [1, 2, 3]}
    {"name": "string_args", "arguments": "raw"}
    {"name": "bool_args", "arguments": true}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let calls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f }
      return nil
    }
    #expect(calls.count == 3)
    #expect(calls[0].arguments == "[1,2,3]")
    #expect(calls[1].arguments == #""raw""#)
    #expect(calls[2].arguments == "true")
  }

  @Test
  func `Lines missing name or arguments are ignored`() {
    var parser = MiniMaxParser()
    let input = """
    <tool_calls>
    {"name": "valid", "arguments": {"city": "Seattle"}}
    {"name": "missing_args"}
    {"arguments": {"city": "Portland"}}
    {"name": "also_valid", "parameters": {"x": 1}}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let calls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f }
      return nil
    }
    #expect(calls.map(\.name) == ["valid", "also_valid"])
  }
}

@Suite("MiniMaxParser — think filtering")
struct MiniMaxThinkFilteringTests {
  @Test
  func `Tool call envelope inside <think>...</think> is treated as content`() {
    // Mirrors vLLM's `preprocess_model_output`: tool calls inside the
    // reasoning block are filtered out before extraction. They surface
    // as part of the message text instead.
    var parser = MiniMaxParser()
    let input = """
    <think>
    Let me try this:
    <tool_calls>
    {"name": "fake", "arguments": {}}
    </tool_calls>
    Actually that's wrong.
    </think>
    <tool_calls>
    {"name": "real", "arguments": {"x": 1}}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let functionCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(functionCalls.count == 1)
    #expect(functionCalls.first?.name == "real")
  }

  @Test
  func `Tool call after a closed <think> block is extracted normally`() {
    var parser = MiniMaxParser()
    let input = """
    <think>plan</think>
    <tool_calls>
    {"name": "f", "arguments": {}}
    </tool_calls>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let functionCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(functionCalls.count == 1)
    #expect(functionCalls.first?.name == "f")
  }
}

@Suite("MiniMaxParser — streaming")
struct MiniMaxStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = """
    <tool_calls>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    {"name": "get_time", "arguments": {"timezone": "UTC"}}
    </tool_calls>
    """

    var oneShot = MiniMaxParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = MiniMaxParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    #expect(oneShotItems.count == 2)
    for (a, b) in zip(oneShotItems, streamedItems) {
      guard case let .functionCall(fa) = a, case let .functionCall(fb) = b else {
        Issue.record("Expected function calls"); continue
      }
      #expect(fa.name == fb.name)
      #expect(fa.arguments == fb.arguments)
    }
  }

  @Test
  func `Split <tool_calls> opener across chunks does not leak`() {
    var parser = MiniMaxParser()
    var events = parser.process(ParserInput(text: "<tool"))
    events += parser.process(ParserInput(
      text: """
      _calls>
      {"name": "f", "arguments": {}}
      </tool_calls>
      """,
    ))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }

  @Test
  func `Split </tool_calls> closer across chunks does not leak`() {
    var parser = MiniMaxParser()
    var events = parser.process(ParserInput(text: """
    <tool_calls>
    {"name": "f", "arguments": {}}
    </tool
    """))
    events += parser.process(ParserInput(text: "_calls>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }
}

@Suite("MiniMaxParser — format dispatch")
struct MiniMaxDispatchTests {
  @Test
  func `Name-prefix infer routes M1 to .miniMax`() {
    let f = ResponseFormat.infer(
      modelName: "MiniMaxAI/MiniMax-M1-40k",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .miniMax)
  }

  @Test
  func `Name-prefix infer routes Text-01 to .miniMax`() {
    let f = ResponseFormat.infer(
      modelName: "MiniMaxAI/MiniMax-Text-01",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .miniMax)
  }

  @Test
  func `Name-prefix infer still routes M2 to .miniMaxM2 (longest-prefix)`() {
    let f = ResponseFormat.infer(
      modelName: "MiniMaxAI/MiniMax-M2",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .miniMaxM2)
  }
}
