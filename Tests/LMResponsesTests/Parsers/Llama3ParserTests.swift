// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

@Suite("Llama3Parser — plain text")
struct Llama3PlainTextTests {
  @Test
  func `Plain text without tool calls emits a single message`() {
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "hello world")
  }

  @Test
  func `Empty stream emits nothing`() {
    var parser = Llama3Parser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("Llama3Parser — single tool call")
struct Llama3SingleToolCallTests {
  @Test
  func `Single call with python_tag prefix`() throws {
    var parser = Llama3Parser()
    let input = #"<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}"#
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
  func `Bare JSON without python_tag prefix is recognized as a tool call`() {
    var parser = Llama3Parser()
    let input = #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Tool call IDs are distinct fc_/call_ pairs`() {
    var parser = Llama3Parser()
    let input = #"<|python_tag|>{"name": "fn", "arguments": {}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
    #expect(f.id != f.callId)
  }
}

@Suite("Llama3Parser — multiple tool calls")
struct Llama3MultipleToolCallTests {
  @Test
  func `Parallel calls separated by ; emit multiple function_call items`() {
    var parser = Llama3Parser()
    let input = #"<|python_tag|>{"name": "f1", "arguments": {}}; {"name": "f2", "arguments": {"x": 1}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "f1")
    #expect(b.name == "f2")
  }
}

@Suite("Llama3Parser — surrounding text")
struct Llama3SurroundingTextTests {
  @Test
  func `Text before python_tag is emitted as message`() {
    var parser = Llama3Parser()
    let input = #"Let me check. <|python_tag|>{"name": "fn", "arguments": {}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }
}

@Suite("Llama3Parser — streaming")
struct Llama3StreamingTests {
  @Test
  func `Char-by-char streaming yields the same items as single-shot`() {
    let input = #"<|python_tag|>{"name": "fn", "arguments": {"x": 1}}"#

    var streaming = Llama3Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = Llama3Parser()
    let oneShotItems = accumulateItems(from:
      oneShot.process(ParserInput(text: input)) + oneShot.finalize())

    #expect(streamingItems.count == oneShotItems.count)
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.functionCall(sf), .functionCall(of)):
          #expect(sf.name == of.name)
          #expect(sf.arguments == of.arguments)
        case let (.message(sm), .message(om)):
          #expect(sm.content == om.content)
        default:
          Issue.record("Item kinds differ"); return
      }
    }
  }

  @Test
  func `Fixed chunks preserve exact text and argument deltas`() {
    let chunks = [
      "prefix ",
      "<|python",
      #"_tag|>{"name":"fn","arguments":{"x":"#,
      "1",
      "}}",
      "suffix",
    ]

    var parser = Llama3Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let outputIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(outputIndexes == [0, 1, 2])
    #expect(llama3OutputTextDeltas(from: events) == ["prefix ", "suffix"])
    #expect(llama3ArgumentDeltas(from: events) == [#"{"x":1}"#])
  }

  @Test
  func `Completed JSON object followed by later JSON object preserves exact deltas`() {
    let chunks = [
      #"<|python_tag|>{"name":"first","arguments":{}}; "#,
      #"{"name":"second","arguments":{"y":2}}"#,
    ]

    var parser = Llama3Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let outputIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(outputIndexes == [0, 1])
    #expect(llama3ArgumentDeltas(from: events) == ["{}", #"{"y":2}"#])
  }
}

private func llama3OutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .outputTextDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

private func llama3ArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .functionCallArgumentsDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

@Suite("Llama3Parser — finalize edge cases")
struct Llama3FinalizeTests {
  @Test
  func `Truncated tool call mid-args is preserved as content`() {
    var parser = Llama3Parser()
    let input = #"<|python_tag|>{"name": "fn", "arguments": {"x": "incomp"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == #"{"name": "fn", "arguments": {"x": "incomp"#)
  }
}

@Suite("Llama3Parser — dispatch")
struct Llama3DispatchTests {
  @Test
  func `ResponseFormat.llama3 dispatches to Llama3Parser`() {
    let parser = ResponseFormat.llama3.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Llama 2 (vocab_size 32000) does NOT route to llama3`() {
    let format = ResponseFormat.resolveByType("llama", config: ["vocab_size": 32000])
    #expect(format == nil)
  }

  @Test
  func `Llama 3 (vocab_size 128256) routes to .llama3`() {
    let format = ResponseFormat.resolveByType("llama", config: ["vocab_size": 128_256])
    #expect(format == .llama3)
  }
}

@Suite("Llama3Parser — adversarial ports")
struct Llama3AdversarialTests {
  // H10: vLLM test_extract_tool_calls_multiple_json and variants
  // (test_llama3_json_tool_parser.py:87-143).
  @Test
  func `Three bare JSON objects separated by ;  extract as three calls`() throws {
    let input = (
      #"{"name": "searchTool", "parameters": {"query": "test1"}}; "#
        + #"{"name": "getOpenIncidentsTool", "parameters": {}}; "#
        + #"{"name": "searchTool", "parameters": {"query": "test2"}}"#,
    )
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 3)
    #expect(toolCalls[0].name == "searchTool")
    let args0Data = try #require(toolCalls[0].arguments.data(using: .utf8))
    let args0 = try #require(JSONSerialization.jsonObject(with: args0Data) as? [String: Any])
    #expect(args0["query"] as? String == "test1")
    #expect(toolCalls[1].name == "getOpenIncidentsTool")
    #expect(toolCalls[1].arguments == "{}")
    #expect(toolCalls[2].name == "searchTool")
    let args2Data = try #require(toolCalls[2].arguments.data(using: .utf8))
    let args2 = try #require(JSONSerialization.jsonObject(with: args2Data) as? [String: Any])
    #expect(args2["query"] as? String == "test2")
  }

  @Test
  func `Three JSONs separated by  ;  (extra whitespace) extract as three calls`() {
    let input = (
      #"{"name": "searchTool", "parameters": {"query": "test1"}} ; "#
        + #"{"name": "getOpenIncidentsTool", "parameters": {}} ; "#
        + #"{"name": "searchTool", "parameters": {"query": "test2"}}"#,
    )
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 3)
  }

  // SGLang test_multiple_json_with_separator_customized: repeated
  // `<|python_tag|>` markers may separate adjacent JSON objects.
  @Test
  func `Repeated python_tag separators extract adjacent calls`() {
    let input = (
      #"<|python_tag|>{"name": "get_weather", "parameters": {}}"#
        + #"<|python_tag|>{"name": "get_tourist_attractions", "parameters": {}}"#,
    )
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "get_weather")
    #expect(toolCalls[1].name == "get_tourist_attractions")
  }

  @Test
  func `Trailing text after JSON call is preserved as message content`() {
    let input = #"{"name": "get_weather", "parameters": {}} Some follow-up text"#
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(f) = items[0] else {
      Issue.record("Expected function call"); return
    }
    #expect(f.name == "get_weather")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected trailing message"); return
    }
    #expect(t.text.contains("follow-up text"))
  }

  @Test
  func `JSON object without name is preserved as content`() {
    let input = #"{"parameters": {}}"#
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == input)
  }

  // SGLang test_multiple_json_objects (test_llama32_detector.py:96-101) —
  // boundary case: `};{` with no space.
  @Test
  func `Two JSONs separated by };{ (no space) extract as two calls`() {
    let input = (
      #"<|python_tag|>{"name": "get_weather", "arguments": {"city": "Beijing"}}"#
        + #";{"name": "search", "arguments": {"query": "restaurants"}}"#,
    )
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "get_weather")
    #expect(toolCalls[1].name == "search")
  }
}

@Suite("Llama3Parser — Python-dict fallback")
struct Llama3PythonDictTests {
  @Test
  func `Single-quoted Python-dict-style call is parsed as a tool call`() throws {
    let input = "<|python_tag|>{'name': 'fn', 'arguments': {'x': 1, 'y': 'hello'}}"
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call"); return
    }
    #expect(f.name == "fn")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
    #expect(decoded["y"] as? String == "hello")
  }

  @Test
  func `Python True/False/None coerce to JSON true/false/null`() throws {
    let input = "<|python_tag|>{'name': 'fn', 'arguments': {'a': True, 'b': False, 'c': None}}"
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call"); return
    }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["a"] as? Bool == true)
    #expect(decoded["b"] as? Bool == false)
    #expect(decoded["c"] is NSNull)
  }
}

@Suite("Llama3Parser — recovery from malformed JSON")
struct Llama3RecoveryTests {
  @Test
  func `Malformed call followed by a valid call still emits the valid call`() {
    // First object has an unterminated string; recovery should skip it
    // and find the second `{"name":` object.
    let input = (
      #"<|python_tag|>{"name": "broken", "arguments": {"x": "unclosed}; "#
        + #"{"name": "good", "arguments": {"y": 2}}"#,
    )
    var parser = Llama3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls.first?.name == "good")
  }
}
