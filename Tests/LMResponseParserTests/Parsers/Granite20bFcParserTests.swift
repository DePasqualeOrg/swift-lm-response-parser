// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Granite-20B-FunctionCalling uses `<function_call>` (plain string) +
// JSON object, with no closing tag. Multiple calls are delimited by
// repeated `<function_call>` markers or end of stream. Fixtures mirror
// vLLM's `test_granite_20b_fc_tool_parser.py`.

@Suite("Granite20bFcParser — tool calls")
struct Granite20bFcToolCallTests {
  @Test
  func `Single call with string argument`() throws {
    var parser = Granite20bFcParser()
    let input = #"<function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }

  @Test
  func `Multiple parallel calls separated by newline`() {
    var parser = Granite20bFcParser()
    let input = """
    <function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    <function_call> {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}
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
  func `Various data types: string, int, float, bool, null, array, object, empty`() throws {
    var parser = Granite20bFcParser()
    let input = #"""
    <function_call> {
      "name": "test_function",
      "arguments": {
        "string_field": "hello",
        "int_field": 42,
        "float_field": 3.14,
        "bool_field": true,
        "null_field": null,
        "array_field": ["a", "b", "c"],
        "object_field": {"nested": "value"},
        "empty_array": [],
        "empty_object": {}
      }
    }
    """#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["string_field"] as? String == "hello")
    #expect(decoded["int_field"] as? Int == 42)
    #expect(decoded["float_field"] as? Double == 3.14)
    #expect(decoded["bool_field"] as? Bool == true)
    #expect(decoded["null_field"] is NSNull)
    let arr = try #require(decoded["array_field"] as? [String])
    #expect(arr == ["a", "b", "c"])
    let nested = try #require(decoded["object_field"] as? [String: String])
    #expect(nested["nested"] == "value")
  }

  @Test
  func `Empty arguments object`() {
    var parser = Granite20bFcParser()
    let input = #"<function_call> {"name": "refresh", "arguments": {}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "refresh")
    #expect(f.arguments == "{}")
  }

  @Test
  func `Surrounding text before the marker becomes a message`() {
    var parser = Granite20bFcParser()
    let input = """
    Let me check the weather for you.
    <function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(t.text == "Let me check the weather for you.\n")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Plain text without any marker passes through as a message`() {
    var parser = Granite20bFcParser()
    let events = parser.process(ParserInput(text: "This is a regular response.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "This is a regular response.")
  }

  @Test
  func `Bare JSON object without marker passes through as a message`() {
    var parser = Granite20bFcParser()
    let input = #"{"name": "func", "arguments": {}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == input)
  }

  @Test
  func `Escaped strings inside arguments`() throws {
    var parser = Granite20bFcParser()
    let input = #"""
    <function_call> {
      "name": "test_function",
      "arguments": {
        "quoted": "He said \"hello\"",
        "newline": "line1\nline2"
      }
    }
    """#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["quoted"] as? String == #"He said "hello""#)
    #expect(decoded["newline"] as? String == "line1\nline2")
  }

  @Test
  func `Truncated tool call closes as incomplete`() {
    var parser = Granite20bFcParser()
    let input = #"<function_call> {"name": "func", "arguments": {"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.status == .incomplete)
    #expect(f.name == "func")
  }

  @Test
  func `Marker followed by array is preserved as malformed content`() {
    var parser = Granite20bFcParser()
    let input = #"<function_call> [{"name": "func", "arguments": {}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected malformed content message"); return
    }
    #expect(t.text == input)
  }

  @Test
  func `Marker followed by non-string name is preserved as malformed content`() {
    var parser = Granite20bFcParser()
    let input = #"<function_call> {"name": 123}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected malformed content message"); return
    }
    #expect(t.text == input)
  }
}

@Suite("Granite20bFcParser — streaming")
struct Granite20bFcStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = """
    <function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    <function_call> {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}
    """

    var oneShot = Granite20bFcParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = Granite20bFcParser()
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
  func `Split <function_call> marker across chunks does not leak as content`() {
    var parser = Granite20bFcParser()
    var events = parser.process(ParserInput(text: "<function_"))
    events += parser.process(ParserInput(text: #"call> {"name": "fn", "arguments": {}}"#))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `JSON arguments split across chunks complete as a single call`() throws {
    var parser = Granite20bFcParser()
    var events = parser.process(ParserInput(text: #"<function_call> {"name": "get_weather", "arguments": {"#))
    events += parser.process(ParserInput(text: #""city": "Paris"}}"#))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
  }
}

@Suite("Granite20bFcParser — format dispatch")
struct Granite20bFcDispatchTests {
  @Test
  func `Name-prefix infer routes to .granite20bFc`() {
    let f = ResponseFormat.infer(
      modelName: "ibm-granite/granite-20b-functioncalling",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .granite20bFc)
  }
}
