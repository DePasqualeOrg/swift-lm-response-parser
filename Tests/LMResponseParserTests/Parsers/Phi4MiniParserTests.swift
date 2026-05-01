// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Fixtures lifted from vLLM's `tests/tool_parsers/test_phi4mini_tool_parser.py`,
// adapted to the streaming event shape this package emits. vLLM's reference
// parser is non-streaming; the streaming-reconstruction tests below verify
// our incremental implementation produces the same items as one-shot
// delivery, which vLLM's parser cannot do.

@Suite("Phi4MiniParser — plain text")
struct Phi4MiniPlainTextTests {
  @Test
  func `Plain text without tool calls emits a single message`() {
    var parser = Phi4MiniParser()
    let events = parser.process(ParserInput(text: "This is a regular response without any tool calls."))
      + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "This is a regular response without any tool calls.")
  }

  @Test
  func `Empty stream emits nothing`() {
    var parser = Phi4MiniParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("Phi4MiniParser — single tool call")
struct Phi4MiniSingleToolCallTests {
  @Test
  func `Single call with string argument`() throws {
    var parser = Phi4MiniParser()
    let input = #"functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#
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
  func `Empty arguments object`() throws {
    var parser = Phi4MiniParser()
    let input = #"functools[{"name": "refresh", "arguments": {}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "refresh")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded.isEmpty)
  }

  @Test
  func `Mixed argument types: string, int, float, bool, null, array, object`() throws {
    var parser = Phi4MiniParser()
    let input = """
    functools[{
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
    }]
    """
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
    #expect(decoded["array_field"] as? [String] == ["a", "b", "c"])
    let nested = decoded["object_field"] as? [String: String]
    #expect(nested?["nested"] == "value")
    #expect((decoded["empty_array"] as? [Any])?.isEmpty == true)
    #expect((decoded["empty_object"] as? [String: Any])?.isEmpty == true)
  }

  @Test
  func `Parameters fallback when arguments field is absent`() throws {
    // Some chat templates emit `parameters` instead of `arguments`.
    // vLLM's reference parser falls back to `parameters` and we match.
    var parser = Phi4MiniParser()
    let input = #"functools[{"name": "fn", "parameters": {"x": 1}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Tool call IDs follow fc_/call_ prefix convention`() {
    var parser = Phi4MiniParser()
    let events = parser.process(ParserInput(text: #"functools[{"name": "fn", "arguments": {}}]"#))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
  }
}

@Suite("Phi4MiniParser — multiple tool calls")
struct Phi4MiniMultipleToolCallTests {
  @Test
  func `Parallel calls in one functools envelope`() throws {
    var parser = Phi4MiniParser()
    let input = """
    functools[
      {"name": "get_weather", "arguments": {"city": "Tokyo"}},
      {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}
    ]
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "get_weather")
    #expect(toolCalls[1].name == "get_time")
    let argsBData = try #require(toolCalls[1].arguments.data(using: .utf8))
    let argsB = try #require(JSONSerialization.jsonObject(with: argsBData) as? [String: String])
    #expect(argsB["timezone"] == "Asia/Tokyo")
  }
}

@Suite("Phi4MiniParser — surrounding text")
struct Phi4MiniSurroundingTextTests {
  @Test
  func `Leading text before envelope emits as message`() {
    // Departure from vLLM's reference: vLLM strips surrounding content
    // when tool calls are present. We surface it instead — consistent
    // with how the rest of this package's parsers handle preamble.
    var parser = Phi4MiniParser()
    let input = #"Let me check the weather. functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(t.text == "Let me check the weather. ")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Trailing text after envelope emits as message`() {
    var parser = Phi4MiniParser()
    let input = """
    functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    Would you like to know more?
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .functionCall = items[0] else { Issue.record("Expected function call"); return }
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected trailing message"); return
    }
    #expect(t.text.contains("Would you like to know more?"))
  }
}

@Suite("Phi4MiniParser — escaped strings")
struct Phi4MiniEscapedStringTests {
  @Test
  func `Escaped quotes, backslashes, newlines, unicode`() throws {
    var parser = Phi4MiniParser()
    let input = #"functools[{"name": "test_function", "arguments": {"quoted": "He said \"hello\"", "path": "C:\\Users\\file.txt", "newline": "line1\nline2", "unicode": "emoji: 🎉"}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: String])
    #expect(decoded["quoted"] == #"He said "hello""#)
    #expect(decoded["path"] == #"C:\Users\file.txt"#)
    #expect(decoded["newline"] == "line1\nline2")
    #expect(decoded["unicode"] == "emoji: 🎉")
  }
}

@Suite("Phi4MiniParser — streaming")
struct Phi4MiniStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = #"functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#

    var oneShot = Phi4MiniParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = Phi4MiniParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    guard case let .functionCall(a) = oneShotItems[0],
          case let .functionCall(b) = streamedItems[0]
    else {
      Issue.record("Expected function calls"); return
    }
    #expect(a.name == b.name)
    #expect(a.arguments == b.arguments)
  }

  @Test
  func `Split functools opener across chunks doesn't leak`() {
    // Hold-back must keep the partial `func` from leaking into a
    // message item so that when `tools[` arrives the parser still
    // recognizes the envelope.
    var parser = Phi4MiniParser()
    var events = parser.process(ParserInput(text: "func"))
    events += parser.process(ParserInput(text: #"tools[{"name":"fn","arguments":{}}]"#))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "fn")
    // No spurious message containing "func".
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] {
        return t.text
      }
      return nil
    }
    #expect(messages.allSatisfy { !$0.contains("func") })
  }
}

@Suite("Phi4MiniParser — malformed input")
struct Phi4MiniMalformedInputTests {
  @Test
  func `Truncated mid-call (unclosed brace) surfaces as content at finalize`() {
    var parser = Phi4MiniParser()
    let input = #"functools[{"name": "func", "arguments": {"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // No tool call; the unbalanced envelope is forwarded as content
    // (graceful degradation, no crash).
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Arguments value is a string, not a dict — emits the call with the string as-is`() {
    // vLLM's xfail case: the JSON parses, and we treat the args field
    // as already-encoded JSON. The call surfaces.
    var parser = Phi4MiniParser()
    let input = #"functools[{"name": "func", "arguments": "not a dict"}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "func")
    #expect(toolCalls[0].arguments == "not a dict")
  }

  @Test
  func `Missing brackets — functools{...} — emits as content`() {
    var parser = Phi4MiniParser()
    let input = #"functools{"name": "func"}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Empty functools array followed by trailing content`() {
    var parser = Phi4MiniParser()
    let input = "functools[] This is just text"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Invalid JSON inside functools envelope emits as content`() {
    var parser = Phi4MiniParser()
    let input = "functools[ This is just text ]"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}
