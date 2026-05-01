// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Fixtures are lifted from vLLM's `tests/tool_parsers/test_granite_tool_parser.py`,
// adapted to the streaming event shape this package emits. vLLM's reference
// parser is non-streaming for the array body; the streaming-reconstruction
// tests below verify our incremental implementation produces the same items
// as one-shot delivery.

@Suite("GraniteParser — plain text")
struct GranitePlainTextTests {
  @Test
  func `Plain text without tool calls emits a single message`() {
    var parser = GraniteParser()
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
    var parser = GraniteParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("GraniteParser — single tool call")
struct GraniteSingleToolCallTests {
  @Test
  func `Granite 3.0 token marker — <|tool_call|>`() throws {
    var parser = GraniteParser()
    let input = #"<|tool_call|> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#
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
  func `Granite 3.1 string marker — <tool_call>`() throws {
    var parser = GraniteParser()
    let input = #"<tool_call> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }

  @Test
  func `Empty arguments object`() throws {
    var parser = GraniteParser()
    let input = #"<|tool_call|> [{"name": "refresh", "arguments": {}}]"#
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
    var parser = GraniteParser()
    let input = """
    <tool_call> [{
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
    // sglang's shared `parse_base_json` accepts both keys; vLLM's
    // Granite parser only accepts `arguments`. We follow sglang's
    // lenient handling, matching `Llama3Parser`, `Phi4MiniParser`, and
    // `MistralParser`.
    var parser = GraniteParser()
    let input = #"<|tool_call|> [{"name": "fn", "parameters": {"x": 1}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Non-object argument values serialize as JSON fragments`() {
    var parser = GraniteParser()
    let input = #"<|tool_call|> [{"name": "array_args", "arguments": [1, 2, 3]}, {"name": "string_args", "arguments": "raw"}, {"name": "bool_args", "arguments": true}]"#
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
  func `Tool call IDs follow fc_/call_ prefix convention`() {
    var parser = GraniteParser()
    let events = parser.process(ParserInput(text: #"<|tool_call|> [{"name": "fn", "arguments": {}}]"#))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
  }
}

@Suite("GraniteParser — multiple tool calls")
struct GraniteMultipleToolCallTests {
  @Test
  func `Parallel calls in one envelope`() throws {
    var parser = GraniteParser()
    let input = """
    <|tool_call|> [
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

@Suite("GraniteParser — surrounding text")
struct GraniteSurroundingTextTests {
  @Test
  func `Leading text before envelope emits as message`() {
    // Departure from vLLM's reference: vLLM strips surrounding content
    // when tool calls are present (and surrounding-text is xfailed in
    // both streaming and nonstreaming). We surface it instead —
    // consistent with how the rest of this package's parsers handle
    // preamble.
    var parser = GraniteParser()
    let input = #"Let me check the weather. <|tool_call|> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#
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
    var parser = GraniteParser()
    let input = """
    <|tool_call|> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
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

@Suite("GraniteParser — escaped strings")
struct GraniteEscapedStringTests {
  @Test
  func `Escaped quotes, backslashes, newlines, unicode`() throws {
    var parser = GraniteParser()
    let input = #"<tool_call> [{"name": "test_function", "arguments": {"quoted": "He said \"hello\"", "path": "C:\\Users\\file.txt", "newline": "line1\nline2", "unicode": "emoji: 🎉"}}]"#
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

@Suite("GraniteParser — streaming")
struct GraniteStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = #"<|tool_call|> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#

    var oneShot = GraniteParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = GraniteParser()
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
  func `Char-by-char reconstruction with <tool_call> marker matches one-shot`() {
    let input = #"<tool_call> [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#

    var oneShot = GraniteParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = GraniteParser()
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
  func `Split <|tool_call|> across chunks doesn't leak as content`() {
    // Hold-back must keep the partial `<|too` from leaking into a
    // message item so that when `l_call|>` arrives the parser still
    // recognizes the marker.
    var parser = GraniteParser()
    var events = parser.process(ParserInput(text: "<|too"))
    events += parser.process(ParserInput(text: #"l_call|> [{"name":"fn","arguments":{}}]"#))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "fn")
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] {
        return t.text
      }
      return nil
    }
    #expect(messages.allSatisfy { !$0.contains("<|too") && !$0.contains("<|tool_call") })
  }

  @Test
  func `Marker arrives in one chunk, array in another`() {
    var parser = GraniteParser()
    var events = parser.process(ParserInput(text: "<|tool_call|> "))
    events += parser.process(ParserInput(text: #"[{"name":"fn","arguments":{}}]"#))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "fn")
  }
}

@Suite("GraniteParser — malformed input")
struct GraniteMalformedInputTests {
  @Test
  func `Truncated mid-call (unclosed brace) surfaces as content at finalize`() {
    var parser = GraniteParser()
    let input = #"<|tool_call|> [{"name": "func", "arguments": {"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty, "Truncated array should not surface a partial tool call")
  }

  @Test
  func `Marker followed by an object (not array) is treated as content`() {
    // vLLM rejects this because the parser only accepts a JSON array
    // after the marker.
    var parser = GraniteParser()
    let input = #"<|tool_call|> {"name": "func", "arguments": {}}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Bare JSON array without marker — content, no tool calls`() {
    // vLLM's `extract_tool_calls` accepts a bare array (no marker) when
    // nothing precedes it but whitespace. We're stricter: the marker
    // is required, and a bare array routes to the JSON fallback parser
    // when callers want that behavior. This avoids false-positive tool
    // detection in normal prose that contains a JSON array.
    var parser = GraniteParser()
    let input = #"[{"name": "func", "arguments": {}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Invalid JSON inside the array surfaces as content`() {
    var parser = GraniteParser()
    let input = "<|tool_call|> [ This is just text ]"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Empty array emits no tool calls`() {
    var parser = GraniteParser()
    let input = "<|tool_call|> []"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}

@Suite("GraniteParser — reasoning")
struct GraniteReasoningTests {
  // Fixtures derived from vLLM's `tests/reasoning/test_granite_reasoning_parser.py`
  // shape: `Here is my thought process: <reasoning> Here is my response:
  // <content>`. Both `Here is my` and `Here's my` variants must be
  // accepted (the latter appears in some quantized checkpoints).

  @Test
  func `Reasoning then content`() {
    var parser = GraniteParser()
    let input = "Here is my thought process: Hmm, let me think. Here is my response: Hello!"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text.contains("Hmm, let me think."))
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else { Issue.record("Expected message"); return }
    #expect(t.text.contains("Hello!"))
  }

  @Test
  func `Reasoning can complete without response content`() {
    var parser = GraniteParser()
    let input = "Here is my thought process: This is a reasoning section Here is my response:"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == " This is a reasoning section ")
    #expect(r.status == .completed)
  }

  @Test
  func `Reasoning and response preserve multiple lines`() {
    var parser = GraniteParser()
    let input = "Here is my thought process: This\nThat Here is my response: This is the rest\nThat"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == " This\nThat ")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == " This is the rest\nThat")
  }

  @Test
  func `Apostrophe variant — Here's my … is accepted`() {
    var parser = GraniteParser()
    let input = "Here's my thought process: thinking. Here's my response: done."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }.first
    let message = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }.first
    #expect(reasoning?.contains("thinking.") == true)
    #expect(message?.contains("done.") == true)
  }

  @Test
  func `Reasoning then content then tool call`() {
    var parser = GraniteParser()
    let input = "Here is my thought process: Need to fetch weather. Here is my response: Calling the API. <|tool_call|> [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}]"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    #expect(reasoning.count == 1)
    #expect(reasoning[0].contains("Need to fetch weather."))
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }

  @Test
  func `No reasoning markers — entire output is content`() {
    var parser = GraniteParser()
    let input = "This is a plain answer with no reasoning markers."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty, "No reasoning expected when markers absent")
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else { Issue.record("Expected message"); return }
    #expect(t.text == "This is a plain answer with no reasoning markers.")
  }

  @Test
  func `Output starting with H but not a marker — flushes as content`() {
    // Mirrors vLLM's `_get_delta_message_with_no_reasoning_bounds` behavior:
    // the buffer is held while it remains a prefix of a think-start;
    // once broken (here at "Hello "), the held bytes flush as content.
    var parser = GraniteParser()
    let input = "Hello, world!"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else { Issue.record("Expected message"); return }
    #expect(t.text == "Hello, world!")
  }

  @Test
  func `Char-by-char streaming preserves reasoning vs content split`() {
    let input = "Here is my thought process: A B C. Here is my response: Done."

    var oneShot = GraniteParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = GraniteParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    let oneShotReasoning = oneShotItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let streamedReasoning = streamedItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    #expect(oneShotReasoning == streamedReasoning)
    #expect(oneShotReasoning.first?.contains("A B C.") == true)
  }

  @Test
  func `Reasoning truncated at EOS — surfaces as incomplete`() {
    // No `Here is my response:` marker arrives before EOS. The parser
    // should still emit the reasoning text and close the item with
    // `incomplete` status.
    var parser = GraniteParser()
    let input = "Here is my thought process: still thinking when truncated"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text.contains("still thinking when truncated"))
    #expect(r.status == .incomplete)
  }

  @Test
  func `Continuation: priorOutput inside reasoning starts in reasoning phase`() {
    // The chat history shows `Here is my thought process:` was emitted
    // but no `Here is my response:` yet — we're mid-reasoning. The
    // parser dispatched via .makeParser must start already inside
    // reasoning so the next chunk's text is reasoning, not content.
    let prior = "Here is my thought process: I started thinking"
    var parser = ResponseFormat.granite.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: prior,
    )
    let events = parser.process(ParserInput(text: " and continued. Here is my response: Done."))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text.contains("and continued."))
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else { Issue.record("Expected message"); return }
    #expect(t.text.contains("Done."))
  }

  @Test
  func `Continuation: priorOutput past response-start starts in normal phase`() {
    // Prior output already contains `Here is my response:`; the new
    // chunk is plain content (no reasoning).
    let prior = "Here is my thought process: x. Here is my response: hello"
    var parser = ResponseFormat.granite.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: prior,
    )
    let events = parser.process(ParserInput(text: " world!"))
      + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty, "No reasoning expected for past-reasoning continuation")
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else { Issue.record("Expected message"); return }
    #expect(t.text == " world!")
  }
}

@Suite("ResponseFormat dispatch — Granite 3.x")
struct GraniteDispatchTests {
  @Test
  func `Granite 3.0 routes to .granite by name`() {
    let f = ResponseFormat.infer(
      modelName: "ibm-granite/granite-3.0-8b-instruct",
      modelType: "granite",
      modelConfig: [:],
    )
    #expect(f == .granite)
  }

  @Test
  func `Granite 3.1 routes to .granite by name`() {
    let f = ResponseFormat.infer(
      modelName: "ibm-granite/granite-3.1-8b-instruct",
      modelType: "granite",
      modelConfig: [:],
    )
    #expect(f == .granite)
  }

  @Test
  func `Granite 3.2 routes to .granite by name`() {
    let f = ResponseFormat.infer(
      modelName: "ibm-granite/granite-3.2-8b-instruct",
      modelType: "granite",
      modelConfig: [:],
    )
    #expect(f == .granite)
  }

  @Test
  func `Bare granite-3 prefix also routes to .granite`() {
    let f = ResponseFormat.infer(
      modelName: "granite-3.0-2b-instruct",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .granite)
  }

  @Test
  func `Granite 4 does NOT route to .granite — distinct wire format`() {
    let f = ResponseFormat.infer(
      modelName: "ibm-granite/granite-4.0-h-tiny",
      modelType: "granitemoehybrid",
      modelConfig: [:],
    )
    #expect(f != .granite, "Granite 4 has its own Hermes-shaped wire format")
  }
}
