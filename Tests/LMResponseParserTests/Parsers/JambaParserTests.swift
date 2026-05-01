// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Jamba-specific behavior: a single `<tool_calls>...</tool_calls>` envelope
// wrapping a JSON array of `{name, arguments}` objects. Both parallel and
// single calls share the envelope.

@Suite("JambaParser — plain text")
struct JambaPlainTextTests {
  @Test
  func `Plain text without tool calls emits a single message`() {
    var parser = JambaParser()
    let events = parser.process(ParserInput(text: "Hello, how can I help?"))
      + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Hello, how can I help?")
  }

  @Test
  func `Empty stream emits nothing`() {
    var parser = JambaParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("JambaParser — single tool call")
struct JambaSingleToolCallTests {
  @Test
  func `Single call inside <tool_calls> envelope`() throws {
    var parser = JambaParser()
    let input = #"<tool_calls>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
  }

  @Test
  func `Empty arguments object`() throws {
    var parser = JambaParser()
    let input = #"<tool_calls>[{"name": "refresh", "arguments": {}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "refresh")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded.isEmpty)
  }

  @Test
  func `Non-object argument values serialize as JSON fragments`() {
    var parser = JambaParser()
    let input = #"<tool_calls>[{"name": "array_args", "arguments": [1, 2, 3]}, {"name": "string_args", "arguments": "raw"}, {"name": "bool_args", "arguments": true}]</tool_calls>"#
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
    var parser = JambaParser()
    let events = parser.process(ParserInput(text: #"<tool_calls>[{"name": "fn", "arguments": {}}]</tool_calls>"#))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
  }
}

@Suite("JambaParser — multiple tool calls")
struct JambaMultipleToolCallTests {
  @Test
  func `Parallel calls in one envelope`() throws {
    var parser = JambaParser()
    let input = #"<tool_calls>[{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "a")
    #expect(toolCalls[1].name == "b")
    let argsAData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let argsA = try #require(JSONSerialization.jsonObject(with: argsAData) as? [String: Int])
    #expect(argsA["x"] == 1)
  }
}

@Suite("JambaParser — surrounding text")
struct JambaSurroundingTextTests {
  @Test
  func `Leading text before envelope emits as message`() {
    var parser = JambaParser()
    let input = #"Let me check. <tool_calls>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(t.text == "Let me check. ")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Trailing text after envelope emits as message`() {
    var parser = JambaParser()
    let input = #"<tool_calls>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</tool_calls> Done."#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .functionCall = items[0] else { Issue.record("Expected function call"); return }
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected trailing message"); return
    }
    #expect(t.text.contains("Done."))
  }
}

@Suite("JambaParser — streaming")
struct JambaStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = #"<tool_calls>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</tool_calls>"#

    var oneShot = JambaParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = JambaParser()
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
  func `Split <tool_calls> across chunks doesn't leak as content`() {
    var parser = JambaParser()
    var events = parser.process(ParserInput(text: "<tool_ca"))
    events += parser.process(ParserInput(text: #"lls>[{"name":"fn","arguments":{}}]</tool_calls>"#))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messages.allSatisfy { !$0.contains("<tool_ca") })
  }
}

@Suite("JambaParser — malformed input")
struct JambaMalformedInputTests {
  @Test
  func `Truncated mid-call surfaces as content at finalize`() {
    var parser = JambaParser()
    let input = #"<tool_calls>[{"name": "func", "arguments": {"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Invalid JSON inside envelope surfaces as content`() {
    var parser = JambaParser()
    let input = "<tool_calls>not json</tool_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Object instead of array — no tool calls`() {
    // Jamba expects a JSON array; an object isn't a valid envelope body.
    var parser = JambaParser()
    let input = #"<tool_calls>{"name": "func", "arguments": {}}</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Empty array emits no tool calls`() {
    var parser = JambaParser()
    let input = "<tool_calls>[]</tool_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}

@Suite("ResponseFormat dispatch — Jamba")
struct JambaDispatchTests {
  @Test
  func `Jamba 1.5 routes to .jamba by name`() {
    let f = ResponseFormat.infer(
      modelName: "ai21labs/Jamba-1.5-Mini",
      modelType: "jamba",
      modelConfig: [:],
    )
    #expect(f == .jamba)
  }

  @Test
  func `Jamba 1.7 routes to .jamba by name`() {
    let f = ResponseFormat.infer(
      modelName: "ai21labs/AI21-Jamba-Large-1.7",
      modelType: "jamba",
      modelConfig: [:],
    )
    #expect(f == .jamba)
  }
}
