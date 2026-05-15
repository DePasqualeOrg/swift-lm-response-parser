// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// Fixtures lifted from vLLM's `tests/tool_parsers/test_internlm2_tool_parser.py`
// and sglang's `python/sglang/srt/function_call/internlm_detector.py`,
// adapted to the streaming event shape this package emits.

@Suite("InternlmParser — plain text")
struct InternlmPlainTextTests {
  @Test
  func `Plain text without tool calls emits a single message`() {
    var parser = InternlmParser()
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
    var parser = InternlmParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("InternlmParser — single tool call")
struct InternlmSingleToolCallTests {
  @Test
  func `Single call with vLLM-style unspaced markers`() throws {
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }

  @Test
  func `Single call with sglang-style spaced markers`() throws {
    // sglang's `InternlmDetector.bot_token = "<|action_start|> <|plugin|>"`
    // (with a literal space) — observed in some checkpoints.
    var parser = InternlmParser()
    let input = #"<|action_start|> <|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }

  @Test
  func `Arguments key fallback when parameters is absent`() throws {
    // Both sglang and vLLM accept `arguments` as an alias for
    // `parameters`. Verify our parser does too.
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "fn", "arguments": {"x": 1}}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Empty arguments object`() throws {
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "refresh", "parameters": {}}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "refresh")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded.isEmpty)
  }

  @Test
  func `Mixed argument types`() throws {
    var parser = InternlmParser()
    let input = """
    <|action_start|><|plugin|>{
      "name": "test_function",
      "parameters": {
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
    }<|action_end|>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["string_field"] as? String == "hello")
    #expect(decoded["int_field"] as? Int == 42)
    #expect(decoded["float_field"] as? Double == 3.14)
    #expect(decoded["bool_field"] as? Bool == true)
    #expect(decoded["null_field"] is NSNull)
    #expect(decoded["array_field"] as? [String] == ["a", "b", "c"])
    let nested = try #require(decoded["object_field"] as? [String: String])
    #expect(nested["nested"] == "value")
    #expect((decoded["empty_array"] as? [Any])?.isEmpty == true)
    #expect((decoded["empty_object"] as? [String: Any])?.isEmpty == true)
  }

  @Test
  func `Escaped strings inside parameters`() throws {
    var parser = InternlmParser()
    let input = #"""
    <|action_start|><|plugin|>{
      "name": "test_function",
      "parameters": {
        "quoted": "He said \"hello\"",
        "path": "C:\\Users\\file.txt",
        "newline": "line1\nline2",
        "unicode": "emoji: 🎉"
      }
    }<|action_end|>
    """#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["quoted"] as? String == #"He said "hello""#)
    #expect(decoded["path"] as? String == #"C:\Users\file.txt"#)
    #expect(decoded["newline"] as? String == "line1\nline2")
    #expect(decoded["unicode"] as? String == "emoji: 🎉")
  }

  @Test
  func `String parameters are serialized as valid JSON fragments`() {
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "func", "parameters": "not a dict"}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "func")
    #expect(f.arguments == #""not a dict""#)
  }

  @Test
  func `Tool call IDs follow fc_/call_ prefix convention`() {
    var parser = InternlmParser()
    let events = parser.process(ParserInput(text: #"<|action_start|><|plugin|>{"name": "fn", "parameters": {}}<|action_end|>"#))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
  }
}

@Suite("InternlmParser — multiple sequential tool calls")
struct InternlmMultipleToolCallTests {
  @Test
  func `Two sequential envelopes both surface as tool calls`() {
    // sglang's detector supports sequential calls (vLLM's reference
    // is single-shot only). Both envelopes use the same start
    // variant.
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "a", "parameters": {"x": 1}}<|action_end|><|action_start|><|plugin|>{"name": "b", "parameters": {"y": 2}}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "a")
    #expect(toolCalls[1].name == "b")
  }

  @Test
  func `Mixed start variants in two sequential envelopes`() {
    // First call uses vLLM-style unspaced; second uses sglang-style
    // spaced. The parser must accept both within one stream.
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "a", "parameters": {}}<|action_end|><|action_start|> <|plugin|>{"name": "b", "parameters": {}}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "a")
    #expect(toolCalls[1].name == "b")
  }
}

@Suite("InternlmParser — surrounding text")
struct InternlmSurroundingTextTests {
  @Test
  func `Leading text before envelope emits as message`() {
    var parser = InternlmParser()
    let input = #"What's the weather like? <|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(t.text.contains("What's the weather like?"))
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Trailing text after envelope emits as message`() {
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|> Done."#
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

@Suite("InternlmParser — streaming")
struct InternlmStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = #"<|action_start|><|plugin|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}<|action_end|>"#

    var oneShot = InternlmParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = InternlmParser()
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
  func `Split <|action_start|> across chunks doesn't leak as content`() {
    var parser = InternlmParser()
    var events = parser.process(ParserInput(text: "<|action_st"))
    events += parser.process(ParserInput(text: #"art|><|plugin|>{"name":"fn","parameters":{}}<|action_end|>"#))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] {
        return t.text
      }
      return nil
    }
    #expect(messages.allSatisfy { !$0.contains("<|action_st") }, "Partial start marker must not leak")
  }
}

@Suite("InternlmParser — malformed input")
struct InternlmMalformedInputTests {
  @Test
  func `Truncated mid-call surfaces as content at finalize`() {
    var parser = InternlmParser()
    let input = #"<|action_start|><|plugin|>{"name": "func", "parameters": {"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Invalid JSON inside envelope surfaces as content`() {
    var parser = InternlmParser()
    let input = "<|action_start|><|plugin|>not json<|action_end|>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Bare <|action_start|> without <|plugin|> is plain content`() {
    // The detector requires both halves of the start sequence. A bare
    // `<|action_start|>` shouldn't trigger tool extraction.
    var parser = InternlmParser()
    let input = #"<|action_start|>{"name": "func"}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}

@Suite("ResponseFormat dispatch — InternLM")
struct InternlmDispatchTests {
  @Test
  func `InternLM2-7B routes to .internlm by name`() {
    let f = ResponseFormat.infer(
      modelName: "internlm/internlm2-chat-7b",
      modelType: "internlm2",
      modelConfig: [:],
    )
    #expect(f == .internlm)
  }

  @Test
  func `InternLM2.5-7B routes to .internlm by name`() {
    let f = ResponseFormat.infer(
      modelName: "internlm/internlm2_5-7b-chat",
      modelType: "internlm2",
      modelConfig: [:],
    )
    #expect(f == .internlm)
  }

  @Test
  func `Intern-S1 routes to .internlm by name`() {
    let f = ResponseFormat.infer(
      modelName: "internlm/Intern-S1",
      modelType: "internlm2",
      modelConfig: [:],
    )
    #expect(f == .internlm)
  }
}
