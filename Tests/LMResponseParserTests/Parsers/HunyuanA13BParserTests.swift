// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Fixtures derived from vLLM's
// `tests/tool_parsers/test_hunyuan_a13b_tool_parser.py` and
// `tests/reasoning/test_hunyuan_reasoning_parser.py`. The reasoning and
// tool-call shapes are tightly coupled in Hunyuan A13B's template so
// they're tested together.

@Suite("HunyuanA13BParser — plain text")
struct HunyuanA13BPlainTextTests {
  @Test
  func `Plain text without markers emits a single message`() {
    var parser = HunyuanA13BParser()
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
  func `Empty stream emits nothing`() {
    var parser = HunyuanA13BParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("HunyuanA13BParser — tool calls")
struct HunyuanA13BToolCallTests {
  @Test
  func `Single call inside <tool_calls> envelope`() throws {
    var parser = HunyuanA13BParser()
    let input = #"<tool_calls>[{"name": "get_weather", "arguments": {"city": "San Francisco", "metric": "celsius"}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    let data = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["city"] as? String == "San Francisco")
    #expect(decoded["metric"] as? String == "celsius")
  }

  @Test
  func `Parallel calls in one envelope`() {
    var parser = HunyuanA13BParser()
    let input = #"<tool_calls>[{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]</tool_calls>"#
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
  func `Nested object arguments are preserved`() throws {
    var parser = HunyuanA13BParser()
    let input = #"<tool_calls>[{"name": "complex_tool", "arguments": {"level1": {"level2": {"level3": {"value": 123}}}}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "complex_tool")
    let data = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let level1 = try #require(decoded["level1"] as? [String: Any])
    let level2 = try #require(level1["level2"] as? [String: Any])
    let level3 = try #require(level2["level3"] as? [String: Any])
    #expect(level3["value"] as? Int == 123)
  }

  @Test
  func `Content before tool call surfaces as message`() {
    var parser = HunyuanA13BParser()
    let input = #"I will call the tool now. <tool_calls>[{"name": "get_weather", "arguments": {"city": "Boston"}}]</tool_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messages.first?.contains("I will call the tool now.") == true)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
  }

  @Test
  func `Content after tool call surfaces as a trailing message`() {
    var parser = HunyuanA13BParser()
    let input = """
    <tool_calls>[{"name": "get_weather", "arguments": {"city": "Seattle"}}]</tool_calls>
    Thank you!
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    #expect(messages.first?.contains("Thank you!") == true)
  }
}

@Suite("HunyuanA13BParser — reasoning")
struct HunyuanA13BReasoningTests {
  @Test
  func `Reasoning then content via <think> + <answer> envelope`() {
    let input = "<think>\nThis is a reasoning section\n</think>\n<answer>\nThis is the rest\n</answer>"
    var parser = HunyuanA13BParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(reasoning.count == 1)
    #expect(reasoning[0].contains("reasoning section"))
    #expect(messages.first?.contains("This is the rest") == true)
  }

  @Test
  func `Empty <think> with answer surfaces only the answer`() {
    // vLLM's NO_REASONING_QUICK_THOUGHT fixture: the model emits an
    // empty think block, then the answer envelope.
    let input = "<think>\n\n</think>\n<answer>\nThis is the rest\n</answer>"
    var parser = HunyuanA13BParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    #expect(reasoning.isEmpty)
    #expect(messages.first?.contains("This is the rest") == true)
  }

  @Test
  func `Reasoning can complete without response content`() {
    let input = "<think>\nThis is a reasoning section\n</think>\n<answer>\n"
    var parser = HunyuanA13BParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "This is a reasoning section")
    #expect(r.status == .completed)
  }

  @Test
  func `Reasoning and response preserve multiple lines`() {
    let input = "<think>\nThis\nThat\n</think>\n<answer>\nThis is the rest\nThat"
    var parser = HunyuanA13BParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "This\nThat")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "This is the rest\nThat")
  }

  @Test
  func `Reasoning then tool call inside the answer envelope`() {
    let input = """
    <think>
    planning
    </think>
    <answer>
    <tool_calls>[{"name": "fn", "arguments": {"x": 1}}]</tool_calls>
    </answer>
    """
    var parser = HunyuanA13BParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.count == 1)
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "fn")
  }

  @Test
  func `Tool call inside <think> is treated as reasoning, not extracted`() {
    // The reasoning phase consumes everything between `<think>\n` and
    // `\n</think>\n` as reasoning text, so an embedded `<tool_calls>`
    // there never reaches the tool-call extractor — matching vLLM's
    // `preprocess_model_output` behavior, which filters out tool
    // calls inside the think block.
    let input = """
    <think>
    <tool_calls>[{"name": "fake", "arguments": {}}]</tool_calls>
    </think>
    <answer>
    Done
    </answer>
    """
    var parser = HunyuanA13BParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty, "Tool calls inside <think> must not be extracted")
  }
}

@Suite("HunyuanA13BParser — streaming")
struct HunyuanA13BStreamingTests {
  @Test
  func `Char-by-char tool-call streaming matches one-shot`() {
    let input = #"<tool_calls>[{"name": "get_weather", "arguments": {"city": "Boston"}}]</tool_calls>"#

    var oneShot = HunyuanA13BParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = HunyuanA13BParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    let oneShotCalls = oneShotItems.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    let streamedCalls = streamedItems.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(oneShotCalls.count == streamedCalls.count)
    #expect(oneShotCalls.first?.name == streamedCalls.first?.name)
    #expect(oneShotCalls.first?.arguments == streamedCalls.first?.arguments)
  }
}

@Suite("HunyuanA13BParser — answer envelope marker splits")
struct HunyuanA13BAnswerSplitTests {
  // The optional `<answer>\n` opener that appears when reasoning is
  // absent must be held back across chunk boundaries — otherwise a
  // partial prefix like `<ans` leaks as content.
  @Test
  func `Split answer-envelope opener does not leak partial bytes as content`() {
    var parser = HunyuanA13BParser()
    var events = parser.process(ParserInput(text: "<ans"))
    events += parser.process(ParserInput(text: "wer>\nbody."))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let messageTexts = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messageTexts.first == "body.", "Partial `<ans` must be held back, not leaked")
  }
}

@Suite("ResponseFormat dispatch — Hunyuan A13B")
struct HunyuanA13BDispatchTests {
  @Test
  func `Hunyuan-A13B-Instruct routes to .hunyuanA13B by name`() {
    let f = ResponseFormat.infer(
      modelName: "tencent/Hunyuan-A13B-Instruct",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .hunyuanA13B)
  }

  @Test
  func `Bare hunyuan-a13b prefix also routes`() {
    let f = ResponseFormat.infer(
      modelName: "hunyuan-a13b-pretrain",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .hunyuanA13B)
  }
}
