// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// vLLM's `DeepSeekV3ReasoningParser` is a delegator: when
// `chat_template_kwargs.thinking=True`, it dispatches to R1's
// `<think>...</think>` extraction. The companion
// `DeepSeekV3ReasoningWithThinkingParser` defaults thinking on and is
// reused for `glm45` and `holo2`. These tests cover the equivalent
// `acceptThink:` flag on Swift's V3-family and GLM 4 parsers.

@Suite("DeepSeekV3Parser — thinking flag")
struct DeepSeekV3ThinkingTests {
  @Test
  func `acceptThink false — base behavior, no reasoning extraction`() {
    // Default V3 parser doesn't observe `<think>`; the markers flow
    // through as plain message text.
    var parser = DeepSeekV3Parser()
    let input = "<think>thinking</think>plain answer"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty, "Default V3 must not extract reasoning")
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messages.first?.contains("<think>") == true)
  }

  @Test
  func `acceptThink true — extracts reasoning preamble`() {
    var parser = DeepSeekV3Parser(acceptThink: true)
    let input = "<think>working it out</think>final answer"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(reasoning.first == "working it out")
    #expect(messages.first == "final answer")
  }

  @Test
  func `acceptThink true — implicit opener (chat template injected <think>)`() {
    // When the chat template injects `<think>` into the prompt, the
    // model output starts mid-reasoning with no opener. The default
    // initial state (.reasoning) handles this.
    var parser = DeepSeekV3Parser(acceptThink: true)
    let input = "thinking content</think>after"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(reasoning.first == "thinking content")
    #expect(messages.first == "after")
  }

  @Test
  func `acceptThink true — reasoning followed by tool call`() {
    let input = """
    <think>need to look up weather</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"city": "Paris"}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekV3Parser(acceptThink: true)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.first == "need to look up weather")
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }

  @Test
  func `Truncated reasoning at EOS surfaces as incomplete`() {
    var parser = DeepSeekV3Parser(acceptThink: true)
    let events = parser.process(ParserInput(text: "<think>still thinking when truncated"))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text.contains("still thinking when truncated"))
    #expect(r.status == .incomplete)
  }

  @Test
  func `Char-by-char streaming preserves reasoning vs content split`() {
    let input = "<think>A B C</think>Done."

    var oneShot = DeepSeekV3Parser(acceptThink: true)
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = DeepSeekV3Parser(acceptThink: true)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    let oneShotReasoning = oneShotItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let streamedReasoning = streamedItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    #expect(oneShotReasoning == streamedReasoning)
  }

  @Test
  func `InitialState normal skips reasoning extraction`() {
    // Continuation request post-`</think>`: the prior chunk has
    // already exited reasoning. New parser starts in normal phase.
    var parser = DeepSeekV3Parser(acceptThink: true, initialState: .normal)
    let events = parser.process(ParserInput(text: "<think>still thinking when truncated"))
      + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty, "InitialState .normal must skip reasoning extraction")
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messages.first?.contains("<think>") == true)
  }
}

@Suite("DeepSeekV31Parser — thinking flag")
struct DeepSeekV31ThinkingTests {
  @Test
  func `acceptThink false — base behavior, no reasoning extraction`() {
    var parser = DeepSeekV31Parser()
    let input = "<think>thinking</think>plain answer"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty)
  }

  @Test
  func `acceptThink true — reasoning followed by V3.1 tool call`() {
    let input = #"<think>need weather</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#
    var parser = DeepSeekV31Parser(acceptThink: true)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.first == "need weather")
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }

  @Test
  func `Char-by-char streaming preserves reasoning vs content split`() {
    let input = #"<think>R</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>fn<｜tool▁sep｜>{"a":1}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#

    var oneShot = DeepSeekV31Parser(acceptThink: true)
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = DeepSeekV31Parser(acceptThink: true)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    let oneShotReasoning = oneShotItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let streamedReasoning = streamedItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    #expect(oneShotReasoning == streamedReasoning)
    #expect(oneShotReasoning.first == "R")
  }
}

@Suite("DeepSeekV32Parser — thinking flag")
struct DeepSeekV32ThinkingTests {
  @Test
  func `acceptThink false — base behavior, no reasoning extraction`() {
    var parser = DeepSeekV32Parser()
    let input = "<think>thinking</think>plain answer"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty)
  }

  @Test
  func `acceptThink true — reasoning followed by DSML tool call`() {
    let input = """
    <think>need weather</think><｜DSML｜function_calls><｜DSML｜invoke name="get_weather"><｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>
    """
    var parser = DeepSeekV32Parser(acceptThink: true)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.first == "need weather")
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }
}

@Suite("Glm4Parser — thinking flag")
struct Glm4ThinkingTests {
  @Test
  func `acceptThink false — base behavior, no reasoning extraction`() {
    var parser = Glm4Parser()
    let input = "<think>thinking</think>plain answer"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty, "Default GLM 4 must not extract reasoning")
  }

  @Test
  func `acceptThink true — extracts reasoning preamble before tool call`() {
    let input = """
    <think>working it out</think><tool_call>get_weather
    <arg_key>city</arg_key>
    <arg_value>Paris</arg_value>
    </tool_call>
    """
    var parser = Glm4Parser(acceptThink: true)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.first == "working it out")
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }

  @Test
  func `acceptThink true — implicit opener (chat template injected <think>)`() {
    // GLM 4.5+ chat templates inject `<think>` into the prompt.
    var parser = Glm4Parser(acceptThink: true)
    let input = "thinking content</think>final answer"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(reasoning.first == "thinking content")
    #expect(messages.first == "final answer")
  }

  @Test
  func `acceptThink true — tool call interrupts reasoning without closing think`() {
    let input = """
    <think>need weather<tool_call>get_weather
    <arg_key>city</arg_key>
    <arg_value>Paris</arg_value>
    </tool_call>
    """
    var parser = Glm4Parser(acceptThink: true)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.first == "need weather")
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }

  @Test
  func `Char-by-char streaming preserves reasoning vs content split`() {
    let input = """
    <think>R</think><tool_call>fn
    <arg_key>k</arg_key>
    <arg_value>v</arg_value>
    </tool_call>
    """

    var oneShot = Glm4Parser(acceptThink: true)
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = Glm4Parser(acceptThink: true)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    let oneShotReasoning = oneShotItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let streamedReasoning = streamedItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    #expect(oneShotReasoning == streamedReasoning)
    #expect(oneShotReasoning.first == "R")
  }

  @Test
  func `Truncated reasoning at EOS surfaces as incomplete`() {
    var parser = Glm4Parser(acceptThink: true)
    let events = parser.process(ParserInput(text: "<think>still thinking when truncated"))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text.contains("still thinking when truncated"))
    #expect(r.status == .incomplete)
  }
}
