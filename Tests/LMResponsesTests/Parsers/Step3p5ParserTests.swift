// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

/// Step-3.5-Flash uses the Qwen 3 Coder XML wire format for tool calls
/// but adds a reasoning quirk: it habitually emits a stray `\n`
/// immediately before and/or after `</think>`. The `.step3p5` format
/// configures Qwen3XmlParser with `trimNewlineAroundThinkEnd: true`
/// to strip those extra newlines.

@Suite("Step3p5Parser — newline trim around </think>")
struct Step3p5NewlineTrimTests {
  @Test
  func `Trailing newline before </think> is dropped from reasoning`() {
    var parser = Qwen3XmlParser(
      initialState: .reasoning,
      trimNewlineAroundThinkEnd: true,
    )
    let text = "Thinking through this.\n</think>The answer is 42."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Thinking through this.", "Trailing \\n must be dropped, not preserved")
    } else {
      Issue.record("Expected reasoning text content")
    }
  }

  @Test
  func `Leading newline after </think> is dropped from message`() {
    var parser = Qwen3XmlParser(
      initialState: .reasoning,
      trimNewlineAroundThinkEnd: true,
    )
    let text = "Thinking.</think>\nThe answer is 42."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "The answer is 42.", "Leading \\n must be dropped from message")
    } else {
      Issue.record("Expected message text content")
    }
  }

  @Test
  func `Both newline trims fire when </think> is bracketed by newlines`() {
    var parser = Qwen3XmlParser(
      initialState: .reasoning,
      trimNewlineAroundThinkEnd: true,
    )
    let text = "Thinking.\n</think>\nThe answer."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Thinking.")
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "The answer.")
    }
  }

  @Test
  func `Mid-reasoning newline that is NOT immediately before </think> is preserved`() {
    var parser = Qwen3XmlParser(
      initialState: .reasoning,
      trimNewlineAroundThinkEnd: true,
    )
    let text = "Line one.\nLine two.</think>Answer."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Line one.\nLine two.", "Internal \\n should not be touched")
    }
  }

  @Test
  func `Char-by-char streaming with trim matches single-shot`() {
    let text = "Thinking through.\n</think>\nThe answer."
    var single = Qwen3XmlParser(
      initialState: .reasoning,
      trimNewlineAroundThinkEnd: true,
    )
    let singleEvents = single.process(ParserInput(text: text)) + single.finalize()
    let singleItems = accumulateItems(from: singleEvents)

    var streamed = Qwen3XmlParser(
      initialState: .reasoning,
      trimNewlineAroundThinkEnd: true,
    )
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in text {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(singleItems.count == streamedItems.count)
    let singleReasoning = singleItems.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let streamedReasoning = streamedItems.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    if let s = singleReasoning.first, let c = streamedReasoning.first,
       case let .reasoningText(sr) = s.content[0],
       case let .reasoningText(cr) = c.content[0]
    {
      #expect(sr.text == cr.text)
    }
    let singleMsgs = singleItems.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let streamedMsgs = streamedItems.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    if let s = singleMsgs.first, let c = streamedMsgs.first,
       case let .outputText(st) = s.content[0],
       case let .outputText(ct) = c.content[0]
    {
      #expect(st.text == ct.text)
    }
  }

  @Test
  func `Trim mode off (default) preserves the surrounding newlines`() {
    var parser = Qwen3XmlParser(initialState: .reasoning)
    let text = "Thinking.\n</think>\nThe answer."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Thinking.\n", "Default behavior preserves trailing \\n")
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "\nThe answer.", "Default behavior preserves leading \\n")
    }
  }
}

@Suite("ResponseFormat dispatch — Step-3.5-Flash")
struct Step3p5DispatchTests {
  @Test
  func `Step-3.5-Flash name routes to .step3p5`() {
    let f = ResponseFormat.infer(
      modelName: "stepfun-ai/step-3.5-flash",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .step3p5)
  }

  @Test
  func `Bare step-3.5 prefix also routes to .step3p5`() {
    let f = ResponseFormat.infer(
      modelName: "step-3.5-flash",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .step3p5)
  }

  @Test
  func `Factory starts in implicit reasoning state and trims bracket newlines`() {
    var parser = ResponseFormat.step3p5.makeParser(tokenizer: StubTokenizer())
    let text = "Thinking.\n</think>\nThe answer."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)

    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Thinking.")
    } else {
      Issue.record("Expected reasoning text content")
    }

    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "The answer.")
    } else {
      Issue.record("Expected message text content")
    }
  }

  @Test
  func `Factory parses Step-3.5 XML tool calls`() {
    var parser = ResponseFormat.step3p5.makeParser(tokenizer: StubTokenizer())
    let text = """
    <tool_call>
    <function=get_weather>
    <parameter=location>Barcelona</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    #expect(toolCalls[0].arguments == #"{"location": "Barcelona"}"#)
  }
}
