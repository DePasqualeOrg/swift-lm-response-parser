// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

/// Nemotron V3 reasoning is the DeepSeek R1 reasoning shape with one
/// quirk: when the chat template ran with `enable_thinking=False` /
/// `force_nonempty_content=True`, the model emits no `<think>` block
/// at all and the entire output should be treated as message content
/// rather than reasoning. vLLM's `NemotronV3ReasoningParser` extends
/// `DeepSeekR1ReasoningParser` and swaps reasoning ↔ content on that
/// case.
///
/// In our streaming parser, the swap is gated by
/// `swapWhenContentEmpty: true`. The implementation holds reasoning-
/// state output until either an exit marker (`</think>` or
/// `<｜tool▁calls▁begin｜>`) proves it was reasoning, or finalize
/// arrives without an exit marker (in which case it's flushed as a
/// message).

@Suite("DeepSeekR1Parser — Nemotron V3 swapWhenContentEmpty")
struct NemotronV3SwapTests {
  @Test
  func `Without an exit marker, held text becomes a message at finalize`() {
    var parser = DeepSeekR1Parser(
      initialState: .reasoning,
      swapWhenContentEmpty: true,
    )
    let events = parser.process(ParserInput(text: "Direct answer with no thinking.")) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.isEmpty, "Swap mode should suppress reasoning when no </think> arrives")
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "Direct answer with no thinking.")
    } else {
      Issue.record("Expected message text content")
    }
  }

  @Test
  func `Without an exit marker, a leading <think> opener is stripped before swap to message`() {
    var parser = DeepSeekR1Parser(
      initialState: .reasoning,
      swapWhenContentEmpty: true,
    )
    let events = parser.process(ParserInput(text: "<think>Direct answer.")) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.isEmpty)
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "Direct answer.")
    } else {
      Issue.record("Expected message text content")
    }
  }

  @Test
  func `With a </think> exit marker, held text is flushed as reasoning`() {
    var parser = DeepSeekR1Parser(
      initialState: .reasoning,
      swapWhenContentEmpty: true,
    )
    let events = parser.process(ParserInput(text: "Real thinking here.</think>Then the answer.")) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Real thinking here.")
    } else {
      Issue.record("Expected reasoning text content")
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "Then the answer.")
    } else {
      Issue.record("Expected message text content")
    }
  }

  @Test
  func `With a tool call exit marker, held text is flushed as reasoning before the call`() {
    let text = """
    Pre-call thinking.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>my_func
    ```json
    {"x": 1}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekR1Parser(
      initialState: .reasoning,
      swapWhenContentEmpty: true,
    )
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Pre-call thinking.")
    } else {
      Issue.record("Expected reasoning text content")
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "my_func")
  }

  @Test
  func `Default swap-off behavior is unchanged: text without </think> still becomes reasoning`() {
    var parser = DeepSeekR1Parser(initialState: .reasoning)
    let events = parser.process(ParserInput(text: "Just thinking, no closer.")) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoning.count == 1, "Without swap mode, missing </think> should still produce reasoning")
    #expect(messages.isEmpty)
  }

  @Test
  func `Char-by-char streaming through swap-as-message matches single-shot`() {
    let text = "No closer here."
    var single = DeepSeekR1Parser(
      initialState: .reasoning,
      swapWhenContentEmpty: true,
    )
    let singleEvents = single.process(ParserInput(text: text)) + single.finalize()
    let singleItems = accumulateItems(from: singleEvents)

    var streamed = DeepSeekR1Parser(
      initialState: .reasoning,
      swapWhenContentEmpty: true,
    )
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in text {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(singleItems.count == streamedItems.count)
    let singleMsg = singleItems.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let streamedMsg = streamedItems.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(singleMsg.count == 1)
    #expect(streamedMsg.count == 1)
    if case let .outputText(s) = singleMsg[0].content[0],
       case let .outputText(c) = streamedMsg[0].content[0]
    {
      #expect(s.text == c.text)
    }
  }
}
