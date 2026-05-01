// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

/// MiniMax M2 has an alternate chat template that does NOT inject the
/// `<think>` opener into the prompt. With that template, the model
/// output starts mid-thought without any structural markers, and the
/// reasoning parser literally prepends `"<think>"` to the output and
/// treats everything as content.
///
/// In our streaming parser, the behavior is gated by
/// `appendThink: true` on `MiniMaxM2Parser`. When set, the parser:
///   - skips reasoning extraction entirely (initialState is forced to
///     `.normal`),
///   - prepends `<think>` to the first message delta it emits.

@Suite("MiniMaxM2Parser — appendThink template variant")
struct MiniMaxM2AppendThinkTests {
  @Test
  func `appendThink prepends <think> to the message and emits no reasoning item`() {
    var parser = MiniMaxM2Parser(appendThink: true)
    let text = "Some thinking here.</think>The answer."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoning.isEmpty, "appendThink mode must not emit reasoning items")
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "<think>Some thinking here.</think>The answer.",
              "Message must include the literal `<think>` prepend")
    } else {
      Issue.record("Expected message text content")
    }
  }

  @Test
  func `appendThink preserves tool calls`() {
    var parser = MiniMaxM2Parser(appendThink: true)
    let text = """
    Hi.<minimax:tool_call>
    <invoke name="get_weather">
    <parameter name="city">Paris</parameter>
    </invoke>
    </minimax:tool_call>
    """
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "<think>Hi.", "Pre-toolcall content gets <think> prepended")
    } else {
      Issue.record("Expected message text content")
    }
  }

  @Test
  func `appendThink ignores initialState`() {
    var parser = MiniMaxM2Parser(initialState: .reasoning, appendThink: true)
    let text = "Hello."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoning.isEmpty,
            "initialState should be ignored when appendThink is on – no reasoning extraction")
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "<think>Hello.")
    }
  }

  @Test
  func `appendThink off preserves the default reasoning extraction`() {
    var parser = MiniMaxM2Parser()
    let text = "Some thinking.</think>The answer."
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoning.count == 1, "Default behavior should still extract reasoning")
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text == "Some thinking.")
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "The answer.")
    }
  }

  @Test
  func `Char-by-char streaming with appendThink matches single-shot`() {
    let text = "Reasoning.</think>Answer."
    var single = MiniMaxM2Parser(appendThink: true)
    let singleEvents = single.process(ParserInput(text: text)) + single.finalize()
    let singleItems = accumulateItems(from: singleEvents)

    var streamed = MiniMaxM2Parser(appendThink: true)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in text {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(singleItems.count == streamedItems.count)
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
      #expect(st.text.hasPrefix("<think>"))
    }
  }
}
