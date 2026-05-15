// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// ERNIE 4.5 wraps Hermes-shaped tool calls in two ERNIE-specific
// envelopes: a `</think>` reasoning closer (opener typically injected
// by the chat template) and an optional `<response>...</response>`
// content envelope. Reasoning extraction is opt-in via `acceptThink:
// true`. Fixtures mirror vLLM's `Ernie45ToolParser` and
// `Ernie45ReasoningParser`.

@Suite("ErnieParser — tool calls (no reasoning)")
struct ErnieToolCallTests {
  @Test
  func `Single Hermes-shaped tool call is extracted`() throws {
    var parser = ErnieParser()
    let input = #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
  }

  @Test
  func `Multiple parallel tool calls`() {
    var parser = ErnieParser()
    let input = (
      #"<tool_call>{"name": "f1", "arguments": {}}</tool_call>"#
        + #"<tool_call>{"name": "f2", "arguments": {}}</tool_call>"#,
    )
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "f1")
    #expect(b.name == "f2")
  }

  @Test
  func `<response>...</response> envelope strips and tool call inside is extracted`() {
    var parser = ErnieParser()
    let input = """
    <response>
    I'll look that up.
    <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
    </response>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls.first?.name == "get_weather")
    // Message content includes the body text between markers but NOT
    // the `<response>` / `</response>` literals themselves.
    let allMessageText = messages.flatMap { m -> [String] in
      m.content.compactMap { part -> String? in
        if case let .outputText(t) = part { return t.text } else { return nil }
      }
    }.joined()
    #expect(!allMessageText.contains("<response>"))
    #expect(!allMessageText.contains("</response>"))
    #expect(allMessageText.contains("I'll look that up."))
  }

  @Test
  func `Plain text without any markers passes through as a message`() {
    var parser = ErnieParser()
    let events = parser.process(ParserInput(text: "Hello there!")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Hello there!")
  }
}

@Suite("ErnieParser — reasoning preamble (acceptThink: true)")
struct ErnieReasoningTests {
  @Test
  func `Reasoning before </think> emits reasoning item then tool call`() {
    var parser = ErnieParser(acceptThink: true)
    let input = """
    weighing options</think>

    <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "weighing options")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Reasoning before </think> followed by <response> envelope`() {
    var parser = ErnieParser(acceptThink: true)
    let input = """
    plan</think>
    <response>
    Sure thing.
    <tool_call>{"name": "f", "arguments": {}}</tool_call>
    </response>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoningTexts = items.compactMap { item -> String? in
      if case let .reasoning(r) = item, case let .reasoningText(t) = r.content[0] {
        return t.text
      }
      return nil
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoningTexts == ["plan"])
    #expect(toolCalls.count == 1)
    #expect(toolCalls.first?.name == "f")
  }

  @Test
  func `Reasoning can complete without response content`() {
    var parser = ErnieParser(acceptThink: true)
    let events = parser.process(ParserInput(text: "abc</think>")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "abc")
    #expect(r.status == .completed)
  }

  @Test
  func `Unclosed reasoning at EOS is surfaced as incomplete reasoning`() {
    var parser = ErnieParser(acceptThink: true)
    let events = parser.process(ParserInput(text: "abc")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "abc")
    #expect(r.status == .incomplete)
  }

  @Test
  func `Multiline reasoning and content split at think close`() {
    var parser = ErnieParser(acceptThink: true)
    let events = parser.process(ParserInput(text: "abc\nABC</think>def\nDEF")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "abc\nABC")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "def\nDEF")
  }

  @Test
  func `Default off: acceptThink not set means </think> leaks as content`() {
    // Verifies the V3-family default-off semantics.
    var parser = ErnieParser()
    let input = "weighing</think>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == input)
  }

  @Test
  func `InitialState .normal skips reasoning extraction (continuation request)`() {
    var parser = ErnieParser(acceptThink: true, initialState: .normal)
    let input = "Now back to work."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Now back to work.")
  }

  @Test
  func `InitialState .normal trims newline before response envelope`() {
    var parser = ErnieParser(acceptThink: true, initialState: .normal)
    let events = parser.process(ParserInput(text: "\n<response>\nDone.</response>")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Done.")
  }
}

@Suite("ErnieParser — streaming")
struct ErnieStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot for reasoning + tool call`() {
    let input = """
    plan</think>
    <tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>
    """

    var oneShot = ErnieParser(acceptThink: true)
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = ErnieParser(acceptThink: true)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    #expect(oneShotItems.count == 2)
    guard case let .reasoning(r1) = oneShotItems[0],
          case let .reasoning(r2) = streamedItems[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(r1.text == r2.text)
    guard case let .functionCall(f1) = oneShotItems[1],
          case let .functionCall(f2) = streamedItems[1]
    else {
      Issue.record("Expected function call"); return
    }
    #expect(f1.name == f2.name)
    #expect(f1.arguments == f2.arguments)
  }

  @Test
  func `Split <response> opener across chunks does not leak as content`() {
    var parser = ErnieParser()
    var events = parser.process(ParserInput(text: "<resp"))
    events += parser.process(ParserInput(
      text: #"onse>I'll do this.<tool_call>{"name": "f", "arguments": {}}</tool_call></response>"#,
    ))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    let allText = messages.flatMap { m -> [String] in
      m.content.compactMap { part -> String? in
        if case let .outputText(t) = part { return t.text } else { return nil }
      }
    }.joined()
    #expect(!allText.contains("<response>"))
    #expect(toolCalls.first?.name == "f")
  }

  @Test
  func `Fixed chunks preserve inner Hermes text and argument deltas`() {
    let chunks = [
      "<response>Intro ",
      "<tool_call>",
      #"{"name":"f","arguments":{"x":"#,
      "1",
      "}}</tool_call>",
      " tail</response>",
    ]
    var parser = ErnieParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let textDeltas = events.compactMap {
      if case let .outputTextDelta(e) = $0 { return e.delta }
      return nil
    }
    let argsDeltas = events.compactMap {
      if case let .functionCallArgumentsDelta(e) = $0 { return e.delta }
      return nil
    }
    #expect(textDeltas == ["Intro ", " tail"])
    #expect(argsDeltas == [#"{"x":"#, "1", "}"])
  }
}

@Suite("ErnieParser — format dispatch")
struct ErnieDispatchTests {
  @Test
  func `Name-prefix infer routes thinking model to .ernieThinking`() {
    let f = ResponseFormat.infer(
      modelName: "baidu/ERNIE-4.5-21B-A3B-Thinking",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .ernieThinking)
  }

  @Test
  func `Name-prefix infer routes non-thinking model to .ernie`() {
    let f = ResponseFormat.infer(
      modelName: "baidu/ERNIE-4.5-21B-A3B-PT",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .ernie)
  }

  @Test
  func `model_type ernie4_5 routes to .ernie`() {
    let f = ResponseFormat.resolveByType("ernie4_5", config: [:])
    #expect(f == .ernie)
  }

  @Test
  func `ERNIE Thinking factory extracts reasoning and response body`() {
    var parser = ResponseFormat.ernieThinking.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: nil,
    )
    let events = parser.process(ParserInput(text: "abc</think><response>def</response>"))
      + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "abc")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "def")
  }
}
