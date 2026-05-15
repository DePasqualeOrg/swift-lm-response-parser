// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// xLAM accepts a JSON array of `{name, arguments}` objects, optionally
// wrapped in any of four envelopes: `<tool_call>...</tool_call>`,
// `[TOOL_CALLS]...`, fenced ```` ```json ... ``` ````, or bare. A
// `<think>...</think>` reasoning block at the start is preserved as
// message content; tool-call detection runs on what follows.

@Suite("XlamParser — wrappers")
struct XlamWrapperTests {
  @Test
  func `Bare JSON array is recognized as tool calls`() throws {
    var parser = XlamParser()
    let input = #"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
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
  func `<tool_call>...</tool_call> envelope wraps a JSON array`() {
    var parser = XlamParser()
    let input = #"<tool_call>[{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {"y": 2}}]</tool_call>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "f")
    #expect(b.name == "g")
  }

  @Test
  func `[TOOL_CALLS] prefix recognized`() {
    var parser = XlamParser()
    let input = #"[TOOL_CALLS][{"name": "f", "arguments": {}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }

  @Test
  func `Fenced JSON code block recognized`() {
    var parser = XlamParser()
    let input = """
    ```json
    [{"name": "f", "arguments": {"x": 1}}]
    ```
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }

  @Test
  func `Plain text without any envelope passes through as message`() {
    var parser = XlamParser()
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
  func `Bare JSON array NOT at start (leading text) is treated as message`() {
    // vLLM only matches a bare array when the whole output strip-
    // starts with `[`. Leading non-bracket text disables the bare-
    // array detection; the entire output flows as message content.
    // Wrap in an explicit envelope (`<tool_call>`, `[TOOL_CALLS]`, or
    // a fenced block) to extract tool calls from such text.
    var parser = XlamParser()
    let input = #"Sure, calling: [{"name": "f", "arguments": {}}]"#
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
  func `Leading text before <tool_call> envelope produces a leading message`() {
    var parser = XlamParser()
    let input = #"Sure, calling: <tool_call>[{"name": "f", "arguments": {}}]</tool_call>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(t.text == "Sure, calling: ")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }

  @Test
  func `<think>...</think> reasoning block is preserved as content`() {
    var parser = XlamParser()
    let input = #"<think>plan</think>[{"name": "f", "arguments": {}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "<think>plan</think>")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }

  @Test
  func `Empty JSON array yields no tool calls`() {
    var parser = XlamParser()
    let events = parser.process(ParserInput(text: "[]")) + parser.finalize()
    let items = accumulateItems(from: events)
    // Empty array means "no tools" — no function call items.
    let functionCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(functionCalls.isEmpty)
  }

  @Test
  func `Malformed entries are skipped while valid calls emit`() {
    var parser = XlamParser()
    let input = #"[{"arguments": {"bad": true}}, {"name": "", "arguments": {}}, {"name": "f", "arguments": {"x": 1}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let functionCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(functionCalls.count == 1)
    #expect(functionCalls[0].name == "f")
  }

  @Test
  func `Argument key alias parameters is accepted`() throws {
    var parser = XlamParser()
    let input = #"[{"name": "f", "parameters": {"x": 1}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Truncated bare array at end of stream falls back to message`() {
    var parser = XlamParser()
    let events = parser.process(ParserInput(text: #"[{"name": "f", "arguments": {"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message fallback"); return }
  }

  @Test
  func `Later explicit envelope is recognized after an invalid first envelope`() {
    var parser = XlamParser()
    let input = #"before <tool_call>not json</tool_call> middle <tool_call>[{"name": "f", "arguments": {}}]</tool_call>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(t.text.contains("not json"))
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
  }
}

@Suite("XlamParser — streaming")
struct XlamStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot for bare array`() {
    let input = #"[{"name": "f", "arguments": {"x": 1}}]"#

    var oneShot = XlamParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = XlamParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    #expect(oneShotItems.count == 1)
    guard case let .functionCall(a) = oneShotItems[0],
          case let .functionCall(b) = streamedItems[0]
    else {
      Issue.record("Expected function calls"); return
    }
    #expect(a.name == b.name)
    #expect(a.arguments == b.arguments)
  }

  @Test
  func `Char-by-char reconstruction matches one-shot for <tool_call> envelope`() {
    let input = #"<tool_call>[{"name": "f", "arguments": {"x": 1}}]</tool_call>"#

    var oneShot = XlamParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = XlamParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    #expect(oneShotItems.count == 1)
  }
}

@Suite("XlamParser — format dispatch")
struct XlamDispatchTests {
  @Test
  func `Name-prefix infer routes Llama-xLAM to .xlam`() {
    let f = ResponseFormat.infer(
      modelName: "Salesforce/Llama-xLAM-2-8B-fc-r",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .xlam)
  }

  @Test
  func `Name-prefix infer routes Qwen-xLAM to .xlam`() {
    let f = ResponseFormat.infer(
      modelName: "Salesforce/Qwen-xLAM-32B-fc-r",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .xlam)
  }

  @Test
  func `Name-prefix infer routes bare xLAM to .xlam`() {
    let f = ResponseFormat.infer(
      modelName: "Salesforce/xLAM-1B-fc-r",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .xlam)
  }
}
