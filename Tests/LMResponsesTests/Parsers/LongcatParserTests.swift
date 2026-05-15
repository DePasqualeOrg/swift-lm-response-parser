// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

/// LongCat shares the Hermes parser implementation, parameterized on the
/// envelope tokens. Coverage focuses on:
/// 1. The envelope swap actually works end-to-end via the format factory.
/// 2. The default Hermes envelope tokens are *not* recognized when the
///    parser is configured for LongCat (so a stray `<tool_call>` in
///    LongCat output flows through as content rather than being mistaken
///    for an envelope).

@Suite("LongcatParser — envelope swap")
struct LongcatEnvelopeTests {
  @Test
  func `Single tool call wrapped in <longcat_tool_call> is extracted`() throws {
    let text = #"<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</longcat_tool_call>"#
    let parser = ResponseFormat.longcat.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: text)) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["city"] as? String == "Beijing")
  }

  @Test
  func `Plain Hermes <tool_call> is treated as content for the LongCat parser`() {
    let text = #"<tool_call>{"name": "f", "arguments": {}}</tool_call>"#
    var parser = HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty, "Hermes-style envelope must not match when configured for LongCat")
  }

  @Test
  func `Plain content surrounding a LongCat call streams correctly`() {
    let text = #"Sure.<longcat_tool_call>{"name": "f", "arguments": {"x": 1}}</longcat_tool_call>Done."#
    var parser = HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count >= 2)
    guard case let .message(first) = items[0], case let .outputText(part) = first.content[0]
    else { Issue.record(""); return }
    #expect(part.text == "Sure.")
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "f")
  }

  @Test
  func `Parallel LongCat calls preserve call order and arguments`() {
    // Mirrors vLLM's `test_longcat_tool_parser.py` parallel fixture.
    let text = (
      #"<longcat_tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</longcat_tool_call>"#
        + "\n"
        + #"<longcat_tool_call>{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}</longcat_tool_call>"#,
    )
    var parser = HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let toolCalls = accumulateItems(from: events).compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }

    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "get_weather")
    #expect(parseLongcatArgs(toolCalls[0].arguments)?["city"] as? String == "Tokyo")
    #expect(toolCalls[1].name == "get_time")
    #expect(parseLongcatArgs(toolCalls[1].arguments)?["timezone"] as? String == "Asia/Tokyo")
  }

  @Test
  func `LongCat mixed JSON argument values round-trip`() throws {
    // Condenses vLLM's common `various_data_types_output` and
    // `escaped_strings_output` fixtures for the LongCat parser.
    let text = """
    <longcat_tool_call>{
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
        "empty_object": {},
        "quoted": "He said \\"hello\\"",
        "path": "C:\\\\Users\\\\file.txt",
        "newline": "line1\\nline2"
      }
    }</longcat_tool_call>
    """

    var parser = HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(decoded["string_field"] as? String == "hello")
    #expect(decoded["int_field"] as? Int == 42)
    #expect(decoded["float_field"] as? Double == 3.14)
    #expect(decoded["bool_field"] as? Bool == true)
    #expect(decoded["null_field"] is NSNull)
    #expect((decoded["array_field"] as? [String]) == ["a", "b", "c"])
    #expect((decoded["object_field"] as? [String: String])?["nested"] == "value")
    #expect((decoded["empty_array"] as? [Any])?.isEmpty == true)
    #expect((decoded["empty_object"] as? [String: Any])?.isEmpty == true)
    #expect(decoded["quoted"] as? String == #"He said "hello""#)
    #expect(decoded["path"] as? String == #"C:\Users\file.txt"#)
    #expect(decoded["newline"] as? String == "line1\nline2")
  }

  @Test
  func `Empty LongCat arguments parse as an empty object`() throws {
    let text = #"<longcat_tool_call>{"name": "refresh", "arguments": {}}</longcat_tool_call>"#
    var parser = HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded.isEmpty)
  }

  @Test
  func `Char-by-char streaming through <longcat_tool_call> matches single-shot`() {
    let text = #"<longcat_tool_call>{"name": "f", "arguments": {"x": 1}}</longcat_tool_call>"#
    var single = HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
    let singleEvents = single.process(ParserInput(text: text)) + single.finalize()
    let singleItems = accumulateItems(from: singleEvents)

    var streamed = HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in text {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(singleItems.count == streamedItems.count)
    if case let .functionCall(s) = singleItems[0], case let .functionCall(c) = streamedItems[0] {
      #expect(s.name == c.name)
      #expect(s.arguments == c.arguments)
    } else {
      Issue.record("expected matching function call items")
    }
  }

  @Test
  func `Fixed chunks preserve exact LongCat deltas`() {
    let chunks = [
      "Intro ",
      "<longcat_tool_call>",
      #"{"name":"f","arguments":{"x":"#,
      "1",
      "}}",
      "</longcat_tool_call>",
      " outro",
    ]
    var parser = HermesParser(
      toolCallStart: "<longcat_tool_call>",
      toolCallEnd: "</longcat_tool_call>",
    )
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    #expect(longcatOutputTextDeltas(from: events) == ["Intro ", " outro"])
    #expect(longcatArgumentDeltas(from: events) == [#"{"x":"#, "1", "}"])
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes == [0, 1, 2])
  }
}

private func parseLongcatArgs(_ args: String) -> [String: Any]? {
  guard let data = args.data(using: .utf8) else { return nil }
  return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

private func longcatOutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap {
    if case let .outputTextDelta(e) = $0 { return e.delta }
    return nil
  }
}

private func longcatArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap {
    if case let .functionCallArgumentsDelta(e) = $0 { return e.delta }
    return nil
  }
}

@Suite("ResponseFormat dispatch — LongCat")
struct LongcatDispatchTests {
  @Test
  func `LongCat-Flash name routes to .longcat`() {
    let f = ResponseFormat.infer(
      modelName: "meituan-longcat/LongCat-Flash-Chat",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .longcat)
  }

  @Test
  func `Bare longcat prefix also routes to .longcat`() {
    let f = ResponseFormat.infer(
      modelName: "longcat-flash-fp8",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .longcat)
  }
}
