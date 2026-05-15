// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// LFM2 reuses PythonicParser with custom delimiters. These tests pin the
// behaviors that matter for LFM2 specifically: the alternate wrapper
// tokens, partial-marker hold-back across streaming chunks, and
// reconstruction equivalence between one-shot and char-by-char delivery.

private func makeLfm2Parser() -> PythonicParser {
  PythonicParser(
    startTag: "<|tool_call_start|>",
    endTag: "<|tool_call_end|>",
    acceptJSON: true,
    requiresWrapper: true,
    acceptBarePythonicCall: true,
  )
}

@Suite("Lfm2 — wrapper tokens")
struct Lfm2WrapperTokenTests {
  @Test
  func `Single call wrapped in <|tool_call_start|>...<|tool_call_end|>`() throws {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>[calculator(expression="5 * 7")]<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "calculator")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["expression"] as? String == "5 * 7")
  }

  @Test
  func `Parallel calls inside a single LFM2 envelope`() {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>[get_weather(city="Rome"), search(query="test")]<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "get_weather")
    #expect(b.name == "search")
  }

  @Test
  func `Llama-4 wrapper tokens are NOT stripped when LFM2 delimiters are configured`() {
    var parser = makeLfm2Parser()
    let input = "<|python_start|>plain text<|python_end|>"
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
  func `Leading text before LFM2 envelope emits as message`() {
    var parser = makeLfm2Parser()
    let input = #"Sure, calling: <|tool_call_start|>[fn()]<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected leading message"); return }
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected text content"); return }
    #expect(t.text == "Sure, calling: ")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `Bare bracket list without LFM2 envelope is plain content`() {
    var parser = makeLfm2Parser()
    let input = #"[get_weather(city="Paris")]"#
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
  func `Single bare Pythonic call inside envelope is accepted`() throws {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>get_weather(city="Paris")<|tool_call_end|>"#
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
  func `Tuple argument inside Pythonic envelope serializes as a JSON array`() throws {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>plot(point=(3, 4))<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "plot")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["point"] as? [Int] == [3, 4])
  }
}

@Suite("Lfm2 — streaming reconstruction")
struct Lfm2StreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = #"<|tool_call_start|>[get_weather(city="Rome")]<|tool_call_end|>"#

    var oneShot = makeLfm2Parser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = makeLfm2Parser()
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
  func `Split <|tool_call_start|> marker across chunks doesn't leak as content`() {
    // Split the opener two characters in. The buffer must hold the
    // partial prefix until the next chunk completes the marker; if
    // hold-back fails the partial bytes leak into a message item.
    var parser = makeLfm2Parser()
    var events = parser.process(ParserInput(text: "<|"))
    events += parser.process(ParserInput(text: "tool_call_start|>[fn()]<|tool_call_end|>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `Split <|tool_call_end|> marker across chunks doesn't leak`() {
    var parser = makeLfm2Parser()
    var events = parser.process(ParserInput(text: "<|tool_call_start|>[fn()]<|tool_"))
    events += parser.process(ParserInput(text: "call_end|>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `Complete Pythonic list waits for end marker before emitting`() {
    var parser = makeLfm2Parser()
    let firstEvents = parser.process(ParserInput(text: #"<|tool_call_start|>[get_weather(city="Berlin")]"#))
    #expect(accumulateItems(from: firstEvents).isEmpty)

    let events = firstEvents
      + parser.process(ParserInput(text: "<|tool_call_end|>"))
      + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }
}

// LFM2's `_parse_tool_calls_content` accepts both Pythonic and JSON
// inside the `<|tool_call_start|>` / `<|tool_call_end|>` envelope. The
// JSON shape can be a list of objects or a single object. These tests
// pin parity with sglang's `Lfm2Detector` for both shapes.

@Suite("Lfm2 — JSON wire shape")
struct Lfm2JsonShapeTests {
  @Test
  func `Single JSON object inside the envelope`() throws {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Vienna"}}<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Vienna")
  }

  @Test
  func `List of JSON objects inside the envelope`() {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "search", "arguments": {"query": "hotels"}}]<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "get_weather")
    #expect(b.name == "search")
  }

  @Test
  func `JSON shape streamed character-by-character matches one-shot`() {
    let input = #"<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Munich"}}<|tool_call_end|>"#

    var oneShot = makeLfm2Parser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = makeLfm2Parser()
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
  func `parameters alias is accepted as an arguments synonym`() throws {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>{"name": "get_weather", "parameters": {"city": "Oslo"}}<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Oslo")
  }

  @Test
  func `Leading text before a JSON envelope emits as a message`() {
    var parser = makeLfm2Parser()
    let input = #"I'll check the weather. <|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Amsterdam"}}<|tool_call_end|>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected leading message"); return }
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected text content"); return }
    #expect(t.text.contains("check the weather"))
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }
}

@Suite("Lfm2 — malformed input")
struct Lfm2MalformedInputTests {
  @Test
  func `Truncated mid-call (missing close marker and bracket) emits no tool call`() {
    var parser = makeLfm2Parser()
    let input = #"<|tool_call_start|>[get_weather(city="Rome""#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    // PythonicParser holds partial bracket content until close. At
    // finalize, unclosed brackets surface as plain content (not as a
    // function call), per the existing PythonicParser contract.
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Empty bracket list emits no tool call and no spurious content`() {
    var parser = makeLfm2Parser()
    let input = "<|tool_call_start|>[]<|tool_call_end|>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}
