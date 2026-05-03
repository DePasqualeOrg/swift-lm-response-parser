// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("DeepSeekV31Parser — basics")
struct DeepSeekV31BasicsTests {
  @Test
  func `Plain text emits a single message`() {
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Single tool call`() throws {
    let input = (
      "<｜tool▁calls▁begin｜>"
        + #"<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city":"Paris"}<｜tool▁call▁end｜>"#
        + "<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
  }

  @Test
  func `Multiple parallel tool calls`() {
    let input = (
      "<｜tool▁calls▁begin｜>"
        + #"<｜tool▁call▁begin｜>f1<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>"#
        + #"<｜tool▁call▁begin｜>f2<｜tool▁sep｜>{"y":2}<｜tool▁call▁end｜>"#
        + "<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record(""); return
    }
    #expect(a.name == "f1")
    #expect(b.name == "f2")
  }

  @Test
  func `Content text before tool call`() {
    let input = (
      "Let me check. "
        + "<｜tool▁calls▁begin｜>"
        + #"<｜tool▁call▁begin｜>f<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>"#
        + "<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record(""); return }
    guard case let .outputText(part) = m.content[0] else { Issue.record(""); return }
    #expect(part.text == "Let me check. ")
    guard case .functionCall = items[1] else { Issue.record(""); return }
  }
}

@Suite("DeepSeekV31Parser — streaming")
struct DeepSeekV31StreamingTests {
  @Test
  func `Char-by-char streaming reconstructs the same items as one-shot`() {
    let input = (
      "<｜tool▁calls▁begin｜>"
        + #"<｜tool▁call▁begin｜>fn<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>"#
        + "<｜tool▁calls▁end｜>",
    )

    var streaming = DeepSeekV31Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = DeepSeekV31Parser()
    let oneShotItems = accumulateItems(from:
      oneShot.process(ParserInput(text: input)) + oneShot.finalize())

    #expect(streamingItems.count == oneShotItems.count)
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.functionCall(sf), .functionCall(of)):
          #expect(sf.name == of.name)
          #expect(sf.arguments == of.arguments)
        case let (.message(sm), .message(om)):
          #expect(sm.content == om.content)
        default:
          Issue.record(""); return
      }
    }
  }

  @Test
  func `Fixed chunks preserve exact text and argument deltas`() {
    let chunks = [
      "prefix ",
      "<｜tool",
      "▁calls▁begin｜><｜tool▁call▁begin｜>fn<｜tool▁sep｜>",
      #"{"x":"#,
      "1",
      "}",
      "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
      " suffix",
    ]

    var parser = DeepSeekV31Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let outputIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(outputIndexes == [0, 1, 2])
    #expect(deepSeekV31OutputTextDeltas(from: events) == ["prefix ", " suffix"])
    #expect(deepSeekV31ArgumentDeltas(from: events) == [#"{"x":"#, "1", "}"])
  }

  @Test
  func `Closed call followed by text and later call preserves exact deltas`() {
    let chunks = [
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>first<｜tool▁sep｜>{}<｜tool▁call▁end｜>"
        + "<｜tool▁calls▁end｜>",
      " gap ",
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>second<｜tool▁sep｜>{\"y\":2}<｜tool▁call▁end｜>"
        + "<｜tool▁calls▁end｜>",
    ]

    var parser = DeepSeekV31Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let outputIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(outputIndexes == [0, 1, 2])
    #expect(deepSeekV31OutputTextDeltas(from: events) == [" gap "])
    #expect(deepSeekV31ArgumentDeltas(from: events) == ["{}", #"{"y":2}"#])
  }
}

private func deepSeekV31OutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .outputTextDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

private func deepSeekV31ArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .functionCallArgumentsDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

@Suite("DeepSeekV31Parser — finalize and dispatch")
struct DeepSeekV31FinalizeTests {
  @Test
  func `Truncated tool call closes as incomplete`() {
    let input = (
      "<｜tool▁calls▁begin｜>"
        + #"<｜tool▁call▁begin｜>fn<｜tool▁sep｜>{"x":"#,
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.status == .incomplete)
    #expect(f.name == "fn")
  }

  @Test
  func `Dispatch via ResponseFormat.deepseekV31.makeParser`() {
    let parser = ResponseFormat.deepseekV31.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Name prefix deepseek-v3.1 routes to .deepseekV31`() {
    let format = ResponseFormat.infer(modelName: "deepseek-v3.1-base", modelType: "", modelConfig: [:])
    #expect(format == .deepseekV31)
  }

  @Test
  func `V3.2-Exp also routes to .deepseekV31 (shares the V3.1 wire format)`() {
    let format = ResponseFormat.infer(modelName: "deepseek-v3.2-exp", modelType: "", modelConfig: [:])
    #expect(format == .deepseekV31)
  }
}

@Suite("DeepSeekV31Parser — malformed envelopes")
struct DeepSeekV31MalformedTests {
  // sglang's V3.1 detector emits `name=""` when the envelope is empty.
  // We diverge defensively: drop the call rather than surface a nameless
  // function_call to consumers. The output_index is recycled so the
  // following item lands at a consecutive slot.
  @Test
  func `Empty name (<call_begin><sep>...<call_end>) drops the call`() {
    let input = (
      "<｜tool▁calls▁begin｜>"
        + "<｜tool▁call▁begin｜><｜tool▁sep｜>{}<｜tool▁call▁end｜>"
        + "<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.isEmpty, "Empty-name envelope should not produce a function call")
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes.isEmpty)
  }

  @Test
  func `Whitespace-only name (<call_begin>   <sep>...) drops the call`() {
    let input = (
      "<｜tool▁calls▁begin｜>"
        + "<｜tool▁call▁begin｜>   <｜tool▁sep｜>{}<｜tool▁call▁end｜>"
        + "<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.isEmpty)
  }

  @Test
  func `Empty-name call followed by valid call: indexes stay consecutive`() {
    let input = (
      "<｜tool▁calls▁begin｜>"
        + "<｜tool▁call▁begin｜><｜tool▁sep｜>{}<｜tool▁call▁end｜>"
        + "<｜tool▁call▁begin｜>fn<｜tool▁sep｜>{\"x\":1}<｜tool▁call▁end｜>"
        + "<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes == [0])
  }

  @Test
  func `Truncation between <｜tool▁call▁begin｜> and <｜tool▁sep｜> emits no events`() {
    // Stream stops after the open marker, before the separator.
    // Lazy outputIndex allocation should prevent an output_item.added
    // from being emitted (and prevent a slot from being burned).
    let input = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.isEmpty)
    let added = events.compactMap { event -> ResponseStreamingEvent? in
      if case .outputItemAdded = event { return event } else { return nil }
    }
    #expect(added.isEmpty)
  }

  @Test
  func `Truncated header before sep keeps output_index consecutive for next item`() {
    // After a truncated header, a subsequent valid call should land
    // at output_index 0. Pins the lazy-allocation invariant for V31.
    var parser = DeepSeekV31Parser()
    _ = parser.process(ParserInput(text: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"))
    let later = "fn<｜tool▁sep｜>{\"x\":1}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    let events = parser.process(ParserInput(text: later)) + parser.finalize()
    let added = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex } else { return nil }
    }
    #expect(added == [0])
  }
}
