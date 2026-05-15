// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

@Suite("DeepSeekV3Parser — basics")
struct DeepSeekV3BasicsTests {
  @Test
  func `Plain text emits a single message`() {
    var parser = DeepSeekV3Parser()
    let events = parser.process(ParserInput(text: "hello")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Single tool call parses correctly`() throws {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"city": "Paris"}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekV3Parser()
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
  func `Whitespace around the function name is trimmed (matches sglang)`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜> get_weather \n```json
    {"city": "Paris"}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekV3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Multiple tool calls in one envelope`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>f1
    ```json
    {}
    ```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>f2
    ```json
    {"x": 1}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekV3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
  }

  @Test
  func `Char-by-char streaming yields the same items as single-shot`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>fn
    ```json
    {"x": 1}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """

    var streaming = DeepSeekV3Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = DeepSeekV3Parser()
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
          Issue.record("Item kinds differ"); return
      }
    }
  }

  @Test
  func `ResponseFormat.deepseekV3.makeParser dispatches correctly`() {
    let parser = ResponseFormat.deepseekV3.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Truncation between callBegin and name leaves no output_index gap`() {
    // Buffer truncates after callBegin but before the name appears.
    let input = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
    var parser = DeepSeekV3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    // No `output_item.added` for the abandoned call — so no slot is
    // consumed and there's no gap for downstream consumers.
    #expect(addedIndexes.isEmpty)
  }

  @Test
  func `Empty or whitespace-only function name drops the call`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>\u{20}\u{20}\u{20}
    ```json
    {}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekV3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.isEmpty)
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes.isEmpty)
  }

  @Test
  func `Empty-name call followed by valid call keeps output_index consecutive`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>\u{20}\u{20}\u{20}
    ```json
    {}
    ```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>fn
    ```json
    {"x": 1}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekV3Parser()
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
  func `Fixed chunks preserve exact text and argument deltas`() {
    let chunks = [
      "prefix ",
      "<｜tool",
      "▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>fn\n```json\n",
      #"{"x":"#,
      "1",
      "}",
      "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
      " suffix",
    ]

    var parser = DeepSeekV3Parser()
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
    #expect(deepSeekV3OutputTextDeltas(from: events) == ["prefix ", " suffix"])
    #expect(deepSeekV3ArgumentDeltas(from: events) == [#"{"x":"#, "1", "}"])
  }

  @Test
  func `Closed call followed by text and later call preserves exact deltas`() {
    let chunks = [
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>first\n```json\n{}"
        + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
      " gap ",
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>second\n```json\n{\"y\":2}"
        + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ]

    var parser = DeepSeekV3Parser()
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
    #expect(deepSeekV3OutputTextDeltas(from: events) == [" gap "])
    #expect(deepSeekV3ArgumentDeltas(from: events) == ["{}", #"{"y":2}"#])
  }
}

private func deepSeekV3OutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .outputTextDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

private func deepSeekV3ArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .functionCallArgumentsDelta(e) = event {
      return e.delta
    }
    return nil
  }
}
