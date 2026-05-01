// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("DeepSeekR1Parser — plain text")
struct DeepSeekR1PlainTextTests {
  @Test
  func `Plain text without <think> is treated as reasoning content (R1 base)`() {
    // R1 base emits reasoning content directly without a `<think>`
    // opener. The default initial state is `.reasoning`, mirroring
    // SGLang's `force_reasoning=True`. With no `</think>` ever
    // arriving the reasoning closes as `.incomplete` on finalize.
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "hello world")
    #expect(r.status == .incomplete)
  }

  @Test
  func `Plain text with .normal initial state still emits a message (opt-out path)`() {
    var parser = DeepSeekR1Parser(initialState: .normal)
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "hello world")
  }

  @Test
  func `Empty stream finalize emits nothing`() {
    var parser = DeepSeekR1Parser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("DeepSeekR1Parser — reasoning")
struct DeepSeekR1ReasoningTests {
  @Test
  func `<think>r</think>c emits reasoning + message`() {
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(
      text: "<think>I need to think about this.</think>The answer is 42.",
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "I need to think about this.")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "The answer is 42.")
  }

  @Test
  func `InitialState .reasoning treats output as already inside <think>`() {
    var parser = DeepSeekR1Parser(initialState: .reasoning)
    let events = parser.process(ParserInput(
      text: "Pre-loaded reasoning</think>Now the answer",
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "Pre-loaded reasoning")
  }

  @Test
  func `Truncated reasoning closes as incomplete`() {
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: "<think>cut off mid-thought")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .incomplete)
  }

  @Test
  func `R1 base (no <think> opener) emits reasoning then message at </think>`() {
    // Original DeepSeek-R1 does not emit a `<think>` start tag — it
    // begins directly in reasoning content. The factory's force-
    // reasoning default treats this correctly.
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(
      text: "I need to think about this.</think>The answer is 42.",
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "I need to think about this.")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "The answer is 42.")
  }

  @Test
  func `R1 base streams char-by-char without <think> and yields the same items as one-shot`() {
    let input = "Reasoning before answer.</think>The answer."
    var streaming = DeepSeekR1Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = DeepSeekR1Parser()
    let oneShotItems = accumulateItems(from:
      oneShot.process(ParserInput(text: input)) + oneShot.finalize())

    #expect(streamingItems.count == oneShotItems.count)
    #expect(streamingItems.count == 2)
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.reasoning(sr), .reasoning(or)):
          #expect(sr.content == or.content)
          #expect(sr.status == or.status)
        case let (.message(sm), .message(om)):
          #expect(sm.content == om.content)
        default:
          Issue.record("Item kinds differ"); return
      }
    }
  }
}

@Suite("DeepSeekR1Parser — tool calls")
struct DeepSeekR1ToolCallTests {
  static let singleToolCall = """
  <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
  ```json
  {"location": "Tokyo"}
  ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
  """

  @Test
  func `Single tool call parses out name + arguments`() throws {
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: Self.singleToolCall)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    #expect(f.status == .completed)
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["location"] as? String == "Tokyo")
  }

  @Test
  func `Function-call event sequence: added → arguments.delta → arguments.done → output_item.done`() {
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: Self.singleToolCall)) + parser.finalize()
    let kinds = events.map { eventKind($0) }
    // outputItemAdded, then 1+ arguments.delta, then arguments.done, then output_item.done
    #expect(kinds.first == "outputItemAdded")
    #expect(kinds.last == "outputItemDone")
    #expect(kinds.contains("functionCallArgumentsDelta"))
    #expect(kinds.contains("functionCallArgumentsDone"))
    // No content_part envelope around function call.
    #expect(!kinds.contains("contentPartAdded"))
  }

  @Test
  func `Multiple tool calls in one envelope each emit a function_call item`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"location": "Tokyo"}
    ```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time
    ```json
    {"timezone": "Asia/Tokyo"}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "get_weather")
    #expect(b.name == "get_time")
    #expect(a.id != b.id)
  }

  @Test
  func `Function name is stripped of surrounding whitespace`() {
    // Mirrors sglang's deepseekv3_detector.py which calls .strip() on
    // the captured name. Some templates introduce stray whitespace
    // inside `<｜tool▁sep｜> name \n`.
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>  get_weather
    ```json
    {"location": "Tokyo"}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Tool call IDs are distinct fc_/call_ pairs`() {
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: Self.singleToolCall)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
    #expect(f.id != f.callId)
  }
}

@Suite("DeepSeekR1Parser — reasoning + tool call")
struct DeepSeekR1ReasoningPlusToolCallTests {
  @Test
  func `Reasoning then tool call: explicit </think> then envelope`() {
    let input = """
    <think>Need weather data.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"location": "Tokyo"}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Implicit reasoning end on <｜tool▁calls▁begin｜> when no </think>`() {
    let input = """
    <think>Need to call a tool.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>fn
    ```json
    {}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "Need to call a tool.")
    #expect(r.status == .completed)
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }
}

@Suite("DeepSeekR1Parser — streaming")
struct DeepSeekR1StreamingTests {
  @Test
  func `Char-by-char streaming yields the same items as single-shot`() {
    let input = """
    <think>r</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>fn
    ```json
    {"x": 1}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """

    var streaming = DeepSeekR1Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = DeepSeekR1Parser()
    let oneShotItems = accumulateItems(from:
      oneShot.process(ParserInput(text: input)) + oneShot.finalize())

    #expect(streamingItems.count == oneShotItems.count)
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.reasoning(sr), .reasoning(or)):
          #expect(sr.content == or.content)
          #expect(sr.status == or.status)
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
}

@Suite("DeepSeekR1Parser — finalize edge cases")
struct DeepSeekR1FinalizeTests {
  @Test
  func `Truncated tool call before fence-close emits incomplete call`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>fn
    ```json
    {"x": "par
    """
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.status == .incomplete)
  }

  @Test
  func `Truncation between <｜tool▁call▁begin｜> and <｜tool▁sep｜> emits no events`() {
    // The model emits the open marker and then stops. With lazy
    // outputIndex allocation, this should leave no dangling slot —
    // no item events, no consumed output_index.
    let input = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.isEmpty)
    // No output_item.added should have been emitted for this truncated header.
    let added = events.compactMap { event -> ResponseStreamingEvent? in
      if case .outputItemAdded = event { return event } else { return nil }
    }
    #expect(added.isEmpty)
  }

  @Test
  func `Truncated header before sep keeps output_index consecutive for next item`() {
    // After a truncated header, a *subsequent* call that completes
    // properly should land at output_index 0 (not 1). This pins the
    // lazy-allocation invariant.
    var parser = DeepSeekR1Parser()
    // First chunk: open marker only — allocates nothing.
    _ = parser.process(ParserInput(text: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"))
    // Subsequent chunks complete a different call. The new call's
    // output_index must be 0 (the slot was never burned).
    let later = "function<｜tool▁sep｜>fn\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    let events = parser.process(ParserInput(text: later)) + parser.finalize()
    let added = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex } else { return nil }
    }
    #expect(added == [0])
  }

  @Test
  func `Empty or whitespace-only function name drops the call`() {
    let input = """
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>\u{20}\u{20}\u{20}
    ```json
    {}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    var parser = DeepSeekR1Parser(initialState: .normal)
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
    var parser = DeepSeekR1Parser(initialState: .normal)
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
}

@Suite("DeepSeekR1Parser — dispatch")
struct DeepSeekR1DispatchTests {
  @Test
  func `ResponseFormat.deepseekR1.makeParser returns a working parser`() {
    let parser = ResponseFormat.deepseekR1.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "<think>r</think>c")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
  }

  @Test
  func `priorOutput with unclosed <think> resumes parser in reasoning state`() {
    let parser = ResponseFormat.deepseekR1.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial",
    )
    var p = parser
    let events = p.process(ParserInput(text: " continues</think>after")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
  }
}

@Suite("DeepSeekR1Parser — leading whitespace before <think>")
struct DeepSeekR1LeadingWhitespaceTests {
  @Test
  func `Leading whitespace then <think> routes to reasoning, not a message`() {
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: "  \n<think>thinking</think>final")) + parser.finalize()
    let items = accumulateItems(from: events)
    // First item should be reasoning (not a message holding pre-`<think>` whitespace).
    guard case let .reasoning(r) = items[0] else {
      Issue.record("Expected reasoning first; got \(items)"); return
    }
    guard case let .reasoningText(part) = r.content[0] else {
      Issue.record("Expected reasoningText"); return
    }
    #expect(part.text == "thinking")
    guard case let .message(m) = items[1] else {
      Issue.record("Expected message second"); return
    }
    guard case let .outputText(t) = m.content[0] else {
      Issue.record("Expected outputText"); return
    }
    #expect(t.text == "final")
  }
}
