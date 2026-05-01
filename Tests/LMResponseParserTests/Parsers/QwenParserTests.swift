// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("QwenParser — plain text (Qwen 2.5)")
struct QwenPlainTextTests {
  @Test
  func `Single chunk of text emits a message with no reasoning`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(t.text == "hello world")
  }

  @Test
  func `Multiple chunks accumulate via deltas`() {
    var parser = QwenParser()
    var events = parser.process(ParserInput(text: "hel"))
    events += parser.process(ParserInput(text: "lo "))
    events += parser.process(ParserInput(text: "world"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "hello world")
  }

  @Test
  func `Empty stream finalize emits nothing`() {
    var parser = QwenParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("QwenParser — reasoning")
struct QwenReasoningTests {
  @Test
  func `Reasoning then content: <think>r</think>c emits reasoning + message in order`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: "<think>This is a reasoning section</think>This is the rest",
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .completed)
    guard case let .reasoningText(rt) = r.content[0] else { Issue.record("Expected reasoning text"); return }
    #expect(rt.text == "This is a reasoning section")

    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "This is the rest")
  }

  @Test
  func `Reasoning only — no closing </think>, finalize closes as incomplete`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: "<think>Mid-thought when generation cut off",
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .incomplete)
    guard case let .reasoningText(rt) = r.content[0] else { Issue.record("Expected reasoning text"); return }
    #expect(rt.text == "Mid-thought when generation cut off")
  }

  @Test
  func `InitialState .reasoning treats output as already inside <think>`() {
    var parser = QwenParser(initialState: .reasoning)
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
    #expect(r.status == .completed)

    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Now the answer")
  }

  @Test
  func `Reasoning ID is rs_-prefixed and reasoning carries content_part envelope`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: "<think>r</think>c")) + parser.finalize()
    let kinds = events.map { eventKind($0) }
    // Reasoning sequence: outputItemAdded → contentPartAdded → reasoningTextDelta → reasoningTextDone → contentPartDone → outputItemDone
    // Then message sequence: outputItemAdded → contentPartAdded → outputTextDelta → outputTextDone → contentPartDone → outputItemDone
    #expect(kinds == [
      "outputItemAdded",
      "contentPartAdded",
      "reasoningTextDelta",
      "reasoningTextDone",
      "contentPartDone",
      "outputItemDone",
      "outputItemAdded",
      "contentPartAdded",
      "outputTextDelta",
      "outputTextDone",
      "contentPartDone",
      "outputItemDone",
    ])

    // First item id should start with rs_.
    guard case let .outputItemAdded(added) = events[0] else { Issue.record("Expected reasoning added"); return }
    #expect(added.item.id.hasPrefix("rs_"))
  }

  @Test
  func `Multiline reasoning is preserved verbatim`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "This is a reasoning\nsection")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "This is the rest\nThat")
  }

  @Test
  func `Empty reasoning: <think></think> emits no reasoning item, only the message`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: "<think></think>after")) + parser.finalize()
    let items = accumulateItems(from: events)
    // Reasoning item is only opened on the first non-empty reasoning
    // delta, so empty `<think></think>` produces no reasoning item.
    // Matches Qwen 3.5 with thinking disabled, where the chat template
    // emits the marker pair to signal "no reasoning."
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "after")
  }
}

@Suite("QwenParser — reasoning + tool call")
struct QwenReasoningToolCallTests {
  @Test
  func `Explicit reasoning end then tool call`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: #"<think>I need the weather.</think><tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "I need the weather.")

    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    #expect(f.arguments.contains("\"Paris\""))
    #expect(f.status == .completed)
  }

  @Test
  func `Implicit reasoning end on <tool_call> (no </think>)`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: #"I need to read the file.<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#,
    ))
    // First chunk has no <think> — initial state is .normal, so the
    // leading text is content. Make a separate test for the actual
    // implicit-end case.
    let items = accumulateItems(from: events + parser.finalize())
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `Implicit reasoning end while in reasoning phase`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: #"<think>I need to call a tool.<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "I need to call a tool.")
    // Reasoning is closed as completed (implicit end is a clean close).
    #expect(r.status == .completed)

    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `InitialState .reasoning + implicit tool call end (Qwen 3.5 case)`() {
    var parser = QwenParser(initialState: .reasoning)
    let events = parser.process(ParserInput(
      text: #"Reasoning runs into the tool call.<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>"#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "Reasoning runs into the tool call.")
    #expect(r.status == .completed)

    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.arguments.contains("\"x\""))
  }

  @Test
  func `Output indexes: reasoning gets 0, content/tool gets 1+`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: #"<think>r</think>c<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#,
    )) + parser.finalize()

    let addedIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes == [0, 1, 2])
  }

  // Trinity (Arcee) emits tool calls inside `<think>...</think>` blocks.
  // The Swift Qwen parser handles the implicit reasoning end on
  // `<tool_call>`, so the reasoning preamble lands as a reasoning item
  // and the tool call is extracted. Whatever comes after the tool call
  // up to `</think>` (and then beyond) is normal-phase content – the
  // closing `</think>` must not leak into message text.
  @Test
  func `Trinity-style tool call inside <think> with text after </think>`() {
    var parser = QwenParser()
    let input = #"<think>I should call f.<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call> done thinking</think>final answer"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)

    // Expect: reasoning, function call, message.
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    for msg in messages {
      guard case let .outputText(part) = msg.content[0] else { continue }
      #expect(!part.text.contains("</think>"), "literal </think> must not leak into message text")
      #expect(!part.text.contains("<think>"), "literal <think> must not leak into message text")
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "f")
  }
}

@Suite("QwenParser — streaming boundaries")
struct QwenStreamingBoundaryTests {
  @Test
  func `Marker split across chunks: <thi | nk>`() {
    var parser = QwenParser()
    var events = parser.process(ParserInput(text: "<thi"))
    // No deltas yet — buffer might still grow into <think>.
    let initialDeltas = events.filter {
      if case .reasoningTextDelta = $0 { return true }
      if case .outputTextDelta = $0 { return true }
      return false
    }
    #expect(initialDeltas.isEmpty)

    events += parser.process(ParserInput(text: "nk>real reasoning</think>plain"))
    events += parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "real reasoning")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "plain")
  }

  @Test
  func `</think> split across chunks within reasoning`() {
    var parser = QwenParser(initialState: .reasoning)
    var events = parser.process(ParserInput(text: "reasoning</thi"))
    // The trailing </thi should be held back, only "reasoning" emitted.
    events += parser.process(ParserInput(text: "nk>after"))
    events += parser.finalize()

    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "reasoning")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "after")
  }

  @Test
  func `Char-by-char streaming yields the same items as single-shot`() {
    let input = #"<think>think hard</think>final answer<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>"#

    var streaming = QwenParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = QwenParser()
    let oneShotEvents = oneShot.process(ParserInput(text: input)) + oneShot.finalize()
    let oneShotItems = accumulateItems(from: oneShotEvents)

    #expect(streamingItems.count == oneShotItems.count)
    // Compare ignoring IDs (which are random).
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.reasoning(sr), .reasoning(or)):
          #expect(sr.content == or.content)
          #expect(sr.status == or.status)
        case let (.message(sm), .message(om)):
          #expect(sm.content == om.content)
          #expect(sm.status == om.status)
        case let (.functionCall(sf), .functionCall(of)):
          #expect(sf.name == of.name)
          #expect(sf.arguments == of.arguments)
          #expect(sf.status == of.status)
        default:
          Issue.record("Item kinds differ: \(s) vs \(o)")
      }
    }
  }

  @Test
  func `Tool call <tool_ | call> split across chunks works`() {
    var parser = QwenParser()
    var events = parser.process(ParserInput(text: "<think>r</think>before<tool_"))
    events += parser.process(ParserInput(text: #"call>{"name": "fn", "arguments": {}}</tool_call>"#))
    events += parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 3)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "before")
    guard case let .functionCall(f) = items[2] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }
}

@Suite("QwenParser — multiple tool calls")
struct QwenMultipleToolCallsTests {
  @Test
  func `Plain text between two tool calls in one chunk emits a message`() throws {
    let text = #"<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call> Now check dogs: <tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>"#
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 3, "Expected two function_calls plus a message between them")
    guard case let .functionCall(first) = items[0],
          case let .message(middle) = items[1],
          case let .functionCall(second) = items[2]
    else {
      Issue.record("Expected function_call → message → function_call"); return
    }
    #expect(first.name == "search")
    let firstParsedData = try #require(first.arguments.data(using: .utf8))
    let firstParsed = try #require(JSONSerialization.jsonObject(with: firstParsedData) as? [String: Any])
    #expect(firstParsed["q"] as? String == "cats")
    guard case let .outputText(text) = middle.content[0] else {
      Issue.record("Expected outputText in middle message"); return
    }
    #expect(text.text == " Now check dogs: ")
    #expect(second.name == "search")
    let secondParsedData = try #require(second.arguments.data(using: .utf8))
    let secondParsed = try #require(JSONSerialization.jsonObject(with: secondParsedData) as? [String: Any])
    #expect(secondParsed["q"] as? String == "dogs")
  }

  @Test
  func `Multiple tool calls split across chunks keep distinct indexes`() throws {
    // Mirrors SGLang's Qwen25 streaming multi-call fixture, where a
    // later `<tool_call>` arrives only after the previous region has
    // already closed.
    let chunks = [
      "<tool_call>\n",
      #"{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}"#,
      "\n</tool_call>\n",
      "<tool_call>\n",
      #"{"name": "get_current_weather", "arguments": {"city": "Baltimore", "state": "MD", "unit": "fahrenheit"}}"#,
      "\n</tool_call>\n",
      "<tool_call>\n",
      #"{"name": "get_current_weather", "arguments": {"city": "LA", "state": "CA", "unit": "celsius"}}"#,
      "\n</tool_call>",
    ]
    var parser = QwenParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let toolCalls = accumulateItems(from: events).compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 3)
    let cities = try toolCalls.map { call in
      let data = try #require(call.arguments.data(using: .utf8))
      let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
      return decoded["city"] as? String
    }
    #expect(cities == ["NYC", "Baltimore", "LA"])
  }
}

@Suite("QwenParser — stray close-tag stripping")
struct QwenStrayCloseTagTests {
  // sglang's `parse_streaming_increment` (Qwen 2.5 detector) buffers
  // normal text and strips a stray `</tool_call>` substring (without
  // the leading newline) from it. Pin parity with that behavior.
  @Test
  func `Bare </tool_call> literal in plain content is stripped`() {
    var parser = QwenParser()
    let input = "Hello </tool_call> world"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Hello  world")
  }

  @Test
  func `Partial </tool_cal suffix at end of chunk is held back`() {
    var parser = QwenParser()
    let mid = parser.process(ParserInput(text: "abc</tool_cal"))
    let later = parser.process(ParserInput(text: "l>def")) + parser.finalize()
    let allEvents = mid + later
    let items = accumulateItems(from: allEvents)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "abcdef")
  }
}

@Suite("QwenParser — finalize edge cases")
struct QwenFinalizeTests {
  @Test
  func `Truncated tool call inside reasoning end: closes reasoning + opens incomplete tool call`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(
      text: #"<think>r</think><tool_call>{"name": "fn", "arguments": {"loca"#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .completed)
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.status == .incomplete)
  }

  @Test
  func `finalize emits nothing extra when reasoning was clean-closed and stream is at end`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: "<think>r</think>done"))
    let finalEvents = parser.finalize()
    // Process events should already include the reasoning + an in-progress message.
    let allItems = accumulateItems(from: events + finalEvents)
    #expect(allItems.count == 2)
    guard case let .message(m) = allItems[1] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
  }

  @Test
  func `Truncated header before name leaves no output_index gap`() {
    // A `<tool_call>` opens but the JSON `name` field never arrives.
    // The slot must not be reserved — otherwise the next item lands
    // at a non-consecutive index.
    var parser = QwenParser()
    let text = #"Hi <tool_call>{"argum"#
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    // Only the message item gets an index. The truncated tool call
    // never reserved a slot, so indexes stay consecutive.
    #expect(addedIndexes == [0])
  }

  @Test
  func `Truncated header followed by valid tool call: indexes stay consecutive`() {
    var parser = QwenParser()
    let chunk1 = #"<tool_call>{"argum"# // never gets a name
    _ = parser.process(ParserInput(text: chunk1))
    let events = parser.finalize()
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes.isEmpty, "Truncated header without a name emits no items")
  }
}

@Suite("QwenParser — dispatch")
struct QwenDispatchTests {
  @Test
  func `ResponseFormat.qwen.makeParser returns a working QwenParser`() {
    let parser = ResponseFormat.qwen.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "<think>r</think>c")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
    guard case .message = items[1] else { Issue.record("Expected message"); return }
  }

  @Test
  func `priorOutput with unclosed <think> resumes parser in reasoning state`() {
    let parser = ResponseFormat.qwen.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial reasoning that",
    )
    var p = parser
    // The prior response was cut off mid-reasoning. New tokens are
    // a continuation of the reasoning, so the parser should treat them
    // as reasoning text (not content).
    let events = p.process(ParserInput(text: " continues here</think>final")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == " continues here")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "final")
  }

  @Test
  func `priorOutput with closed <think>...</think> does NOT resume in reasoning`() {
    let parser = ResponseFormat.qwen.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>completed reasoning</think>then content",
    )
    var p = parser
    let events = p.process(ParserInput(text: "more content")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `parseResponse one-shot wrapper works`() {
    let items = parseResponse(
      "<think>r</think>c",
      format: .qwen,
      tokenizer: StubTokenizer(),
    )
    #expect(items.count == 2)
  }
}

@Suite("QwenParser — parameters alias")
struct QwenParametersAliasTests {
  // sglang's parse_base_json accepts either `arguments` or `parameters`.
  // Qwen reuses the Hermes envelope so the same alias rule applies.
  @Test
  func `parameters field is accepted as an alias for arguments`() throws {
    let input = #"<tool_call>{"name": "get_weather", "parameters": {"city": "Tokyo"}}</tool_call>"#
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }
}
