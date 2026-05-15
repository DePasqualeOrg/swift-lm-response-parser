// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

@Suite("HarmonyParser — basics")
struct HarmonyBasicsTests {
  @Test
  func `Plain text outside any block emits a no-phase message`() {
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: "hello there")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == nil)
    guard case let .outputText(part) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part.text == "hello there")
  }

  @Test
  func `Single analysis block produces a reasoning item`() {
    let input = "<|channel|>analysis<|message|>Let me think about this<|end|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .completed)
    guard case let .reasoningText(part) = r.content[0] else { Issue.record("Expected reasoningText"); return }
    #expect(part.text == "Let me think about this")
  }

  @Test
  func `Single commentary block produces a commentary-phase message`() {
    let input = "<|channel|>commentary<|message|>User-visible preamble<|end|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == .commentary)
    guard case let .outputText(part) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part.text == "User-visible preamble")
  }

  @Test
  func `Single final block produces a final-answer message`() {
    let input = "<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == .finalAnswer)
    guard case let .outputText(part) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part.text == "The answer is 42")
  }

  @Test
  func `Tool call on commentary channel with functions. prefix`() throws {
    let input = #"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}<|call|>"#
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    #expect(f.status == .completed)
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
  }

  @Test
  func `Tool recipient before channel marker is carried into the channel block`() throws {
    let input = #"<|start|>assistant to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{"location":"Tokyo"}<|end|>"#
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
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
  func `Empty function name (to=functions.) routes as plain commentary, not a function call`() {
    // Adversarial input: trailing dot with no function name. We strip
    // the `functions.` prefix and reject the empty result rather than
    // emitting a function call with name="". sglang's regex captures
    // the same way; their downstream dispatch silently drops empty
    // keys. We make the rejection explicit at the parser boundary.
    let input = #"<|channel|>commentary to=functions. <|constrain|>json<|message|>{"x":1}<|call|>"#
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // No function call emitted; the block becomes a no-phase commentary
    // message containing the raw JSON content.
    for item in items {
      if case .functionCall = item {
        Issue.record("Should not emit a function call with empty name")
      }
    }
  }

  @Test
  func `Built-in tool call on analysis channel keeps the prefix`() {
    let input = #"<|channel|>analysis to=browser.search<|message|>{"query":"swift"}<|call|>"#
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "browser.search")
    #expect(f.arguments == #"{"query":"swift"}"#)
  }

  @Test
  func `Reasoning then tool call sequence`() {
    let input = (
      "<|channel|>analysis<|message|>Need to use get_weather.<|end|>"
        + "<|start|>assistant<|channel|>commentary to=functions.get_weather<|message|>"
        + #"{"city":"SF"}<|call|>"#,
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning first"); return }
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call second"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Reasoning then final message`() {
    let input = (
      "<|channel|>analysis<|message|>2 + 2 = 4<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>The answer is 4.<|return|>",
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .message(m) = items[1] else { Issue.record("Expected message"); return }
    #expect(m.phase == .finalAnswer)
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record("Expected reasoningText"); return }
    #expect(rPart.text == "2 + 2 = 4")
    guard case let .outputText(mPart) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(mPart.text == "The answer is 4.")
  }

  @Test
  func `Empty content blocks open and close cleanly without empty items`() {
    let input = "<|channel|>analysis<|message|><|end|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // We don't open a reasoning item with no content — saves a useless empty item.
    #expect(items.isEmpty)
  }

  @Test
  func `Multiple parallel function calls`() {
    let input = (
      #"<|channel|>commentary to=functions.f1<|message|>{"a":1}<|call|>"#
        + #"<|start|>assistant<|channel|>commentary to=functions.f2<|message|>{"b":2}<|call|>"#,
    )
    var parser = HarmonyParser()
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
    #expect(a.arguments == #"{"a":1}"#)
    #expect(b.arguments == #"{"b":2}"#)
  }
}

@Suite("HarmonyParser — tool-call argument whitespace")
struct HarmonyToolCallWhitespaceTests {
  // sglang's harmony_parser strips both ends of the tool-call body
  // before producing a `tool_call` event. Some templates put a newline
  // between `<|message|>` and the JSON body. We mirror sglang by skipping
  // leading whitespace before the first delta and trimming trailing
  // whitespace immediately before the close marker.
  @Test
  func `Leading newline between <|message|> and JSON body is stripped`() throws {
    let input = "<|channel|>commentary to=functions.f<|message|>\n{\"x\":1}<|call|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.arguments == #"{"x":1}"#)
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Trailing whitespace before <|call|> is stripped`() {
    let input = "<|channel|>commentary to=functions.f<|message|>{\"x\":1}\n<|call|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.arguments == #"{"x":1}"#)
  }

  @Test
  func `Both ends padded: surrounding whitespace removed`() {
    let input = "<|channel|>commentary to=functions.f<|message|>\n  {\"x\":1}  \n<|call|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.arguments == #"{"x":1}"#)
  }

  @Test
  func `Trailing whitespace held mid-stream emits as interior bytes when content follows`() throws {
    // The parser holds trailing whitespace mid-stream so the
    // cumulative emit matches sglang's stripped one-shot output.
    // When non-whitespace follows the held whitespace in a later
    // chunk, the held bytes become interior and emit naturally.
    var parser = HarmonyParser()
    var events: [ResponseStreamingEvent] = []
    events += parser.process(ParserInput(text:
      "<|channel|>commentary to=functions.f<|message|>{\"a\":1,\n"))
    events += parser.process(ParserInput(text:
      "  \"b\":2}<|call|>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["a"] as? Int == 1)
    #expect(decoded["b"] as? Int == 2)
  }

  @Test
  func `Mid-stream trailing whitespace doesn't leak into args deltas`() {
    // After receiving `<|message|>{"x":1}\n` (no close marker yet),
    // the cumulative `args` field on the in-flight item must equal
    // `{"x":1}` — the trailing newline is held back for sglang
    // strip-equivalence, not emitted as a delta. (When the next
    // chunk arrives with `<|call|>` the held bytes are dropped.)
    var parser = HarmonyParser()
    var events: [ResponseStreamingEvent] = []
    events += parser.process(ParserInput(text:
      "<|channel|>commentary to=functions.f<|message|>{\"x\":1}\n"))
    // Inspect the cumulative args from emitted deltas so far.
    let cumulative = events.compactMap { ev -> String? in
      if case let .functionCallArgumentsDelta(e) = ev { return e.delta } else { return nil }
    }.joined()
    #expect(cumulative == #"{"x":1}"#)

    events += parser.process(ParserInput(text: "<|call|>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.arguments == #"{"x":1}"#)
  }

  @Test
  func `Char-by-char streaming with leading newline still strips it`() {
    let input = "<|channel|>commentary to=functions.f<|message|>\n{\"x\":1}<|call|>"
    var parser = HarmonyParser()
    var events: [ResponseStreamingEvent] = []
    for char in input {
      events += parser.process(ParserInput(text: String(char)))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.arguments == #"{"x":1}"#)
  }
}

@Suite("HarmonyParser — streaming")
struct HarmonyStreamingTests {
  @Test
  func `Char-by-char streaming reconstructs the same items as one-shot`() {
    let input = (
      "<|channel|>analysis<|message|>thinking content<|end|>"
        + "<|start|>assistant<|channel|>commentary to=functions.f<|message|>"
        + #"{"k":"v"}<|call|>"#
        + "<|start|>assistant<|channel|>final<|message|>Done<|return|>",
    )

    var streaming = HarmonyParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = HarmonyParser()
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
          #expect(sm.phase == om.phase)
        case let (.reasoning(sr), .reasoning(or)):
          #expect(sr.content == or.content)
        default:
          Issue.record("Item kinds differ at index"); return
      }
    }
  }

  @Test
  func `Marker split across chunks does not leak as content`() {
    let chunks = [
      "<|chan",
      "nel|>analysis<|mes",
      "sage|>partial<|en",
      "d|>",
    ]
    var parser = HarmonyParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .reasoningText(part) = r.content[0] else { Issue.record("Expected reasoningText"); return }
    #expect(part.text == "partial")
  }

  @Test
  func `Reasoning streams incremental deltas`() {
    let chunks = [
      "<|channel|>analysis<|message|>",
      "first ",
      "second ",
      "third",
      "<|end|>",
    ]
    var parser = HarmonyParser()
    var allEvents: [ResponseStreamingEvent] = []
    for chunk in chunks {
      allEvents += parser.process(ParserInput(text: chunk))
    }
    allEvents += parser.finalize()
    let deltas = allEvents.compactMap { ev -> String? in
      if case let .reasoningDelta(e) = ev { return e.delta } else { return nil }
    }
    // Three streamed deltas — opener and closer arrive separately.
    #expect(deltas.count == 3)
    #expect(deltas.joined() == "first second third")
  }

  @Test
  func `Tool-call args stream incrementally`() {
    let chunks = [
      "<|channel|>commentary to=functions.f<|message|>",
      #"{"x":"#,
      #" 42}"#,
      "<|call|>",
    ]
    var parser = HarmonyParser()
    var allEvents: [ResponseStreamingEvent] = []
    for chunk in chunks {
      allEvents += parser.process(ParserInput(text: chunk))
    }
    allEvents += parser.finalize()
    let deltas = allEvents.compactMap { ev -> String? in
      if case let .functionCallArgumentsDelta(e) = ev { return e.delta } else { return nil }
    }
    #expect(deltas.joined() == #"{"x": 42}"#)
  }

  @Test
  func `Final-answer streams incrementally`() {
    let chunks = [
      "<|start|>assistant<|channel|>final<|message|>",
      "The ",
      "answer ",
      "is 42.",
      "<|return|>",
    ]
    var parser = HarmonyParser()
    var allEvents: [ResponseStreamingEvent] = []
    for chunk in chunks {
      allEvents += parser.process(ParserInput(text: chunk))
    }
    allEvents += parser.finalize()
    let deltas = allEvents.compactMap { ev -> String? in
      if case let .outputTextDelta(e) = ev { return e.delta } else { return nil }
    }
    #expect(deltas.joined() == "The answer is 42.")
  }

  @Test
  func `Fixed chunks preserve exact reasoning tool and final deltas`() {
    let chunks = [
      "<|channel|>analysis<|message|>",
      "think ",
      "more",
      "<|end|>",
      "<|start|>assistant<|channel|>commentary to=functions.f<|message|>",
      #"{"x":"#,
      "1",
      "}",
      "<|call|>",
      "comment",
      "ary",
      "<|start|>assistant<|channel|>final<|message|>",
      "Done",
      "<|return|>",
    ]

    var parser = HarmonyParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes == [0, 1, 2])
    #expect(harmonyReasoningDeltas(from: events) == ["think ", "more"])
    #expect(harmonyArgumentDeltas(from: events) == [#"{"x":"#, "1", "}"])
    #expect(harmonyOutputTextDeltas(from: events) == ["Done"])
  }
}

private func harmonyReasoningDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap {
    if case let .reasoningDelta(e) = $0 { return e.delta }
    return nil
  }
}

private func harmonyOutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap {
    if case let .outputTextDelta(e) = $0 { return e.delta }
    return nil
  }
}

private func harmonyArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap {
    if case let .functionCallArgumentsDelta(e) = $0 { return e.delta }
    return nil
  }
}

@Suite("HarmonyParser — commentary filler")
struct HarmonyCommentaryFillerTests {
  @Test
  func `Standalone commentary word after <|call|> is filtered`() {
    let input = (
      #"<|channel|>commentary to=functions.f1<|message|>{"x":1}<|call|>"#
        + "commentary"
        + #"<|channel|>commentary to=functions.f2<|message|>{"x":2}<|call|>"#,
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Two tool calls, no spurious "commentary" message.
    #expect(items.count == 2)
    for item in items {
      guard case .functionCall = item else { Issue.record("Expected function call"); return }
    }
  }

  @Test
  func `Commentary filler split across chunks is also filtered`() {
    let chunks = [
      #"<|channel|>commentary to=functions.f1<|message|>{"x":1}<|call|>"#,
      "comment",
      "ary",
      #"<|channel|>commentary to=functions.f2<|message|>{"x":2}<|call|>"#,
    ]
    var parser = HarmonyParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    for item in items {
      guard case .functionCall = item else { Issue.record("Expected function call"); return }
    }
  }

  @Test
  func `Real text after <|call|> is not filtered as commentary`() {
    let input = (
      #"<|channel|>commentary to=functions.f1<|message|>{"x":1}<|call|>"#
        + "real content here"
        + "<|start|>assistant<|channel|>final<|message|>Done<|return|>",
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Tool call, then idle "real content here" message, then final message.
    #expect(items.count == 3)
    guard case .functionCall = items[0] else { Issue.record("Expected function call"); return }
    guard case let .message(m1) = items[1] else { Issue.record("Expected idle message"); return }
    #expect(m1.phase == nil)
    guard case let .outputText(part1) = m1.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part1.text == "real content here")
    guard case let .message(m2) = items[2] else { Issue.record("Expected final message"); return }
    #expect(m2.phase == .finalAnswer)
  }
}

@Suite("HarmonyParser — finalize")
struct HarmonyFinalizeTests {
  @Test
  func `Truncated reasoning closes as incomplete`() {
    let input = "<|channel|>analysis<|message|>partial thought"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .incomplete)
    guard case let .reasoningText(part) = r.content[0] else { Issue.record("Expected reasoningText"); return }
    #expect(part.text == "partial thought")
  }

  @Test
  func `Truncated tool call closes as incomplete`() {
    let input = #"<|channel|>commentary to=functions.f<|message|>{"k":"#
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.status == .incomplete)
    #expect(f.name == "f")
    #expect(f.arguments == #"{"k":"#)
  }

  @Test
  func `Final block without <|return|> closes as completed`() {
    let input = "<|channel|>final<|message|>The answer is 42"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == .finalAnswer)
    #expect(m.status == .completed)
  }

  @Test
  func `Truncated commentary block closes as incomplete`() {
    let input = "<|channel|>commentary<|message|>incomplete commentary"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == .commentary)
    #expect(m.status == .incomplete)
  }
}

@Suite("HarmonyParser — continuation")
struct HarmonyContinuationTests {
  @Test
  func `priorOutput ending mid-analysis resumes in reasoning state`() {
    var parser = ResponseFormat.harmony.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<|channel|>analysis<|message|>Let me think about",
    )
    let events = parser.process(ParserInput(text: " this carefully<|end|>")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == " this carefully")
  }

  @Test
  func `priorOutput with closed analysis block starts fresh in idle`() {
    var parser = ResponseFormat.harmony.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<|channel|>analysis<|message|>thinking<|end|>",
    )
    let events = parser.process(ParserInput(text: "plain content")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    // No phase: idle text, not a finalAnswer or commentary block.
    #expect(m.phase == nil)
  }

  @Test
  func `priorOutput ending mid-commentary does not resume reasoning`() {
    var parser = ResponseFormat.harmony.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<|channel|>commentary<|message|>partial preamble",
    )
    let events = parser.process(ParserInput(text: "more text")) + parser.finalize()
    let items = accumulateItems(from: events)
    // The mid-commentary case is not a reasoning resume; per decision #7
    // we resume only at item boundaries. The parser starts in idle and
    // emits the new tokens as a no-phase message.
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record(""); return }
    #expect(m.phase == nil)
  }

  @Test
  func `priorOutput ending mid-tool-call does not resume reasoning`() {
    var parser = ResponseFormat.harmony.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: #"<|channel|>commentary to=functions.f<|message|>{"partial":"#,
    )
    let events = parser.process(ParserInput(text: "more text")) + parser.finalize()
    let items = accumulateItems(from: events)
    // Mid-tool-call resume is explicitly unsupported (decision #7).
    guard case .message = items[0] else { Issue.record(""); return }
  }

  @Test
  func `priorOutput with analysis closed by <|call|> (built-in tool) does not resume`() {
    var parser = ResponseFormat.harmony.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: #"<|channel|>analysis to=browser.search<|message|>{"q":"x"}<|call|>"#,
    )
    let events = parser.process(ParserInput(text: "after call")) + parser.finalize()
    let items = accumulateItems(from: events)
    // The analysis block ended at <|call|>; the new tokens are fresh idle text.
    guard case .message = items[0] else { Issue.record(""); return }
  }

  @Test
  func `HarmonyParser InitialState.inReasoning starts inside an analysis block`() {
    var parser = HarmonyParser(initialState: .inReasoning)
    let events = parser.process(ParserInput(text: "thinking<|end|>")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
  }

  @Test
  func `nil priorOutput starts fresh in idle`() {
    var parser = ResponseFormat.harmony.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: "hello")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record(""); return }
    #expect(m.phase == nil)
  }
}

@Suite("HarmonyParser — SGLang adversarial")
struct HarmonySGLangAdversarialTests {
  // Adversarial fixtures lifted from
  // sglang/test/registered/unit/parser/test_harmony_parser.py.

  @Test
  func `Unknown channel name routes to no-phase message (deliberate SGLang divergence)`() {
    // SGLang holds the block as incomplete (zero events) when the channel
    // name is unrecognized. We deliberately fall through to a no-phase
    // message so the content is preserved rather than dropped — see the
    // `parseChannelHeader` `.unknown` branch in HarmonyParser.swift.
    let input = "<|channel|>unknown<|message|>content<|end|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == nil)
    guard case let .outputText(part) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part.text == "content")
  }

  @Test
  func `Mixed unknown structural tokens pass through as content; valid block parses`() {
    let input = "text <|weird|> more text <|channel|>analysis<|message|>content<|end|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected leading message"); return }
    #expect(m.phase == nil)
    guard case let .outputText(part) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part.text == "text <|weird|> more text ")
    guard case let .reasoning(r) = items[1] else { Issue.record("Expected reasoning"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record("Expected reasoningText"); return }
    #expect(rPart.text == "content")
  }

  @Test
  func `Stray structural tokens in TEXT position are silently consumed`() {
    // Stray `<|end|>` and `<|call|>` outside any block are dropped without
    // emitting spurious items; surrounding text is preserved.
    let input = "<|end|>text <|call|>more<|channel|>analysis<|message|>think<|end|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // One idle message (the `<|call|>` arms commentary-filler filtering,
    // which discards the leading "more" word; "text " survives).
    // The reasoning block follows.
    // We accept either commentary-filtered or pass-through behavior so the
    // test pins resilience without over-specifying.
    let messageItems = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let reasoningItems = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoningItems.count == 1)
    guard case let .reasoningText(rPart) = reasoningItems[0].content[0] else {
      Issue.record("Expected reasoningText"); return
    }
    #expect(rPart.text == "think")
    let allMessageText = messageItems.compactMap { msg -> String? in
      if case let .outputText(p) = msg.content[0] { return p.text } else { return nil }
    }.joined()
    #expect(allMessageText.contains("text "))
  }

  @Test
  func `Partial structural-token suffix is held until next chunk`() {
    // Input ends in `<|ret`, a prefix of `<|return|>`. Verify the parser
    // holds the suffix and emits only `complete text ` as content.
    let input = "complete text <|ret"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input))
    let deltas = events.compactMap { ev -> String? in
      if case let .outputTextDelta(e) = ev { return e.delta } else { return nil }
    }
    // The suffix `<|ret` is not yet emitted — it's held in the buffer.
    #expect(deltas.joined() == "complete text ")
  }

  @Test
  func `Repetitive tool calls with commentary filler do not emit spurious messages`() {
    let input = (
      "<|channel|>analysis<|message|>Need to get weather<|end|>"
        + #"<|start|>assistant<|channel|>commentary to=functions.get_weather<|message|>{"city":"Boston"}<|call|>"#
        + "commentary"
        + #"<|channel|>commentary to=functions.get_weather<|message|>{"city":"Boston"}<|call|>"#
        + "commentary"
        + #"<|channel|>commentary to=functions.get_weather<|message|>{"city":"Boston"}<|call|>"#
        + "<|channel|>analysis<|message|>Tool not responding<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>Unable to fetch weather data<|return|>",
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoningItems = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    let toolItems = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    let messageItems = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(reasoningItems.count == 2)
    #expect(toolItems.count == 3)
    #expect(messageItems.count == 1)
    // No "commentary" filler leaked as a normal message.
    for msg in messageItems {
      guard case let .outputText(part) = msg.content[0] else { continue }
      #expect(part.text.trimmingCharacters(in: .whitespaces).lowercased() != "commentary")
    }
    guard case let .reasoningText(r0) = reasoningItems[0].content[0] else {
      Issue.record("Expected reasoningText"); return
    }
    #expect(r0.text == "Need to get weather")
    guard case let .reasoningText(r1) = reasoningItems[1].content[0] else {
      Issue.record("Expected reasoningText"); return
    }
    #expect(r1.text == "Tool not responding")
    guard case let .outputText(mPart) = messageItems[0].content[0] else {
      Issue.record("Expected outputText"); return
    }
    #expect(mPart.text == "Unable to fetch weather data")
  }

  @Test
  func `Final answer reaches the consumer despite stray <|call|>commentary<|return|> sequence`() {
    // SGLang's `test_canonical_call_with_text_commentary_after` fixture:
    // an analysis block closes with `<|end|>`, then a stray `<|call|>` is
    // followed by the word "commentary" (filler), then `<|return|>`,
    // then a fresh final block. We verify the final answer arrives.
    let input = (
      "<|start|><|channel|>analysis<|message|>think<|end|>"
        + "<|call|>commentary<|return|>"
        + "<|channel|>final<|message|>result<|end|>",
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let messageItems = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    // Final answer must be present.
    let finalAnswers = messageItems.filter { $0.phase == .finalAnswer }
    #expect(finalAnswers.count == 1)
    guard case let .outputText(part) = finalAnswers[0].content[0] else {
      Issue.record("Expected outputText"); return
    }
    #expect(part.text == "result")
  }
}

@Suite("HarmonyParser — dispatch and edge cases")
struct HarmonyDispatchTests {
  @Test
  func `Dispatch via ResponseFormat.harmony.makeParser`() {
    let parser = ResponseFormat.harmony.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Name prefix gpt-oss routes to .harmony`() {
    let format = ResponseFormat.infer(modelName: "gpt-oss-20b", modelType: "", modelConfig: [:])
    #expect(format == .harmony)
  }

  @Test
  func `Model type gpt_oss routes to .harmony`() {
    let format = ResponseFormat.infer(modelName: "", modelType: "gpt_oss", modelConfig: [:])
    #expect(format == .harmony)
  }

  @Test
  func `Tool response message is treated as no-phase content`() {
    let input = #"<|start|>functions.get_weather to=assistant<|message|>{"sunny":true}<|end|>"#
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == nil)
    guard case let .outputText(part) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part.text == #"{"sunny":true}"#)
  }

  @Test
  func `Trailing text after <|return|> emits as a fresh no-phase message`() {
    // Pins divergence from sglang's `_parse_block`, which absorbs the
    // trailing TEXT into the same final message. Auto-injected halt
    // means streaming callers don't reach this path; offline-parse
    // callers get cleaner item separation when prior output contains
    // post-`<|return|>` content.
    let input = (
      "<|start|>assistant<|channel|>final<|message|>The answer.<|return|>"
        + " Trailing text.",
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(final) = items[0] else { Issue.record("Expected final message"); return }
    #expect(final.phase == .finalAnswer)
    guard case let .outputText(finalPart) = final.content[0] else { Issue.record(""); return }
    #expect(finalPart.text == "The answer.")
    guard case let .message(trailing) = items[1] else { Issue.record("Expected trailing no-phase message"); return }
    #expect(trailing.phase == nil)
    guard case let .outputText(trailingPart) = trailing.content[0] else { Issue.record(""); return }
    #expect(trailingPart.text == " Trailing text.")
  }

  @Test
  func `Interspersed plain text emits as separate no-phase messages`() {
    let input = (
      "Some text "
        + "<|channel|>analysis<|message|>thinking<|end|>"
        + " more text "
        + "<|start|>assistant<|channel|>final<|message|>answer<|return|>",
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Layout: idle "Some text ", reasoning "thinking", idle " more text ", final "answer".
    #expect(items.count == 4)
    guard case let .message(m0) = items[0] else { Issue.record("Expected leading message"); return }
    #expect(m0.phase == nil)
    guard case let .outputText(p0) = m0.content[0] else { Issue.record(""); return }
    #expect(p0.text == "Some text ")
    guard case .reasoning = items[1] else { Issue.record("Expected reasoning"); return }
    guard case let .message(m2) = items[2] else { Issue.record("Expected interstitial message"); return }
    #expect(m2.phase == nil)
    guard case let .outputText(p2) = m2.content[0] else { Issue.record(""); return }
    #expect(p2.text == " more text ")
    guard case let .message(m3) = items[3] else { Issue.record("Expected final message"); return }
    #expect(m3.phase == .finalAnswer)
  }

  @Test
  func `Output_index counters increment monotonically across items`() {
    let input = (
      "<|channel|>analysis<|message|>think<|end|>"
        + "<|start|>assistant<|channel|>commentary to=functions.f<|message|>{}<|call|>"
        + "<|start|>assistant<|channel|>final<|message|>done<|return|>",
    )
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let addedIndices = events.compactMap { ev -> Int? in
      if case let .outputItemAdded(e) = ev { return e.outputIndex } else { return nil }
    }
    // Three items → 0, 1, 2.
    #expect(addedIndices == [0, 1, 2])
  }

  @Test
  func `All function-call IDs are distinct from call IDs`() {
    let input = #"<|channel|>commentary to=functions.f<|message|>{}<|call|>"#
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
    #expect(f.id != f.callId)
  }
}

@Suite("HarmonyParser — channel name matching")
struct HarmonyChannelMatchingTests {
  @Test
  func `Channel name analysis-foo does not match the analysis channel`() {
    let input = "<|channel|>analysis-foo<|message|>body<|end|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // `analysis-foo` is unknown; routes to a no-phase message rather
    // than reasoning.
    for item in items {
      if case .reasoning = item {
        Issue.record("`analysis-foo` should not produce a reasoning item")
      }
    }
  }
}

@Suite("HarmonyParser — text fallback (skip_special_tokens)")
struct HarmonyTextFallbackTests {
  @Test
  func `assistantfinal-only input emits a final-answer message`() {
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: "assistantfinal The direct answer.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == .finalAnswer)
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(t.text == "The direct answer.")
  }

  @Test
  func `analysis...assistantfinal splits reasoning and final answer`() {
    let input = "analysis I need to think about this. assistantfinal The answer is 42."
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning first"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record("Expected reasoning text"); return }
    #expect(rPart.text == "I need to think about this. ")
    guard case let .message(m) = items[1] else { Issue.record("Expected message second"); return }
    #expect(m.phase == .finalAnswer)
    guard case let .outputText(mPart) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(mPart.text == "The answer is 42.")
  }

  @Test
  func `analysis and assistantfinal labels do not require following spaces`() {
    let input = "analysisThe user typed random strings.assistantfinalIt looks like you're testing."
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0],
          case let .message(m) = items[1],
          case let .outputText(mt) = m.content[0]
    else {
      Issue.record("Expected reasoning then final message"); return
    }
    #expect(rt.text == "The user typed random strings.")
    #expect(m.phase == .finalAnswer)
    #expect(mt.text == "It looks like you're testing.")
  }

  @Test
  func `commentary...assistantfinal splits commentary and final answer`() {
    let input = "commentary User-visible preamble. assistantfinal The answer is 42."
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(c) = items[0] else { Issue.record("Expected commentary first"); return }
    #expect(c.phase == .commentary)
    guard case let .message(f) = items[1] else { Issue.record("Expected final second"); return }
    #expect(f.phase == .finalAnswer)
  }

  @Test
  func `Char-by-char streaming holds back a partial assistantfinal`() {
    let input = "analysis reasoning content assistantfinal the answer"
    var streaming = HarmonyParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = HarmonyParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )
    #expect(streamingItems.count == oneShotItems.count)
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.reasoning(sr), .reasoning(or)):
          guard case let .reasoningText(sp) = sr.content[0],
                case let .reasoningText(op) = or.content[0]
          else {
            Issue.record("Reasoning content shape differs"); return
          }
          #expect(sp.text == op.text)
        case let (.message(sm), .message(om)):
          #expect(sm.phase == om.phase)
          guard case let .outputText(sp) = sm.content[0],
                case let .outputText(op) = om.content[0]
          else {
            Issue.record("Message content shape differs"); return
          }
          #expect(sp.text == op.text)
        default:
          Issue.record("Item kinds differ"); return
      }
    }
  }

  @Test
  func `Canonical input takes priority when markers are present`() {
    let input = "<|channel|>analysis<|message|>Reasoning<|end|>assistantfinal stray"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Canonical mode handles the analysis block; the trailing literal
    // "assistantfinal stray" is plain content (no marker entry).
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
  }

  @Test
  func `Plain text (no labels, no markers) routes to plain message`() {
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: "Just plain content.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == nil)
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(t.text == "Just plain content.")
  }

  @Test
  func `assistant analysis (with whitespace) is recognized as the analysis channel`() {
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: "assistant analysis I need to think.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "I need to think.")
  }

  @Test
  func `assistant commentary (with whitespace) is recognized as the commentary channel`() {
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: "assistant commentary User-visible preamble.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.phase == .commentary)
  }

  @Test
  func `assistant analysis ... assistant final splits reasoning and final`() {
    let input = "assistant analysis I need to think. assistantfinal The answer is 42."
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning first"); return
    }
    #expect(rt.text == "I need to think. ")
    guard case let .message(m) = items[1] else { Issue.record("Expected message second"); return }
    #expect(m.phase == .finalAnswer)
  }

  @Test
  func `Char-by-char streaming holds back a partial assistant analy`() {
    let input = "assistant analysis reasoning content"
    var streaming = HarmonyParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = HarmonyParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )
    #expect(streamingItems.count == oneShotItems.count)
    guard case let .reasoning(sr) = streamingItems[0],
          case let .reasoning(or) = oneShotItems[0],
          case let .reasoningText(sp) = sr.content[0],
          case let .reasoningText(op) = or.content[0]
    else {
      Issue.record("Expected reasoning in both"); return
    }
    #expect(sp.text == op.text)
  }
}

@Suite("HarmonyParser — reasoning lenient close")
struct HarmonyReasoningLenientCloseTests {
  @Test
  func `Reasoning closed by stray <|return|> instead of <|end|>`() {
    let input = "<|channel|>analysis<|message|>Some reasoning.<|return|>"
    var parser = HarmonyParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "Some reasoning.", "Reasoning text must not contain `<|return|>` literal bytes")
    #expect(r.status == .completed)
  }
}
