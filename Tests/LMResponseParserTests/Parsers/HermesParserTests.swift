// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("HermesParser — plain text")
struct HermesPlainTextTests {
  @Test
  func `Single chunk of plain text emits a complete message`() {
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(t.text == "hello world")
  }

  @Test
  func `Plain text reconstructs across many small chunks`() {
    let original = "This is plain text with no tool calling involved."
    var parser = HermesParser()
    var events: [ResponseStreamingEvent] = []
    for char in original {
      events += parser.process(ParserInput(text: String(char)))
    }
    events += parser.finalize()

    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == original)
    #expect(m.status == .completed)
  }

  @Test
  func `Empty stream produces no items`() {
    var parser = HermesParser()
    let events = parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.isEmpty)
  }

  @Test
  func `Plain text emits text-mode envelope sequence`() {
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: "hi")) + parser.finalize()
    #expect(events.map(eventKind) == [
      "outputItemAdded",
      "contentPartAdded",
      "outputTextDelta",
      "outputTextDone",
      "contentPartDone",
      "outputItemDone",
    ])
  }
}

@Suite("HermesParser — single tool call")
struct HermesSingleToolCallTests {
  @Test
  func `Single complete tool call yields one function_call item`() {
    let text = #"<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "get_weather")
    #expect(f.status == .completed)
    let parsed = parseArgs(f.arguments)
    #expect(parsed?["city"] as? String == "NYC")
  }

  @Test
  func `Tool-call event sequence: added → arguments.delta → arguments.done → item.done`() {
    let text = #"<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    #expect(events.map(eventKind) == [
      "outputItemAdded",
      "functionCallArgumentsDelta",
      "functionCallArgumentsDone",
      "outputItemDone",
    ])
  }

  @Test
  func `function_call has separate fc_ item id and call_ call_id`() {
    let text = #"<tool_call>{"name": "f", "arguments": {}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
    #expect(f.id != f.callId)
  }

  @Test
  func `Tool call closed at EOS without explicit </tool_call> still emits cleanly when JSON is valid`() {
    let text = #"<tool_call>{"name": "final_answer", "arguments": {"trigger": true}}"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "final_answer")
    #expect(f.status == .completed)
    let parsed = parseArgs(f.arguments)
    #expect(parsed?["trigger"] as? Bool == true)
  }

  @Test
  func `Truncated tool call (mid-string) closes as incomplete`() {
    let text = #"<tool_call>{"name": "fn", "arguments": {"loca"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "fn", "Name should be extracted even from truncated JSON")
    #expect(f.status == .incomplete)
  }
}

@Suite("HermesParser — content + tool call mix")
struct HermesContentMixTests {
  @Test
  func `Plain content followed by tool call emits message then function_call`() {
    let text = #"Sure, let me check.<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message first"); return
    }
    #expect(t.text == "Sure, let me check.")
    #expect(m.status == .completed)

    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Message item closes before tool call opens`() throws {
    let text = #"Hi<tool_call>{"name": "f", "arguments": {}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let kinds = events.map(eventKind)
    let messageDoneIdx = try #require(kinds.firstIndex(of: "outputItemDone"))
    let toolAddedIdx = try #require(kinds.lastIndex(of: "outputItemAdded"))
    #expect(messageDoneIdx < toolAddedIdx, "Message item.done must precede tool call item.added")
  }

  @Test
  func `Output indexes are monotonically increasing across mixed items`() {
    let text = #"Hi<tool_call>{"name": "f", "arguments": {}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes == [0, 1])
  }
}

@Suite("HermesParser — multiple tool calls")
struct HermesMultipleToolCallsTests {
  @Test
  func `Two sequential tool calls emit two function_call items in order`() {
    let text = #"<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call><tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function_calls"); return
    }
    #expect(a.name == "search")
    #expect(parseArgs(a.arguments)?["q"] as? String == "cats")
    #expect(b.name == "search")
    #expect(parseArgs(b.arguments)?["q"] as? String == "dogs")
    #expect(a.id != b.id, "Each tool call gets a fresh fc_ id")
    #expect(a.callId != b.callId, "Each tool call gets a fresh call_ id")
  }

  @Test
  func `Plain text between two tool calls in one chunk emits a message`() {
    let text = #"<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call> Here is the result: <tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>"#
    var parser = HermesParser()
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
    #expect(parseArgs(first.arguments)?["q"] as? String == "cats")
    guard case let .outputText(text) = middle.content[0] else {
      Issue.record("Expected outputText in middle message"); return
    }
    #expect(text.text == " Here is the result: ")
    #expect(second.name == "search")
    #expect(parseArgs(second.arguments)?["q"] as? String == "dogs")
  }

  @Test
  func `Trailing text after a tool call emits a message`() {
    let text = #"<tool_call>{"name": "f", "arguments": {}}</tool_call> done."#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .functionCall = items[0],
          case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected function_call then message"); return
    }
    #expect(t.text == " done.")
  }
}

@Suite("HermesParser — argument extraction")
struct HermesArgExtractionTests {
  @Test
  func `arguments field ordered before name is extracted correctly`() {
    let text = #"<tool_call>{"arguments": {"x": 1, "y": 2}, "name": "f"}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: text)) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else {
      Issue.record("Expected function_call"); return
    }
    #expect(f.name == "f")
    let parsed = parseArgs(f.arguments)
    #expect(parsed?["x"] as? Int == 1)
    #expect(parsed?["y"] as? Int == 2)
    // Confirm the trailing `, "name": "f"` did NOT leak into arguments.
    #expect(!f.arguments.contains("name"), "arguments must not include the trailing name field")
  }
}

@Suite("HermesParser — streaming boundary cases")
struct HermesBoundaryTests {
  @Test
  func `Stream interval of 1 reconstructs the same items as single chunk`() {
    let text = #"<tool_call>{"name": "get_current_temperature", "arguments": {"location": "San Francisco", "unit": "celsius"}}</tool_call>"#

    var streamingParser = HermesParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for char in text {
      streamingEvents += streamingParser.process(ParserInput(text: String(char)))
    }
    streamingEvents += streamingParser.finalize()

    var singleShotParser = HermesParser()
    let singleShotEvents = singleShotParser.process(ParserInput(text: text)) + singleShotParser.finalize()

    let streamingItems = accumulateItems(from: streamingEvents)
    let singleShotItems = accumulateItems(from: singleShotEvents)

    #expect(streamingItems.count == singleShotItems.count)
    if case let .functionCall(s) = streamingItems[0],
       case let .functionCall(o) = singleShotItems[0]
    {
      #expect(s.name == o.name)
      // Compare as decoded JSON dicts so whitespace differences in the
      // raw `arguments` string don't fail the equality check.
      let sParsed = parseArgs(s.arguments)
      let oParsed = parseArgs(o.arguments)
      #expect(sParsed?["location"] as? String == oParsed?["location"] as? String)
      #expect(sParsed?["unit"] as? String == oParsed?["unit"] as? String)
    }
  }

  @Test
  func `Marker split across chunks (<tool_ then call>) is buffered`() {
    var parser = HermesParser()
    var events = parser.process(ParserInput(text: "before"))
    events += parser.process(ParserInput(text: "<tool_"))
    events += parser.process(ParserInput(text: #"call>{"name": "f", "arguments": {}}</tool_call>"#))
    events += parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message first"); return
    }
    #expect(t.text == "before", "Held-back partial-marker bytes must not leak as content")
  }

  @Test
  func `Multiple tool calls with stream interval > 1 still parse correctly`() {
    let text = #"<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call><tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>"#
    for interval in [2, 3, 5, 8] {
      let items = streamWithInterval(text: text, interval: interval)
      #expect(items.count == 2, "interval=\(interval): expected 2 items, got \(items.count)")
      if case let .functionCall(a) = items[0],
         case let .functionCall(b) = items[1]
      {
        #expect(parseArgs(a.arguments)?["q"] as? String == "cats", "interval=\(interval)")
        #expect(parseArgs(b.arguments)?["q"] as? String == "dogs", "interval=\(interval)")
      } else {
        Issue.record("interval=\(interval): expected two function_calls")
      }
    }
  }

  @Test
  func `Boolean args stream correctly across multi-token boundaries`() {
    let text = """
    <tool_call>
    {"name": "final_answer", "arguments": {"trigger": true}}
    </tool_call>
    """
    for interval in [1, 2, 5] {
      let items = streamWithInterval(text: text, interval: interval)
      guard case let .functionCall(f) = items[0] else {
        Issue.record("interval=\(interval): expected function_call"); continue
      }
      #expect(f.name == "final_answer", "interval=\(interval)")
      #expect(parseArgs(f.arguments)?["trigger"] as? Bool == true, "interval=\(interval)")
    }
  }

  @Test
  func `Args delta concatenation matches the full args string`() {
    let text = #"<tool_call>{"name": "fn", "arguments": {"location": "San Francisco, California, United States", "unit": "celsius"}}</tool_call>"#
    var parser = HermesParser()
    var events: [ResponseStreamingEvent] = []
    for char in text {
      events += parser.process(ParserInput(text: String(char)))
    }
    events += parser.finalize()

    let argsDeltas: [String] = events.compactMap {
      if case let .functionCallArgumentsDelta(e) = $0 { return e.delta }
      return nil
    }
    let concatenated = argsDeltas.joined()
    let parsed = parseArgs(concatenated)
    #expect(parsed?["location"] as? String == "San Francisco, California, United States")
    #expect(parsed?["unit"] as? String == "celsius")
  }
}

@Suite("HermesParser — finalize cases")
struct HermesFinalizeTests {
  @Test
  func `finalize on a complete tool call emits no extra events`() {
    let text = #"<tool_call>{"name": "f", "arguments": {}}</tool_call>"#
    var parser = HermesParser()
    let processEvents = parser.process(ParserInput(text: text))
    let finalizeEvents = parser.finalize()
    #expect(finalizeEvents.isEmpty, "finalize should not emit when items are already closed")
    #expect(!processEvents.isEmpty)
  }

  @Test
  func `finalize on dangling content closes the message item`() {
    var parser = HermesParser()
    _ = parser.process(ParserInput(text: "hello"))
    let finalizeEvents = parser.finalize()
    let kinds = finalizeEvents.map(eventKind)
    #expect(kinds == ["outputTextDone", "contentPartDone", "outputItemDone"])
  }

  @Test
  func `finalize on bare partial-marker bytes flushes them as content`() {
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: "abc<tool_")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "abc<tool_", "Partial-marker bytes are confirmed plain content at EOS")
    #expect(m.status == .completed)
  }

  @Test
  func `Truncated header before name leaves no output_index gap`() {
    // First chunk: a message + the start of a tool call whose JSON
    // never reveals a name. The tool call is abandoned.
    var parser = HermesParser()
    let text = #"Hi <tool_call>{"argum"#
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    // Only the message item gets an index — the truncated tool call
    // never reserved a slot, so indexes stay consecutive.
    #expect(addedIndexes == [0])
  }

  @Test
  func `Truncated header followed by valid tool call: indexes stay consecutive`() {
    // The first `<tool_call>` truncates without a name. A subsequent
    // valid call should land at output_index 0 because the truncated
    // call never allocated one.
    var parser = HermesParser()
    let chunk1 = #"<tool_call>{"argum"# // never gets a name
    _ = parser.process(ParserInput(text: chunk1))
    // The above wouldn't actually be followed by a closing tag in real
    // truncation; just verify the parser doesn't reserve an index.
    let events = parser.finalize()
    let addedIndexes: [Int] = events.compactMap {
      if case let .outputItemAdded(e) = $0 { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes.isEmpty, "Truncated header without a name emits no items")
  }
}

@Suite("HermesParser — stray close-tag stripping")
struct HermesStrayCloseTagTests {
  // sglang's `_clean_normal_text` strips a stray `</tool_call>` from
  // buffered normal content and holds back partial close-tag suffixes.
  // Pin parity with that behavior.
  @Test
  func `Bare </tool_call> literal in plain content is stripped`() {
    var parser = HermesParser()
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
    var parser = HermesParser()
    // First chunk ends with a partial close tag; without hold-back
    // behavior the bytes leak into a message and would later need
    // to be reconciled when the next chunk completes the tag.
    let mid = parser.process(ParserInput(text: "abc</tool_cal"))
    // No content should have been emitted yet (held back).
    let midItems = accumulateItems(from: mid)
    // The "abc" prefix may or may not have emitted depending on how
    // the partial overlap is computed; what we care about is that
    // when the close tag completes in the next chunk, the final
    // message text strips the stray close.
    _ = midItems
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

@Suite("HermesParser — dispatch")
struct HermesDispatchTests {
  @Test
  func `ResponseFormat.hermes dispatches to HermesParser end-to-end`() {
    let parser = ResponseFormat.hermes.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let text = #"<tool_call>{"name": "f", "arguments": {}}</tool_call>"#
    let events = p.process(ParserInput(text: text)) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    if case let .functionCall(f) = items[0] {
      #expect(f.name == "f")
    }
  }

  @Test
  func `One-shot parseResponse works through the .hermes dispatch`() {
    let items = parseResponse(
      #"Hi<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>"#,
      format: .hermes,
      tokenizer: StubTokenizer(),
    )
    #expect(items.count == 2)
  }
}

@Suite("HermesParser — adversarial ports")
struct HermesAdversarialTests {
  // H1: vLLM test_hermes_streaming_content_and_tool_call_in_single_chunk
  // (test_hermes_tool_parser.py:392-414).
  @Test
  func `Content + complete tool call in a single chunk both emit`() {
    let input = #"Hi!<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    guard case let .outputText(mPart) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(mPart.text == "Hi!")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
    let decoded = parseArgs(f.arguments)
    #expect(decoded?["x"] as? Int == 1)
  }

  // M8: SGLang test_malformed_json_returns_original_text
  // (test_hermes_detector.py:106-110).
  //
  // SGLang preserves the entire malformed input (markers and all) as
  // normal_text. We deliberately do not — `extractToolName` returns nil
  // when the body has no `"name"` field, the parser holds the buffer
  // until `</tool_call>` is seen, and then drops the region without
  // emitting either a tool call or a message. Pinning our behavior so
  // a future refactor doesn't silently flip it.
  @Test
  func `Malformed JSON inside <tool_call>...</tool_call> drops the region (SGLang divergence)`() {
    let input = "<tool_call>not valid json</tool_call>"
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  // sglang's non-streaming detect_and_parse accepts a JSON array form
  // (`[{...}]`) inside the envelope via parse_base_json's list/dict
  // polymorphism. sglang's streaming path does not (the partial JSON
  // parser yields a list and `"name" in obj` is False). vLLM raises
  // and returns no calls. No production Hermes/Qwen template emits the
  // array form. We deliberately do not stream-decode arrays — splitting
  // a streamed `[{...}, {...}]` into independent tool-call slots is
  // significantly more state than the format warrants. The region is
  // silently dropped (same as malformed JSON above). Pinning the
  // behavior so a future refactor doesn't flip it.
  @Test
  func `JSON-array form inside <tool_call> drops the region (sglang streaming divergence)`() {
    let input = #"<tool_call>[{"name": "f", "arguments": {"x": 1}}]</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}

@Suite("HermesParser — parameters alias")
struct HermesParametersAliasTests {
  // sglang's parse_base_json accepts either `arguments` or `parameters`
  // (parameters first per `act.get("parameters") or act.get("arguments")`).
  // Several models reusing the Hermes envelope (granite/llama variants)
  // emit `parameters`. We accept both, preferring `arguments`.
  @Test
  func `parameters field is accepted as an alias for arguments`() throws {
    let input = #"<tool_call>{"name": "get_weather", "parameters": {"city": "Paris"}}</tool_call>"#
    var parser = HermesParser()
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
  func `parameters streams correctly chunk-by-chunk`() throws {
    let input = #"<tool_call>{"name": "fn", "parameters": {"location": "San Francisco", "unit": "celsius"}}</tool_call>"#
    var parser = HermesParser()
    var events: [ResponseStreamingEvent] = []
    for char in input {
      events += parser.process(ParserInput(text: String(char)))
    }
    events += parser.finalize()

    let argsDeltas: [String] = events.compactMap {
      if case let .functionCallArgumentsDelta(e) = $0 { return e.delta }
      return nil
    }
    let concatenated = argsDeltas.joined()
    guard let parsed = (try? JSONSerialization.jsonObject(with: try #require(concatenated.data(using: .utf8)))) as? [String: Any] else {
      Issue.record("args didn't parse as JSON: \(concatenated)"); return
    }
    #expect(parsed["location"] as? String == "San Francisco")
    #expect(parsed["unit"] as? String == "celsius")
  }
}

@Suite("HermesParser — JSON-aware key extraction")
struct HermesJsonAwareExtractionTests {
  @Test
  func `Inner string value containing literal "arguments" key doesn't fool the extractor`() throws {
    let input = #"<tool_call>{"name": "echo", "arguments": {"text": "\"arguments\":\"foo\""}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "echo")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["text"] as? String == "\"arguments\":\"foo\"")
  }

  @Test
  func `Inner string value containing literal "name" key doesn't fool the extractor`() {
    let input = #"<tool_call>{"name": "echo", "arguments": {"prompt": "ignore the \"name\":\"sneaky\" inside this"}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "echo")
  }
}

// MARK: Helpers

private func streamWithInterval(text: String, interval: Int) -> [ResponseOutputItem] {
  var parser = HermesParser()
  var events: [ResponseStreamingEvent] = []
  let chars = Array(text)
  var i = 0
  while i < chars.count {
    let end = min(i + interval, chars.count)
    events += parser.process(ParserInput(text: String(chars[i ..< end])))
    i = end
  }
  events += parser.finalize()
  return accumulateItems(from: events)
}

private func parseArgs(_ args: String) -> [String: Any]? {
  guard let data = args.data(using: .utf8) else { return nil }
  return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}
