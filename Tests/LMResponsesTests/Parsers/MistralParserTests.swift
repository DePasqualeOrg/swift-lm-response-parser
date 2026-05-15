// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

@Suite("MistralParser — plain text")
struct MistralPlainTextTests {
  @Test
  func `Plain text without tool calls emits a single message`() {
    var parser = MistralParser()
    let events = parser.process(ParserInput(text: "hello there")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "hello there")
  }

  @Test
  func `Empty stream emits nothing`() {
    var parser = MistralParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("MistralParser — JSON-array format")
struct MistralJsonArrayTests {
  @Test
  func `Single JSON-array tool call`() throws {
    var parser = MistralParser()
    let input = #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
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
  func `callId matches Mistral chat template constraint`() {
    var parser = MistralParser()
    let input = #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    // Upstream Mistral templates (e.g.
    // `mistralai/Mistral-7B-Instruct-v0.3`,
    // `mistralai/Ministral-8B-Instruct-2410`) enforce
    // `tool_call.id|length != 9`. Jinja's `length` counts code points,
    // so assert on `utf8.count` rather than `count` (grapheme clusters)
    // to lock in the byte-level contract. Membership in the Crockford
    // alphabet is the strongest assertion — it rules out future
    // alphabet widening that would slip past a generic
    // `Character.isLetter || isNumber` check (which matches every
    // Unicode letter / digit, including non-BMP code points).
    let crockford: Set<Character> = Set("0123456789abcdefghjkmnpqrstvwxyz")
    #expect(f.callId.utf8.count == 9)
    #expect(f.callId.allSatisfy { crockford.contains($0) })
  }

  @Test
  func `Multiple JSON-array tool calls`() {
    var parser = MistralParser()
    let input = #"[TOOL_CALLS] [{"name": "f1", "arguments": {}}, {"name": "f2", "arguments": {"x": 1}}]"#
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
}

@Suite("MistralParser — compact format")
struct MistralCompactTests {
  @Test
  func `Single compact tool call`() throws {
    var parser = MistralParser()
    let input = #"[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#
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
  func `Multiple consecutive compact tool calls`() {
    var parser = MistralParser()
    let input = #"[TOOL_CALLS]f1[ARGS]{}[TOOL_CALLS]f2[ARGS]{"x": 1}"#
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
  func `vLLM v11+ compact form without [ARGS] separator`() throws {
    var parser = MistralParser()
    let input = #"[TOOL_CALLS]get_weather{"city": "Paris"}"#
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
  func `vLLM v11+ compact form with multiple consecutive calls`() {
    var parser = MistralParser()
    let input = #"[TOOL_CALLS]f1{}[TOOL_CALLS]f2{"x": 1}"#
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
    #expect(a.arguments == "{}")
  }

  @Test
  func `vLLM v11+ compact form streamed char-by-char`() {
    let input = #"[TOOL_CALLS]fn{"x": 1}"#

    var streaming = MistralParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    #expect(streamingItems.count == 1)
    guard case let .functionCall(f) = streamingItems[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.arguments.contains("\"x\""))
  }
}

@Suite("MistralParser — surrounding text")
struct MistralSurroundingTextTests {
  @Test
  func `Text before tool call is emitted as message content`() {
    var parser = MistralParser()
    let input = #"Let me check the weather. [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }
}

@Suite("MistralParser — streaming")
struct MistralStreamingTests {
  @Test
  func `Char-by-char streaming yields the same items as single-shot`() {
    let input = #"[TOOL_CALLS]fn[ARGS]{"x": 1}"#

    var streaming = MistralParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = MistralParser()
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
  func `Marker split across chunks still parses`() {
    var parser = MistralParser()
    var events = parser.process(ParserInput(text: "[TOOL"))
    events += parser.process(ParserInput(text: "_CALLS]fn[ARGS]{}"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `Parametric stream intervals [1, 2, 4, 8] reconstruct the same args`() throws {
    // Mirrors vLLM's `@pytest.mark.parametrize("stream_interval", [...])`
    // pattern. Each interval re-runs the same input across different
    // chunk widths to catch state-machine bugs that only fire at
    // specific boundaries.
    let input = #"[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "unit": "celsius"}"#
    for interval in [1, 2, 4, 8] {
      let items = streamItems(text: input, interval: interval) { MistralParser() }
      #expect(items.count == 1, "interval=\(interval): expected 1 item, got \(items.count)")
      guard case let .functionCall(f) = items.first else {
        Issue.record("interval=\(interval): expected function call, got \(items)")
        continue
      }
      #expect(f.name == "get_weather", "interval=\(interval)")
      let parsed = try JSONSerialization.jsonObject(
        with: Data(f.arguments.utf8),
      ) as? [String: Any]
      #expect(parsed?["city"] as? String == "Paris", "interval=\(interval)")
      #expect(parsed?["unit"] as? String == "celsius", "interval=\(interval)")
    }
  }

  @Test
  func `Fixed chunks preserve exact text and argument deltas`() {
    let chunks = [
      "prefix ",
      "[TOOL",
      #"_CALLS]fn[ARGS]{"x":"#,
      "1",
      "}",
      "suffix",
    ]

    var parser = MistralParser()
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
    #expect(mistralOutputTextDeltas(from: events) == ["prefix ", "suffix"])
    #expect(mistralArgumentDeltas(from: events) == [#"{"x":1}"#])
  }

  @Test
  func `Completed compact call followed by later compact call preserves exact deltas`() {
    let chunks = [
      "[TOOL_CALLS]first[ARGS]{}",
      #"[TOOL_CALLS]second[ARGS]{"y":2}"#,
    ]

    var parser = MistralParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let outputIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(outputIndexes == [0, 1])
    #expect(mistralArgumentDeltas(from: events) == ["{}", #"{"y":2}"#])
  }
}

private func mistralOutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .outputTextDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

private func mistralArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .functionCallArgumentsDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

@Suite("MistralParser — finalize edge cases")
struct MistralFinalizeTests {
  @Test
  func `Truncated compact tool call mid-args is not emitted`() {
    var parser = MistralParser()
    let events = parser.process(ParserInput(text: #"[TOOL_CALLS]fn[ARGS]{"x": "incomp"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Either no items (parser refused) or an incomplete-status item.
    if items.count == 1 {
      guard case let .functionCall(f) = items[0] else { Issue.record("Unexpected"); return }
      #expect(f.status == .incomplete)
    } else {
      #expect(items.isEmpty)
    }
  }
}

@Suite("MistralParser — dispatch")
struct MistralDispatchTests {
  @Test
  func `ResponseFormat.mistral.makeParser returns a working MistralParser`() {
    let parser = ResponseFormat.mistral.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }
}

@Suite("MistralParser — adversarial ports")
struct MistralAdversarialTests {
  // H11: vLLM argument_before_name and argument_before_name_and_name_in_argument
  // (test_mistral_tool_parser.py:644-673).
  @Test
  func `JSON-array with arguments key before name key extracts correctly`() throws {
    let input = #"[TOOL_CALLS] [{"arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]"#
    var parser = MistralParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_current_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "San Francisco")
    #expect(decoded["unit"] as? String == "celsius")
  }

  @Test
  func `JSON-array with name as an argument value AND as the function name resolves to function name`() throws {
    let input = #"[TOOL_CALLS] [{"arguments": {"name": "John Doe"}, "name": "get_age"}]"#
    var parser = MistralParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    // Function name comes from the dict's `name` key (= "get_age"),
    // NOT from the argument value `John Doe`.
    #expect(f.name == "get_age")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["name"] as? String == "John Doe")
  }

  // H14: SGLang test_compact_format_with_leading_text and test_streaming_text_then_tool_call
  // (test_mistral_detector.py:106-110, 198-218).
  @Test
  func `Compact format with leading text: Let me help. [TOOL_CALLS]name[ARGS]{...}`() throws {
    let input = #"Let me help. [TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}"#
    var parser = MistralParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
    // The leading text reaches the consumer as a message.
    let combinedText = messages.compactMap { msg -> String? in
      if case let .outputText(p) = msg.content[0] { return p.text } else { return nil }
    }.joined()
    #expect(combinedText.contains("Let me help."))
  }

  @Test
  func `Streaming compact format with leading text: chunks Sure! , [TOOL_CALLS]name, [ARGS]{...}`() throws {
    let chunks = [
      "Sure! ",
      "[TOOL_CALLS]get_weather",
      #"[ARGS]{"city": "Tokyo"}"#,
    ]
    var parser = MistralParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
    // Accumulated message text includes the leading chunk.
    let combinedText = messages.compactMap { msg -> String? in
      if case let .outputText(p) = msg.content[0] { return p.text } else { return nil }
    }.joined()
    #expect(combinedText == "Sure! ")
  }
}

@Suite("MistralParser — malformed tool-call payload at EOS")
struct MistralMalformedPayloadTests {
  @Test
  func `Truncated JSON-array after [TOOL_CALLS] surfaces as message content`() {
    let input = #"Here goes: [TOOL_CALLS] [{"name": "f", "arguments""#
    var parser = MistralParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    for item in items {
      if case .functionCall = item {
        Issue.record("Truncated array should not produce a function call")
      }
    }
    let combined = items.compactMap { item -> String? in
      guard case let .message(m) = item, case let .outputText(p) = m.content[0] else {
        return nil
      }
      return p.text
    }.joined()
    #expect(combined.contains("Here goes:"))
    #expect(combined.contains("[TOOL_CALLS]"))
    #expect(combined.contains("\"name\": \"f\""))
  }
}
