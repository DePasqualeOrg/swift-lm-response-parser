// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Fixtures derived from vLLM's `tests/tool_parsers/test_functiongemma_tool_parser.py`.

@Suite("GemmaFunctionCallParser — plain text")
struct GemmaFunctionCallPlainTextTests {
  @Test
  func `Plain text without markers emits a single message`() {
    var parser = GemmaFunctionCallParser()
    let events = parser.process(ParserInput(text: "Hello there.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Hello there.")
  }

  @Test
  func `Empty stream emits nothing`() {
    var parser = GemmaFunctionCallParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("GemmaFunctionCallParser — single tool call")
struct GemmaFunctionCallSingleTests {
  @Test
  func `Single string-arg call`() throws {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:get_weather{location:<escape>San Francisco<escape>,unit:<escape>celsius<escape>}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["location"] as? String == "San Francisco")
    #expect(decoded["unit"] as? String == "celsius")
  }

  @Test
  func `Bare numeric and boolean values`() throws {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:fn{count:42,enabled:true,ratio:3.14}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["count"] as? Int == 42)
    #expect(decoded["enabled"] as? Bool == true)
    #expect(decoded["ratio"] as? Double == 3.14)
  }

  @Test
  func `Escaped JSON literal values decode to their JSON types`() throws {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:fn{count:<escape>42<escape>,enabled:<escape>true<escape>}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["count"] as? Int == 42)
    #expect(decoded["enabled"] as? Bool == true)
  }

  @Test
  func `Bare null value`() throws {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:fn{x:null}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] is NSNull)
  }

  @Test
  func `Empty arguments`() throws {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:refresh{}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "refresh")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded.isEmpty)
  }

  @Test
  func `Tool call IDs follow fc_/call_ prefix convention`() {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:fn{}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
  }
}

@Suite("GemmaFunctionCallParser — multiple tool calls")
struct GemmaFunctionCallMultipleTests {
  @Test
  func `Two concatenated calls without separator`() {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:f1{x:1}<end_function_call><start_function_call>call:f2{y:<escape>hello<escape>}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "f1")
    #expect(toolCalls[1].name == "f2")
  }
}

@Suite("GemmaFunctionCallParser — surrounding text")
struct GemmaFunctionCallSurroundingTests {
  @Test
  func `Leading and trailing text emit as messages around the call`() {
    var parser = GemmaFunctionCallParser()
    let input = "Pre. <start_function_call>call:fn{x:1}<end_function_call> Post."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Three items: pre-message, function call, post-message.
    #expect(items.count == 3)
    guard case let .message(pre) = items[0],
          case let .outputText(preT) = pre.content[0]
    else {
      Issue.record("Expected pre message"); return
    }
    #expect(preT.text == "Pre. ")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected call"); return }
    #expect(f.name == "fn")
    guard case let .message(post) = items[2],
          case let .outputText(postT) = post.content[0]
    else {
      Issue.record("Expected post message"); return
    }
    #expect(postT.text == " Post.")
  }
}

@Suite("GemmaFunctionCallParser — streaming")
struct GemmaFunctionCallStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = "<start_function_call>call:get_weather{location:<escape>Paris<escape>}<end_function_call>"

    var oneShot = GemmaFunctionCallParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = GemmaFunctionCallParser()
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
  func `Split <start_function_call> opener across chunks doesn't leak`() {
    var parser = GemmaFunctionCallParser()
    var events = parser.process(ParserInput(text: "<start_func"))
    events += parser.process(ParserInput(text: "tion_call>call:fn{}<end_function_call>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "fn")
  }
}

@Suite("GemmaFunctionCallParser — malformed input")
struct GemmaFunctionCallMalformedTests {
  @Test
  func `Parseable unclosed envelope at finalize emits incomplete call`() throws {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>call:fn{x:<escape>1<escape>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.status == .incomplete)
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Body without call: prefix surfaces as content`() {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>fn{x:1}<end_function_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }

  @Test
  func `Unparseable unclosed envelope at finalize surfaces as content`() {
    var parser = GemmaFunctionCallParser()
    let input = "<start_function_call>fn{x:1"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
    guard case let .message(m) = items.first,
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected content fallback"); return
    }
    #expect(t.text == input)
  }

  @Test
  func `Unmatched <escape> opener falls through gracefully`() {
    var parser = GemmaFunctionCallParser()
    // Single `<escape>` without a closing pair. The parser bails out
    // of arg parsing for that pair but does not crash; the call may
    // be emitted with the args parsed up to the bad pair, or
    // surfaced as content depending on where the bail-out hits.
    let input = "<start_function_call>call:fn{x:<escape>unclosed}<end_function_call>"
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }
}
