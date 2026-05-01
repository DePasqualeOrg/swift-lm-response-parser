// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("DeepSeekV32Parser — basics")
struct DeepSeekV32BasicsTests {
  @Test
  func `Plain text emits a single message`() {
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: "hello")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record(""); return }
  }

  @Test
  func `Single tool call with XML parameter tags (string typed)`() throws {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="get_weather">"#
        + #"<｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "San Francisco")
  }

  @Test
  func `XML parameter with string=false treats body as JSON literal`() throws {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="get_weather">"#
        + #"<｜DSML｜parameter name="days" string="false">5</｜DSML｜parameter>"#
        + #"<｜DSML｜parameter name="active" string="false">true</｜DSML｜parameter>"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["days"] as? Int == 5)
    #expect(decoded["active"] as? Bool == true)
  }

  @Test
  func `Direct JSON body inside <invoke> is preserved`() throws {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="get_weather">"#
        + #"{"city":"Paris","days":5}"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
    #expect(decoded["days"] as? Int == 5)
  }

  @Test
  func `Multiple invokes in one envelope`() {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="f1"><｜DSML｜parameter name="x" string="false">1</｜DSML｜parameter></｜DSML｜invoke>"#
        + #"<｜DSML｜invoke name="f2"><｜DSML｜parameter name="y" string="false">2</｜DSML｜parameter></｜DSML｜invoke>"#
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
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
  func `Each </｜DSML｜invoke> emits its function call without waiting for envelope close`() {
    // Mirrors vLLM's `_extract_delta_tool_calls` which emits each
    // invoke as soon as its `</｜DSML｜invoke>` arrives — driven by
    // `current_tool_index`, not by the outer envelope close. We feed
    // the envelope incrementally and verify the first invoke's
    // function-call events are visible before the second invoke is
    // even started.
    var parser = DeepSeekV32Parser()
    var events: [ResponseStreamingEvent] = []
    events += parser.process(ParserInput(text:
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="f1"><｜DSML｜parameter name="x" string="false">1</｜DSML｜parameter></｜DSML｜invoke>"#))
    // After the first `</｜DSML｜invoke>` has arrived but before the
    // outer envelope closes, the first function call must already
    // have streamed `output_item.added` and `output_item.done`.
    let f1Done = events.contains { ev in
      if case let .outputItemDone(e) = ev,
         case let .functionCall(f) = e.item,
         f.name == "f1"
      {
        return true
      }
      return false
    }
    #expect(f1Done)

    // Now feed the rest of the envelope.
    events += parser.process(ParserInput(text:
      #"<｜DSML｜invoke name="f2"><｜DSML｜parameter name="y" string="false">2</｜DSML｜parameter></｜DSML｜invoke>"#
        + "</｜DSML｜function_calls>"))
    events += parser.finalize()

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
  func `Char-by-char streaming reconstructs the same items as one-shot`() {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="fn"><｜DSML｜parameter name="k" string="true">v</｜DSML｜parameter></｜DSML｜invoke>"#
        + "</｜DSML｜function_calls>",
    )

    var streaming = DeepSeekV32Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = DeepSeekV32Parser()
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
}

@Suite("DeepSeekV32Parser — dispatch")
struct DeepSeekV32DispatchTests {
  @Test
  func `Dispatch via ResponseFormat.deepseekV32.makeParser`() {
    let parser = ResponseFormat.deepseekV32.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Name prefix deepseek-v3.2 routes to .deepseekV32 (not the V3.1 alias)`() {
    let format = ResponseFormat.infer(modelName: "deepseek-v3.2-base", modelType: "", modelConfig: [:])
    #expect(format == .deepseekV32)
  }
}

@Suite("DeepSeekV32Parser — malformed envelopes")
struct DeepSeekV32MalformedTests {
  // Same defensive choice as V3.1: drop empty-name invokes rather
  // than surface a nameless function_call to consumers.
  @Test
  func `Empty name="" attribute drops the invoke`() {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="">"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.isEmpty)
  }

  @Test
  func `Empty-name invoke followed by valid invoke: only the valid one emits`() {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="">"#
        + "</｜DSML｜invoke>"
        + #"<｜DSML｜invoke name="fn">"#
        + #"<｜DSML｜parameter name="x" string="false">1</｜DSML｜parameter>"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  // Both vLLM and sglang require `string="..."` on every parameter
  // tag — a missing attribute causes their regexes not to match, which
  // drops the parameter. We mirror that behavior.
  @Test
  func `Parameter tag without string= attribute is dropped`() throws {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="fn">"#
        + #"<｜DSML｜parameter name="dropped">5</｜DSML｜parameter>"#
        + #"<｜DSML｜parameter name="kept" string="true">v</｜DSML｜parameter>"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["dropped"] == nil)
    #expect(decoded["kept"] as? String == "v")
  }

  @Test
  func `Single-quoted attribute values are not accepted (matches references)`() {
    let input = (
      "<｜DSML｜function_calls>"
        + "<｜DSML｜invoke name='fn'>"
        + "<｜DSML｜parameter name='x' string='true'>v</｜DSML｜parameter>"
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Single-quoted attributes don't match the regexes in either
    // reference. vLLM/sglang's streaming paths both silently drop
    // malformed envelope contents — neither surfaces a function
    // call nor falls back to emitting the raw text as content. We
    // mirror that behavior.
    #expect(!items.contains { item in
      if case .functionCall = item { return true }
      return false
    })
  }

  @Test
  func `Uppercase attribute names are not accepted (case-sensitive)`() {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke NAME="fn">"#
        + #"<｜DSML｜parameter NAME="x" STRING="true">v</｜DSML｜parameter>"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Uppercase NAME doesn't match → invoke dropped.
    #expect(!items.contains { item in
      if case .functionCall = item { return true }
      return false
    })
  }
}
