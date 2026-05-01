// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("KimiK2Parser — basics")
struct KimiK2BasicsTests {
  @Test
  func `Plain text emits a single message`() {
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: "hello there")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Single tool call with functions. prefix`() throws {
    let input = #"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>"#
    var parser = KimiK2Parser()
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
  func `callId preserves wire-format functions.NAME:INDEX header`() {
    let input = #"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>"#
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    // The Kimi K2 chat template inserts `tool_call.id` verbatim, so the
    // callId must round-trip the wire header for re-rendered history to
    // match the format the model was trained on.
    #expect(f.callId == "functions.get_weather:0")
  }

  @Test
  func `callId without functions. prefix is preserved verbatim`() {
    let input = #"<|tool_calls_section_begin|><|tool_call_begin|>get_weather:3<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"#
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.callId == "get_weather:3")
  }

  @Test
  func `callId across multiple tool calls preserves each wire index`() {
    let input = """
    <|tool_calls_section_begin|>\
    <|tool_call_begin|>functions.f1:0<|tool_call_argument_begin|>{}<|tool_call_end|>\
    <|tool_call_begin|>functions.f2:1<|tool_call_argument_begin|>{"x": 1}<|tool_call_end|>\
    <|tool_calls_section_end|>
    """
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.callId == "functions.f1:0")
    #expect(b.callId == "functions.f2:1")
  }

  @Test
  func `callId survives chunk boundary inside the second call's header`() {
    // Split the buffer mid-header on the second call: the prefix
    // `functions.f2` arrives before the `:1<|tool_call_argument_begin|>`
    // tail. parseFunctionId returns false on the first scan (no
    // argBegin found), the placeholder sits in OpenToolCall.callId,
    // and the second chunk completes the header. This locks in the
    // invariant that the placeholder is overwritten before any
    // outputItemAdded event fires for the second call.
    let chunk1 = """
    <|tool_calls_section_begin|>\
    <|tool_call_begin|>functions.f1:0<|tool_call_argument_begin|>{}<|tool_call_end|>\
    <|tool_call_begin|>functions.f2
    """
    let chunk2 = """
    :1<|tool_call_argument_begin|>{"x": 1}<|tool_call_end|>\
    <|tool_calls_section_end|>
    """
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: chunk1))
      + parser.process(ParserInput(text: chunk2))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.callId == "functions.f1:0")
    #expect(b.callId == "functions.f2:1")
  }

  @Test
  func `Multiple tool calls in one section`() {
    let input = """
    <|tool_calls_section_begin|>\
    <|tool_call_begin|>functions.f1:0<|tool_call_argument_begin|>{}<|tool_call_end|>\
    <|tool_call_begin|>functions.f2:1<|tool_call_argument_begin|>{"x": 1}<|tool_call_end|>\
    <|tool_calls_section_end|>
    """
    var parser = KimiK2Parser()
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
  func `Hyphenated function names (MCP-style) are preserved`() {
    let input = #"<|tool_calls_section_begin|><|tool_call_begin|>functions.mcp__portal__search-documents:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"#
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "mcp__portal__search-documents")
  }

  @Test
  func `Char-by-char streaming yields the same items as single-shot`() {
    let input = #"<|tool_calls_section_begin|><|tool_call_begin|>functions.fn:0<|tool_call_argument_begin|>{"x": 1}<|tool_call_end|><|tool_calls_section_end|>"#

    var streaming = KimiK2Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = KimiK2Parser()
    let oneShotItems = accumulateItems(from:
      oneShot.process(ParserInput(text: input)) + oneShot.finalize())

    #expect(streamingItems.count == oneShotItems.count)
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.functionCall(sf), .functionCall(of)):
          #expect(sf.name == of.name)
          #expect(sf.arguments == of.arguments)
          // Wire-extracted callId must survive arbitrary chunk
          // boundaries. The placeholder mint at OpenToolCall init is
          // overwritten only on the parseFunctionId-success path; this
          // assertion locks in the invariant that the placeholder
          // never escapes into a public event under streaming.
          #expect(sf.callId == of.callId)
        case let (.message(sm), .message(om)):
          #expect(sm.content == om.content)
        default:
          Issue.record("Item kinds differ"); return
      }
    }
  }

  @Test
  func `Dispatch via ResponseFormat.kimiK2.makeParser`() {
    let parser = ResponseFormat.kimiK2.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Name prefix kimi-k2 routes to .kimiK2`() {
    let format = ResponseFormat.infer(modelName: "kimi-k2-instruct", modelType: "", modelConfig: [:])
    #expect(format == .kimiK2)
  }
}

@Suite("KimiK2Parser — adversarial ports")
struct KimiK2AdversarialTests {
  private static let sectionBegin = "<|tool_calls_section_begin|>"
  private static let sectionEnd = "<|tool_calls_section_end|>"
  private static let callBegin = "<|tool_call_begin|>"
  private static let callEnd = "<|tool_call_end|>"
  private static let argBegin = "<|tool_call_argument_begin|>"

  // H3: vLLM test_invalid_funcall_id_skipped
  // (test_kimi_k2_tool_parser.py:165-176).
  @Test
  func `Tool call with malformed ID (no :digit suffix) is skipped`() throws {
    let input = (
      "Help. "
        + Self.sectionBegin
        + Self.callBegin + #"functions.invalid.0 "# + Self.argBegin + #"{"city": "Beijing"}"# + Self.callEnd
        + Self.callBegin + #"functions.valid:1 "# + Self.argBegin + #"{"city": "Shanghai"}"# + Self.callEnd
        + Self.sectionEnd,
    )
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "valid")
    // Surviving call's wire ID must be preserved verbatim — dropping a
    // sibling call must not leak a placeholder into the surviving one.
    #expect(toolCalls[0].callId == "functions.valid:1")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Shanghai")
  }

  @Test
  func `Tool call with non-word name chars is rejected (mirrors sglang regex)`() {
    // sglang's `tool_call_id_regex` enforces `[\w.\-]+` for the name
    // body. A name containing a space (or other non-allowed char)
    // must be dropped, not forwarded as a function name.
    let input = (
      Self.sectionBegin
        + Self.callBegin + #"functions.bad name:0 "# + Self.argBegin + #"{}"# + Self.callEnd
        + Self.callBegin + #"functions.good:1 "# + Self.argBegin + #"{"x": 1}"# + Self.callEnd
        + Self.sectionEnd,
    )
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "good")
    #expect(toolCalls[0].callId == "functions.good:1")
  }

  @Test
  func `Stray Kimi tokens outside a section are stripped from message content`() {
    // No `<|tool_calls_section_begin|>`; a stray inner marker shouldn't
    // leak into the content the consumer sees.
    let input = "Before <|tool_call_begin|> middle <|tool_call_end|> after"
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let combined = items.compactMap { item -> String? in
      guard case let .message(m) = item, case let .outputText(p) = m.content[0] else {
        return nil
      }
      return p.text
    }.joined()
    #expect(!combined.contains("<|tool_call_begin|>"))
    #expect(!combined.contains("<|tool_call_end|>"))
    #expect(combined.contains("Before"))
    #expect(combined.contains("after"))
  }

  @Test
  func `Malformed ID with < inside the discarded args does not hang`() throws {
    // After a malformed function ID, the args bytes are scanned for the
    // next marker. If those bytes contain a stray `<` that doesn't start
    // a Kimi marker (e.g., HTML inside a JSON string), the parser must
    // still advance past it.
    let input = (
      Self.sectionBegin
        + Self.callBegin + #"functions.bad.0 "# + Self.argBegin
        + #"{"html": "<div>x</div>"}"# + Self.callEnd
        + Self.callBegin + #"functions.good:1 "# + Self.argBegin
        + #"{"city": "Tokyo"}"# + Self.callEnd
        + Self.sectionEnd,
    )
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "good")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }

  // H4: vLLM test_content_after_tool_section
  // (test_kimi_k2_tool_parser.py:435-458).
  //
  // SGLang/vLLM drop trailing text after `<|tool_calls_section_end|>`.
  // We emit it as a message — more permissive (if a model emits content
  // after a tool call, the user should see it). Pinning our behavior.
  // The key invariant from the vLLM test still holds: no structural
  // markers leak into the content.
  @Test
  func `Content after section_end is emitted as a message; no marker leakage (divergence)`() throws {
    let input = (
      "Before. "
        + Self.sectionBegin
        + Self.callBegin + #"functions.get_weather:0 "# + Self.argBegin + #"{"city": "Tokyo"} "# + Self.callEnd
        + Self.sectionEnd
        + " After tools.",
    )
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
    // No structural marker leaks into any message content.
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    for msg in messages {
      guard case let .outputText(part) = msg.content[0] else { continue }
      for marker in [Self.sectionBegin, Self.sectionEnd, Self.callBegin, Self.callEnd, Self.argBegin] {
        #expect(!part.text.contains(marker))
      }
    }
  }

  // H13: SGLang test_tool_call_inside_think_without_close_tag
  // (test_kimik2_detector.py:320-347).
  //
  // SGLang's KimiK2ReasoningDetector splits `<think>...` from the tool
  // call markers. Our KimiK2Parser has no reasoning support — `<think>`
  // is not recognized. The fixture verifies the tool call is still
  // extracted correctly and that no structural tool markers leak into
  // any emitted message content.
  @Test
  func `Tool markers inside <think> without </think> still extract the tool call (no reasoning support)`() throws {
    let input = (
      "<think>Let me read this file..."
        + Self.sectionBegin
        + Self.callBegin + #"functions.ReadFile:0"# + Self.argBegin + #"{"path": "/test.py"}"# + Self.callEnd
        + Self.sectionEnd,
    )
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "ReadFile")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["path"] as? String == "/test.py")
    // No tool-call structural markers leak into any message content.
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    for msg in messages {
      guard case let .outputText(part) = msg.content[0] else { continue }
      for marker in [Self.sectionBegin, Self.sectionEnd, Self.callBegin, Self.callEnd, Self.argBegin] {
        #expect(!part.text.contains(marker))
      }
    }
  }

  // M7: SGLang test_streaming_partial_marker_buffering
  // (test_kimik2_detector.py:442-464).
  //
  // SGLang's reasoning detector buffers partial structural markers across
  // chunk boundaries. Our parser uses `partialOverlap` in `emitNormalText`
  // to do the same. Verify the partial `<|tool` does not leak as message
  // content before the marker completes.
  @Test
  func `Partial structural marker <|tool across chunks does not leak into content`() {
    let chunks = [
      "some reasoning",
      "<|tool",
      "_calls_section_begin|>rest",
    ]
    var parser = KimiK2Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let deltas = events.compactMap { ev -> String? in
      if case let .outputTextDelta(e) = ev { return e.delta } else { return nil }
    }
    let combined = deltas.joined()
    // Reasoning text reaches the consumer.
    #expect(combined.contains("some reasoning"))
    // The partial structural marker does not leak.
    #expect(!combined.contains("<|tool"))
  }
}

@Suite("KimiK2Parser — reasoning (Kimi-K2-Thinking)")
struct KimiK2ReasoningTests {
  // Mirrors vLLM's `test_extract_reasoning_with_think_tags`.
  @Test
  func `Explicit <think>...</think> then content`() {
    let input = "<think>step by step reasoning</think>final answer"
    var parser = KimiK2Parser(initialState: .reasoning)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("expected reasoning"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "step by step reasoning")
    #expect(r.status == .completed)
    guard case let .message(m) = items[1] else { Issue.record("expected message"); return }
    guard case let .outputText(mPart) = m.content[0] else { Issue.record(""); return }
    #expect(mPart.text == "final answer")
  }

  // Mirrors vLLM's `test_extract_reasoning_empty_thinking` (reasoning="",
  // content="final answer"). In Swift's event-based interface we elide
  // the empty reasoning item entirely – no chars between opener and
  // terminator means no reasoning to surface – so only the message
  // item appears.
  @Test
  func `Empty <think></think> emits only a message (no empty reasoning item)`() {
    let input = "<think></think>final answer"
    var parser = KimiK2Parser(initialState: .reasoning)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let hasReasoning = items.contains { if case .reasoning = $0 { true } else { false } }
    #expect(!hasReasoning, "Empty reasoning block should not surface a reasoning item")
    guard case let .message(m) = items.last else { Issue.record("expected message"); return }
    guard case let .outputText(mPart) = m.content[0] else { Issue.record(""); return }
    #expect(mPart.text == "final answer")
  }

  // Mirrors vLLM's `test_extract_reasoning_implicit_start`.
  @Test
  func `Implicit reasoning (no <think> opener) without terminator stays as incomplete reasoning`() {
    let input = "implicit reasoning with no tags"
    var parser = KimiK2Parser(initialState: .reasoning)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("expected reasoning"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "implicit reasoning with no tags")
    #expect(r.status == .incomplete)
  }

  // Mirrors vLLM's `test_extract_reasoning_tool_section_ends_reasoning`.
  @Test
  func `<|tool_calls_section_begin|> implicitly ends reasoning and opens tool call`() throws {
    let input = #"some reasoning<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{"x":1}<|tool_call_end|><|tool_calls_section_end|>"#
    var parser = KimiK2Parser(initialState: .reasoning)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("expected reasoning first"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "some reasoning")
    #expect(r.status == .completed)
    guard case let .functionCall(f) = items[1] else { Issue.record("expected function call"); return }
    #expect(f.name == "f")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }

  @Test
  func `Default initialState .normal skips reasoning entirely`() {
    let input = "<think>this is reasoning</think>but treated as message"
    var parser = KimiK2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let hasReasoning = items.contains { if case .reasoning = $0 { true } else { false } }
    #expect(!hasReasoning, "Default Kimi K2 (Instruct) must not emit reasoning items")
  }

  @Test
  func `Char-by-char streaming through <think>...</think> matches single-shot`() {
    let input = "<think>step one\nstep two</think>final"
    var single = KimiK2Parser(initialState: .reasoning)
    let singleEvents = single.process(ParserInput(text: input)) + single.finalize()
    let singleItems = accumulateItems(from: singleEvents)

    var streamed = KimiK2Parser(initialState: .reasoning)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(singleItems.count == streamedItems.count)
    // Reasoning text is identical across both modes.
    if case let .reasoning(sr) = singleItems[0],
       case let .reasoning(strr) = streamedItems[0]
    {
      #expect(sr.text == strr.text)
      #expect(sr.text == "step one\nstep two")
    } else {
      Issue.record("expected reasoning items in both single and streamed runs")
    }
  }

  @Test
  func `Char-by-char streaming where tool section ends reasoning matches single-shot`() {
    let input = #"reasoning prelude<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"#
    var single = KimiK2Parser(initialState: .reasoning)
    let singleEvents = single.process(ParserInput(text: input)) + single.finalize()
    let singleItems = accumulateItems(from: singleEvents)

    var streamed = KimiK2Parser(initialState: .reasoning)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(singleItems.count == streamedItems.count)
    #expect(singleItems.count == 2)
    if case let .reasoning(sr) = singleItems[0] {
      #expect(sr.text == "reasoning prelude")
    } else {
      Issue.record("expected reasoning at index 0")
    }
    if case let .functionCall(f) = singleItems[1] {
      #expect(f.name == "f")
    } else {
      Issue.record("expected function call at index 1")
    }
  }
}

@Suite("ResponseFormat dispatch — Kimi K2 Thinking")
struct KimiK2ThinkingDispatchTests {
  @Test
  func `Kimi-K2-Thinking name routes to .kimiK2Thinking (longest-prefix)`() {
    let f = ResponseFormat.infer(
      modelName: "moonshotai/Kimi-K2-Thinking",
      modelType: "deepseek_v3",
      modelConfig: [:],
    )
    #expect(f == .kimiK2Thinking)
  }

  @Test
  func `Kimi-K2-Instruct name still routes to .kimiK2`() {
    let f = ResponseFormat.infer(
      modelName: "moonshotai/Kimi-K2-Instruct",
      modelType: "deepseek_v3",
      modelConfig: [:],
    )
    #expect(f == .kimiK2)
  }
}
