// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// Cohere-specific behavior: marker-driven state machine with reasoning,
// tool-action, and grounded-answer regions; inline `<co>` citations in
// grounded answers; JSON-array tool calls inside `<|START_ACTION|>` /
// `<|END_ACTION|>`. Tests are ported from melody's Rust unit tests in
// `src/parsing/filter.rs` and `src/parsing/citations_filter.rs`.

// MARK: Helpers

private func cmd3Parser() -> CohereParser {
  CohereParser(variant: .cmd3)
}

private func cmd4Parser(initialState: CohereParser.InitialState = .reasoning) -> CohereParser {
  CohereParser(variant: .cmd4, initialState: initialState)
}

private func runFull(_ text: String, parser: CohereParser) -> [ResponseOutputItem] {
  var p = parser
  let events = p.process(ParserInput(text: text)) + p.finalize()
  return accumulateItems(from: events)
}

private func runStream(_ chunks: [String], parser: CohereParser) -> [ResponseOutputItem] {
  var p = parser
  var events: [ResponseStreamingEvent] = []
  for c in chunks {
    events += p.process(ParserInput(text: c))
  }
  events += p.finalize()
  return accumulateItems(from: events)
}

private func messageText(_ items: [ResponseOutputItem]) -> String? {
  for item in items {
    if case let .message(m) = item, case let .outputText(t) = m.content.first {
      return t.text
    }
  }
  return nil
}

private func reasoningText(_ items: [ResponseOutputItem]) -> String? {
  for item in items {
    if case let .reasoning(r) = item, case let .reasoningText(t) = r.content.first {
      return t.text
    }
  }
  return nil
}

private func messageAnnotations(_ items: [ResponseOutputItem]) -> [ResponseOutputText.Annotation] {
  for item in items {
    if case let .message(m) = item, case let .outputText(t) = m.content.first {
      return t.annotations
    }
  }
  return []
}

private func toolCalls(_ items: [ResponseOutputItem]) -> [ResponseFunctionToolCall] {
  items.compactMap { item in
    if case let .functionCall(f) = item { return f }
    return nil
  }
}

// MARK: State-machine tests (ported from filter.rs)

@Suite("CohereParser — state machine (cmd3)")
struct CohereCmd3StateTests {
  @Test
  func `Thinking and response emit reasoning then message`() {
    let text =
      "<|START_THINKING|>Let me think about this.<|END_THINKING|>"
        + "<|START_RESPONSE|>Here is the answer.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "Let me think about this.")
    #expect(messageText(items) == "Here is the answer.")
  }

  @Test
  func `Response only (no thinking) emits message`() {
    let text = "<|START_RESPONSE|>Just a response.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == nil)
    #expect(messageText(items) == "Just a response.")
  }

  @Test
  func `Thinking only (no response) emits reasoning`() {
    let text = "<|START_THINKING|>I am thinking.<|END_THINKING|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "I am thinking.")
    #expect(messageText(items) == nil)
  }

  @Test
  func `Plain text without markers becomes message content`() {
    let text = "Hello world, no special tokens here."
    let items = runFull(text, parser: cmd3Parser())
    #expect(messageText(items) == "Hello world, no special tokens here.")
    #expect(reasoningText(items) == nil)
  }

  @Test
  func `Preamble text before the first marker is treated as content`() {
    let text = "Preamble text <|START_RESPONSE|>Response.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    let content = messageText(items) ?? ""
    #expect(content.contains("Preamble text"))
    #expect(content.contains("Response."))
  }

  @Test
  func `Empty string yields no items`() {
    let items = runFull("", parser: cmd3Parser())
    #expect(items.isEmpty)
  }

  @Test
  func `Empty thinking block emits message only`() {
    let text = "<|START_THINKING|><|END_THINKING|><|START_RESPONSE|>Content.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(messageText(items) == "Content.")
  }

  @Test
  func `Empty response block emits reasoning only`() {
    let text = "<|START_THINKING|>Thinking.<|END_THINKING|><|START_RESPONSE|><|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "Thinking.")
  }

  @Test
  func `Adjacent special tokens yield no content`() {
    let text = "<|START_THINKING|><|END_THINKING|><|START_RESPONSE|><|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == nil)
    #expect(messageText(items) == nil)
  }

  @Test
  func `Special-token-like substrings are not consumed`() {
    let text = "<|START_RESPONSE|>The tag <|NOT_A_TOKEN|> is not special.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    let content = messageText(items) ?? ""
    #expect(content.contains("<|NOT_A_TOKEN|>"))
  }

  @Test
  func `UTF8 multibyte content roundtrips`() {
    let text =
      "<|START_THINKING|>Réflexion 🤔 über Ñoño<|END_THINKING|>"
        + "<|START_RESPONSE|>Ответ: café ☕<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "Réflexion 🤔 über Ñoño")
    #expect(messageText(items) == "Ответ: café ☕")
  }

  @Test
  func `Long content survives the buffer cycle`() {
    let filler = String(repeating: "word ", count: 10000)
    let text =
      "<|START_THINKING|>\(filler)<|END_THINKING|>"
        + "<|START_RESPONSE|>\(filler)<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    let r = reasoningText(items) ?? ""
    let m = messageText(items) ?? ""
    #expect(r.contains("word"))
    #expect(m.contains("word"))
    #expect(r.count > 40000)
    #expect(m.count > 40000)
  }

  @Test
  func `Text after END_RESPONSE is dropped`() {
    let text = "<|START_RESPONSE|>Inside.<|END_RESPONSE|>After."
    let items = runFull(text, parser: cmd3Parser())
    #expect(messageText(items) == "Inside.")
  }

  @Test
  func `Repeated thinking and response blocks emit each region as its own item`() {
    let text =
      "<|START_THINKING|>First thought.<|END_THINKING|>"
        + "<|START_RESPONSE|>Middle answer.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "First thought.")
    #expect(messageText(items) == "Middle answer.")
  }
}

// MARK: Whitespace trimming

@Suite("CohereParser — whitespace trimming")
struct CohereWhitespaceTrimTests {
  @Test
  func `Trailing whitespace before END_THINKING is dropped`() {
    // Mirrors melody's `right_trimmed = true` cmd3 preset: trailing
    // whitespace held in the buffer is drained alongside the marker,
    // so the reasoning text ends at the last non-whitespace character.
    let text = "<|START_THINKING|> think <|END_THINKING|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "think")
  }

  @Test
  func `Trailing whitespace before END_RESPONSE is dropped`() {
    let text = "<|START_RESPONSE|>answer  <|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(messageText(items) == "answer")
  }

  @Test
  func `EOS without a closing marker drops trailing whitespace`() {
    // Finalizing a buffer that ends with whitespace mirrors melody's
    // `process_full_text` flush: the trailing run is treated as
    // incidental and dropped rather than emitted as a final delta.
    let items = runFull("answer ", parser: cmd3Parser())
    #expect(messageText(items) == "answer")
  }

  @Test
  func `Trailing whitespace held across chunks reattaches to following content`() {
    // Mid-stream trailing whitespace must be held back so that text
    // straddling chunk boundaries reassembles correctly.
    let items = runStream(
      ["<|START_THINKING|>think ", " more<|END_THINKING|>"],
      parser: cmd3Parser(),
    )
    #expect(reasoningText(items) == "think  more")
  }
}

// MARK: cmd4-specific state machine

@Suite("CohereParser — state machine (cmd4)")
struct CohereCmd4StateTests {
  @Test
  func `Thinking and START_TEXT emit reasoning then message`() {
    let text =
      "<|START_THINKING|>Step 1: analyze.<|END_THINKING|>"
        + "<|START_TEXT|>The result is 42.<|END_TEXT|>"
    let items = runFull(text, parser: cmd4Parser())
    #expect(reasoningText(items) == "Step 1: analyze.")
    #expect(messageText(items) == "The result is 42.")
  }

  @Test
  func `Implicit reasoning start (no opening marker) treats prefix as reasoning`() {
    let text =
      "Plan first.<|END_THINKING|>"
        + "<|START_TEXT|>Final answer.<|END_TEXT|>"
    let items = runFull(text, parser: cmd4Parser(initialState: .reasoning))
    #expect(reasoningText(items) == "Plan first.")
    #expect(messageText(items) == "Final answer.")
  }

  @Test
  func `cmd4 still recognizes START_RESPONSE`() {
    let text =
      "<|START_THINKING|>Think.<|END_THINKING|>"
        + "<|START_RESPONSE|>Response text.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd4Parser())
    #expect(reasoningText(items) == "Think.")
    #expect(messageText(items) == "Response text.")
  }

  @Test
  func `cmd4 with empty implicit thinking prefix emits only content`() {
    let text =
      "<|START_THINKING|><|END_THINKING|>"
        + "<|START_TEXT|>Content.<|END_TEXT|>"
    let items = runFull(text, parser: cmd4Parser(initialState: .reasoning))
    #expect(reasoningText(items) == nil)
    #expect(messageText(items) == "Content.")
  }
}

// MARK: Tool calls

@Suite("CohereParser — tool calls")
struct CohereToolCallTests {
  @Test
  func `Reasoning then tool action emits reasoning and function call`() throws {
    let text =
      "<|START_THINKING|>I should search.<|END_THINKING|>"
        + "<|START_ACTION|>\n[{\"tool_call_id\": \"call_0\", \"tool_name\": \"web_search\", \"parameters\": {\"query\": \"test\"}}]<|END_ACTION|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "I should search.")
    let calls = toolCalls(items)
    try #require(calls.count == 1)
    #expect(calls[0].name == "web_search")
    let data = try #require(calls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["query"] as? String == "test")
  }

  @Test
  func `cmd4 implicit reasoning then tool action`() throws {
    let text =
      "I should search.<|END_THINKING|>"
        + "<|START_ACTION|>\n[{\"tool_call_id\": \"call_0\", \"tool_name\": \"web_search\", \"parameters\": {\"query\": \"test\"}}]<|END_ACTION|>"
    let items = runFull(text, parser: cmd4Parser(initialState: .reasoning))
    #expect(reasoningText(items) == "I should search.")
    let calls = toolCalls(items)
    try #require(calls.count == 1)
    #expect(calls[0].name == "web_search")
  }

  @Test
  func `Multiple tool calls in one action block`() throws {
    let body =
      "[{\"tool_call_id\": \"0\", \"tool_name\": \"a\", \"parameters\": {\"x\": 1}}, "
        + "{\"tool_call_id\": \"1\", \"tool_name\": \"b\", \"parameters\": {\"y\": 2}}]"
    let text = "<|START_ACTION|>\(body)<|END_ACTION|>"
    let items = runFull(text, parser: cmd3Parser())
    let calls = toolCalls(items)
    try #require(calls.count == 2)
    #expect(calls[0].name == "a")
    #expect(calls[1].name == "b")
  }

  @Test
  func `tool_call_id is preserved verbatim on call_id when emitted by model`() {
    // Mirrors melody / vLLM: the model-emitted `tool_call_id` field
    // flows into `call_id` unchanged. Positional `"0"` stays `"0"`;
    // a model-supplied `"call_42"` stays `"call_42"`. The library's
    // `call_…` prefix convention only applies to freshly minted IDs.
    let body =
      "[{\"tool_call_id\": \"0\", \"tool_name\": \"a\", \"parameters\": {}}, "
        + "{\"tool_call_id\": \"call_42\", \"tool_name\": \"b\", \"parameters\": {}}]"
    let text = "<|START_ACTION|>\(body)<|END_ACTION|>"
    let items = runFull(text, parser: cmd3Parser())
    let calls = toolCalls(items)
    #expect(calls.count == 2)
    #expect(calls.first?.callId == "0")
    #expect(calls.last?.callId == "call_42")
  }

  @Test
  func `Omitted tool_call_id mints a fresh call_ prefix`() {
    // No `tool_call_id` field in the wire payload — the parser falls
    // back to `IDFactory.make(.callId)`, which mints the `call_…`
    // shape every other parser in this codebase uses.
    let body = "[{\"tool_name\": \"foo\", \"parameters\": {}}]"
    let text = "<|START_ACTION|>\(body)<|END_ACTION|>"
    let items = runFull(text, parser: cmd3Parser())
    let calls = toolCalls(items)
    #expect(calls.first?.callId.hasPrefix("call_") == true)
  }

  @Test
  func `Tool action arrives mid-chunk with end token in same chunk`() throws {
    // Regression test for the speculative-decoding shape — when
    // `<|END_THINKING|><|START_ACTION|>` arrives as one chunk, both
    // markers must be consumed cleanly.
    var parser = cmd4Parser()
    let events1 = parser.process(ParserInput(text: "<|START_THINKING|>"))
    let events2 = parser.process(ParserInput(text: " think "))
    let events3 = parser.process(ParserInput(text: "<|END_THINKING|><|START_ACTION|>"))
    let events4 = parser.process(ParserInput(text: "[{\"tool_call_id\": \"0\", \"tool_name\": \"foo\", \"parameters\": {\"q\": \"x\"}}]"))
    let events5 = parser.process(ParserInput(text: "<|END_ACTION|>"))
    let events6 = parser.finalize()
    let items = accumulateItems(from: events1 + events2 + events3 + events4 + events5 + events6)
    let calls = toolCalls(items)
    try #require(calls.count == 1)
    #expect(calls[0].name == "foo")
  }
}

// MARK: Citations

@Suite("CohereParser — citations")
struct CohereCitationTests {
  @Test
  func `cmd3 bare citation emits span text and a sourced annotation`() throws {
    let text =
      "<|START_RESPONSE|>The sky is <co>blue</co: 0:[0]>.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    let content = try #require(messageText(items))
    #expect(content == "The sky is blue.")
    let ann = messageAnnotations(items)
    try #require(ann.count == 1)
    guard case let .cohereToolResultCitation(tci, rids, start, end) = ann[0] else {
      Issue.record("Expected cohereToolResultCitation annotation, got \(ann[0])")
      return
    }
    #expect(tci == 0)
    #expect(rids == [0])
    // "The sky is " is 11 UTF-16 units; "blue" follows.
    #expect(start == 11)
    #expect(end == 15)
  }

  @Test
  func `cmd3 citation references multiple tool calls`() throws {
    let text =
      "<|START_RESPONSE|>Here is a <co>citation</co: 0:[1,2],1:[0]>.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(messageText(items) == "Here is a citation.")
    let ann = messageAnnotations(items)
    try #require(ann.count == 2)
    guard case let .cohereToolResultCitation(tci0, rids0, _, _) = ann[0],
          case let .cohereToolResultCitation(tci1, rids1, _, _) = ann[1]
    else {
      Issue.record("Expected cohereToolResultCitation annotations")
      return
    }
    #expect(tci0 == 0)
    #expect(rids0 == [1, 2])
    #expect(tci1 == 1)
    #expect(rids1 == [0])
  }

  @Test
  func `Legacy open form still produces the span text`() throws {
    // melody's `test_process_full_text_citations_in_response`: the
    // legacy `<co: 0>` open shape is accepted and "blue" reaches the
    // content stream. cmd3 parsing of the close-tag content `0`
    // yields an empty source list; the parser emits a single
    // annotation with `toolCallIndex = 0` and empty result indices
    // so callers can still observe the citation surface.
    let text = "<|START_RESPONSE|>The sky is <co: 0>blue</co: 0>.<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    let content = try #require(messageText(items))
    #expect(content.contains("blue"))
    let ann = messageAnnotations(items)
    try #require(ann.count == 1)
    if case let .cohereToolResultCitation(_, _, start, end) = ann[0] {
      #expect(end - start == "blue".utf16.count)
    } else {
      Issue.record("Expected cohereToolResultCitation annotation")
    }
  }

  @Test
  func `Multiple citations in one response`() {
    let text =
      "<|START_RESPONSE|>hello <co>foo</co: 0:[2,1]> hi <co>barber</co: 0:[0]><|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(messageText(items) == "hello foo hi barber")
    let ann = messageAnnotations(items)
    #expect(ann.count == 2)
  }

  @Test
  func `Citation span across chunks completes when close arrives`() {
    let chunks = [
      "<|START_RESPONSE|>",
      "pre <co>ci",
      "tation</co: 0:[0]>",
      "<|END_RESPONSE|>",
    ]
    let items = runStream(chunks, parser: cmd3Parser())
    #expect(messageText(items) == "pre citation")
    #expect(messageAnnotations(items).count == 1)
  }

  @Test
  func `Citation markers inside reasoning are stripped but span text survives`() {
    // melody's `test_citation_start_in_thinking_bug` — a malformed
    // `<co>` open inside reasoning that never receives a closing
    // `</co:` is stripped (the bare-open path) and a fully-formed
    // citation inside `<|START_RESPONSE|>` emits as message content
    // with the span preserved.
    let text =
      "<|START_THINKING|>I will use some <co> tags to make citations<|END_THINKING|>"
        + "<|START_RESPONSE|>here is a <co>citation</co: 0:[0]>!!!<|END_RESPONSE|>"
    let items = runFull(text, parser: cmd3Parser())
    #expect(reasoningText(items) == "I will use some  tags to make citations")
    #expect(messageText(items) == "here is a citation!!!")
  }
}

// MARK: Streaming reconstruction

@Suite("CohereParser — streaming reconstruction")
struct CohereStreamingReconstructionTests {
  @Test
  func `cmd3 thinking and response match across streaming and one-shot paths`() {
    let text =
      "<|START_THINKING|>Think hard.<|END_THINKING|>"
        + "<|START_RESPONSE|>The answer is 7.<|END_RESPONSE|>"
    assertStreamingReconstruction(text, parser: { cmd3Parser() })
  }

  @Test
  func `cmd4 thinking and text match across streaming and one-shot paths`() {
    let text =
      "<|START_THINKING|>Plan: step 1, step 2.<|END_THINKING|>"
        + "<|START_TEXT|>Final result here.<|END_TEXT|>"
    assertStreamingReconstruction(text, parser: { cmd4Parser() })
  }

  @Test
  func `Plain text matches`() {
    let text = "Just plain text without any markers."
    assertStreamingReconstruction(text, parser: { cmd3Parser() })
  }

  @Test
  func `UTF-8 multibyte content matches across both paths`() {
    let text =
      "<|START_THINKING|>日本語テスト<|END_THINKING|>"
        + "<|START_RESPONSE|>中文回答<|END_RESPONSE|>"
    assertStreamingReconstruction(text, parser: { cmd3Parser() })
  }

  @Test
  func `Tool action stream reconstructs`() {
    let text =
      "<|START_THINKING|>plan<|END_THINKING|>"
        + "<|START_ACTION|>[{\"tool_call_id\":\"0\",\"tool_name\":\"foo\",\"parameters\":{\"q\":\"x\"}}]<|END_ACTION|>"
    assertStreamingReconstruction(text, parser: { cmd3Parser() })
  }
}

// MARK: Dispatch

@Suite("CohereParser — format dispatch")
struct CohereFormatDispatchTests {
  @Test(arguments: [
    "c4ai-command-r7b-12-2024",
    "CohereForAI-c4ai-command-r7b-12-2024",
    "command-r7b",
    "command-a-reasoning-08-2025",
    "CohereLabs-command-a-reasoning-08-2025",
  ])
  func `Cohere checkpoint names route to cohereCmd3`(name: String) {
    let resolved = ResponseFormat.infer(
      modelName: name,
      modelType: "",
      modelConfig: [:],
    )
    #expect(resolved == .cohereCmd3)
  }

  @Test
  func `model_type cohere2 routes to cohereCmd3`() {
    let resolved = ResponseFormat.infer(
      modelName: "",
      modelType: "cohere2",
      modelConfig: [:],
    )
    #expect(resolved == .cohereCmd3)
  }

  @Test
  func `model_type cohere2_vision routes to cohereCmd4`() {
    let resolved = ResponseFormat.infer(
      modelName: "",
      modelType: "cohere2_vision",
      modelConfig: [:],
    )
    #expect(resolved == .cohereCmd4)
  }

  @Test
  func `model_type cohere2_moe routes to cohereCmd3`() {
    let resolved = ResponseFormat.infer(
      modelName: "",
      modelType: "cohere2_moe",
      modelConfig: [:],
    )
    #expect(resolved == .cohereCmd3)
  }

  @Test
  func `Original cohere model_type does not resolve to a Cohere parser`() {
    // `CohereForCausalLM` (text-marker `Action:` / `Grounded answer:`
    // format) is deliberately out of scope; the inferer must fall
    // through to nil rather than routing to a Cohere parser that
    // doesn't speak that wire shape.
    let resolved = ResponseFormat.infer(
      modelName: "",
      modelType: "cohere",
      modelConfig: [:],
    )
    #expect(resolved == nil)
  }
}
