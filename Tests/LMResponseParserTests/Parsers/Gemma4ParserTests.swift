// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("Gemma4Parser — reasoning")
struct Gemma4ReasoningTests {
  @Test
  func `Plain text without markers emits a single message`() {
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Reasoning marker emits a reasoning item`() {
    let input = "<|channel>thought\nLet me think.<channel|>The answer."
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "Let me think.")
    guard case let .message(m) = items[1] else { Issue.record("Expected message"); return }
    guard case let .outputText(mPart) = m.content[0] else { Issue.record(""); return }
    #expect(mPart.text == "The answer.")
  }

  @Test
  func `Reasoning without trailing newline still strips the start marker`() {
    let input = "<|channel>thoughtThinking.<channel|>Done."
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0] else { Issue.record(""); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "Thinking.")
  }

  @Test
  func `Initial state .reasoning resumes inside a reasoning block`() {
    var parser = Gemma4Parser(initialState: .reasoning)
    let events = parser.process(ParserInput(text: "continuing<channel|>now answer")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record(""); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "continuing")
  }

  @Test
  func `Truncated reasoning closes as incomplete`() {
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: "<|channel>thoughtPartial reasoning")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record(""); return }
    #expect(r.status == .incomplete)
  }

  // Pins vLLM's `is_reasoning_end` semantics for Gemma 4: a `<|tool_call>`
  // marker is treated as an implicit reasoning end, even without a
  // matching `<channel|>` closer. Mirrors vLLM's
  // `Gemma4ReasoningParser.is_reasoning_end`, which returns True on
  // `<|tool_call>` regardless of whether `<channel|>` was seen.
  @Test
  func `<|tool_call> ends reasoning even without an explicit <channel|> closer`() throws {
    let input = "<|channel>thought\nweighing options<|tool_call>call:f{x:1}<tool_call|>"
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "weighing options")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "f")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["x"] as? Int == 1)
  }
}

@Suite("Gemma4Parser — tool calls")
struct Gemma4ToolCallTests {
  @Test
  func `Single tool call with string argument`() throws {
    let input = #"<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["location"] as? String == "Paris")
  }

  @Test
  func `Mixed arg types: string, int, float, bool, null`() throws {
    let input = #"<|tool_call>call:complex{name:<|"|>Alice<|"|>,age:30,score:9.5,active:true,prev:null}<tool_call|>"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["name"] as? String == "Alice")
    #expect(decoded["age"] as? Int == 30)
    #expect(decoded["score"] as? Double == 9.5)
    #expect(decoded["active"] as? Bool == true)
    #expect(decoded["prev"] is NSNull)
  }

  @Test
  func `Integers exceeding Int64 range emit as JSON numbers, not strings`() {
    let input = #"<|tool_call>call:f{count:100000000000000000000}<tool_call|>"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    // Emitted as a raw JSON number (no surrounding quotes).
    #expect(f.arguments.contains("100000000000000000000"))
    #expect(!f.arguments.contains("\"100000000000000000000\""))
  }

  @Test
  func `Array argument with mixed element types`() throws {
    let input = #"<|tool_call>call:f{items:[1,2,<|"|>three<|"|>]}<tool_call|>"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let arr = decoded["items"] as? [Any]
    #expect(arr?.count == 3)
    #expect(arr?[0] as? Int == 1)
    #expect(arr?[1] as? Int == 2)
    #expect(arr?[2] as? String == "three")
  }

  @Test
  func `Nested object argument`() throws {
    let input = #"<|tool_call>call:f{outer:{inner:42,label:<|"|>x<|"|>}}<tool_call|>"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let outer = decoded["outer"] as? [String: Any]
    #expect(outer?["inner"] as? Int == 42)
    #expect(outer?["label"] as? String == "x")
  }

  @Test
  func `String value containing comma is preserved`() throws {
    let input = #"<|tool_call>call:f{location:<|"|>Paris, France<|"|>}<tool_call|>"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["location"] as? String == "Paris, France")
  }

  @Test
  func `Multiple parallel tool calls`() {
    let input = (
      #"<|tool_call>call:f1{x:1}<tool_call|>"#
        + #"<|tool_call>call:f2{y:2}<tool_call|>"#,
    )
    var parser = Gemma4Parser()
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
  func `Text between complete tool calls stays in output order`() throws {
    let input = (
      #"Hello! <|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>"#
        + #" Let me also check <|tool_call>call:get_time{timezone:<|"|>UTC<|"|>}<tool_call|> done."#,
    )
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)

    #expect(items.count == 5)
    guard case let .message(firstMessage) = items[0],
          case let .functionCall(firstCall) = items[1],
          case let .message(middleMessage) = items[2],
          case let .functionCall(secondCall) = items[3],
          case let .message(finalMessage) = items[4],
          case let .outputText(firstText) = firstMessage.content[0],
          case let .outputText(middleText) = middleMessage.content[0],
          case let .outputText(finalText) = finalMessage.content[0]
    else {
      Issue.record("Expected message, call, message, call, message ordering")
      return
    }

    #expect(firstText.text == "Hello! ")
    #expect(firstCall.name == "get_weather")
    #expect(middleText.text == " Let me also check ")
    #expect(secondCall.name == "get_time")
    #expect(finalText.text == " done.")

    let firstDecodedData = try #require(firstCall.arguments.data(using: .utf8))
    let firstDecoded = try #require(JSONSerialization.jsonObject(with: firstDecodedData) as? [String: Any])
    #expect(firstDecoded["location"] as? String == "Paris")

    let secondDecodedData = try #require(secondCall.arguments.data(using: .utf8))
    let secondDecoded = try #require(JSONSerialization.jsonObject(with: secondDecodedData) as? [String: Any])
    #expect(secondDecoded["timezone"] as? String == "UTC")
  }

  @Test
  func `Empty argument body`() {
    let input = "<|tool_call>call:f{}<tool_call|>"
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.name == "f")
    #expect(f.arguments == "{}")
  }

  // Pin the vLLM `test_empty_value` fixture (`_parse_gemma4_args("key:")
  // == {"key": ""}`) and sglang's matching behavior. A bare key with
  // no value records the empty string rather than dropping the key.
  @Test
  func `key: with no value records key with empty-string value`() throws {
    let input = "<|tool_call>call:f{key:}<tool_call|>"
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["key"] as? String == "")
  }

  // Pins vLLM's `test_unterminated_string` fixture
  // (`_parse_gemma4_args('key:<|"|>unterminated') == {"key": "unterminated"}`).
  // We exercise the closest-equivalent path: a truncated tool call where
  // the body is `key:<|"|>unterminated` — no closing string-delim, no
  // closing `}`, no closing `<tool_call|>`. The parser falls back to
  // taking the rest of the args content as the string value.
  @Test
  func `Unterminated <|"|> string at truncation consumes the rest`() throws {
    let input = #"<|tool_call>call:f{key:<|"|>unterminated"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["key"] as? String == "unterminated")
  }

  @Test
  func `Truncated tool call closes as incomplete`() {
    let input = #"<|tool_call>call:f{key:<|"|>val"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.status == .incomplete)
    #expect(f.name == "f")
  }

  @Test
  func `Reasoning then tool call sequence`() {
    let input = (
      "<|channel>thought\nFigure out weather.<channel|>"
        + #"<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>"#,
    )
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record(""); return }
    guard case let .functionCall(f) = items[1] else { Issue.record(""); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Tool call with text content before it`() {
    let input = #"Let me check. <|tool_call>call:f{x:1}<tool_call|>"#
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record(""); return }
    guard case let .outputText(part) = m.content[0] else { Issue.record(""); return }
    #expect(part.text == "Let me check. ")
    guard case .functionCall = items[1] else { Issue.record(""); return }
  }

  @Test
  func `Malformed <|tool_call> region (no call: prefix) followed by valid call: indexes stay consecutive`() {
    // The first `<|tool_call>` region body never matches the
    // `call:NAME{` prefix, so no name is extracted. With lazy
    // outputIndex allocation, no slot is consumed for the malformed
    // region. The subsequent valid call should land at output_index 0.
    let input = (
      #"<|tool_call>not-a-call-prefix<tool_call|>"#
        + #"<|tool_call>call:f{x:1}<tool_call|>"#,
    )
    var parser = Gemma4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Only the second region produced a function call.
    let functionCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(functionCalls.count == 1)
    #expect(functionCalls.first?.name == "f")
    let addedIndexes: [Int] = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex } else { return nil }
    }
    #expect(addedIndexes == [0])
  }
}

@Suite("Gemma4Parser — streaming")
struct Gemma4StreamingTests {
  @Test
  func `Char-by-char streaming reconstructs the same items as one-shot`() {
    let input = (
      "<|channel>thought\nthinking step<channel|>"
        + #"<|tool_call>call:f{key:<|"|>value<|"|>,n:42}<tool_call|>"#,
    )

    var streaming = Gemma4Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = Gemma4Parser()
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
        case let (.reasoning(sr), .reasoning(or)):
          #expect(sr.content == or.content)
        default:
          Issue.record("Item kinds differ"); return
      }
    }
  }

  @Test
  func `Reasoning streams incremental deltas`() {
    let chunks = [
      "<|channel>thought\n",
      "first ",
      "second ",
      "third",
      "<channel|>",
    ]
    var parser = Gemma4Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let deltas = events.compactMap { ev -> String? in
      if case let .reasoningTextDelta(e) = ev { return e.delta } else { return nil }
    }
    #expect(deltas.joined() == "first second third")
  }
}

@Suite("Gemma4Parser — continuation")
struct Gemma4ContinuationTests {
  @Test
  func `priorOutput with unclosed <|channel>thought resumes in reasoning`() {
    var parser = ResponseFormat.gemma4.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<|channel>thought\nstarting to think",
    )
    let events = parser.process(ParserInput(text: "more reasoning<channel|>The answer.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "more reasoning")
    guard case let .message(m) = items[1] else { Issue.record("Expected message"); return }
    guard case let .outputText(mPart) = m.content[0] else { Issue.record(""); return }
    #expect(mPart.text == "The answer.")
  }

  @Test
  func `priorOutput with closed <|channel>thought ... <channel|> starts in normal`() {
    var parser = ResponseFormat.gemma4.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<|channel>thought\nthought content<channel|>partial answer",
    )
    let events = parser.process(ParserInput(text: " continues here.")) + parser.finalize()
    let items = accumulateItems(from: events)
    // Resumes in normal phase: no reasoning item.
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `priorOutput with multiple reasoning blocks: only the last's state matters`() {
    // First reasoning block was closed; second was opened mid-stream.
    var parser = ResponseFormat.gemma4.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<|channel>thought\nfirst<channel|>middle<|channel>thought\nsecond",
    )
    let events = parser.process(ParserInput(text: " more<channel|>done")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
    guard case .message = items[1] else { Issue.record("Expected message"); return }
  }

  @Test
  func `nil priorOutput starts in normal phase`() {
    var parser = ResponseFormat.gemma4.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: "just text")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }
}

@Suite("Gemma4Parser — dispatch")
struct Gemma4DispatchTests {
  @Test
  func `Dispatch via ResponseFormat.gemma4.makeParser`() {
    let parser = ResponseFormat.gemma4.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Model type gemma4 routes to .gemma4`() {
    let format = ResponseFormat.infer(modelName: "", modelType: "gemma4", modelConfig: [:])
    #expect(format == .gemma4)
  }

  @Test
  func `Model type gemma4_text also routes to .gemma4`() {
    let format = ResponseFormat.infer(modelName: "", modelType: "gemma4_text", modelConfig: [:])
    #expect(format == .gemma4)
  }

  @Test
  func `Plain gemma model_type routes to .gemmaFunctionCall, not .gemma4`() {
    // `model_type == "gemma"` is the legacy Gemma 1/2 function-call
    // format with `<start_function_call>...{escape}...<end_function_call>`
    // markers, distinct from Gemma 4's multi-token reasoning format.
    let format = ResponseFormat.infer(modelName: "", modelType: "gemma", modelConfig: [:])
    #expect(format == .gemmaFunctionCall)
  }
}

@Suite("Gemma4Parser — adversarial ports")
struct Gemma4AdversarialTests {
  // H9: vLLM test_streaming_split_delimiter_no_invalid_json
  // (test_gemma4_tool_parser.py:572-599). Issue #38946 regression.
  @Test
  func `Closing <|"|> delimiter split across chunks does not leak as <| in JSON`() throws {
    let chunks = [
      "<|tool_call>",
      "call:todowrite{",
      "content:<|\"|>Buy milk<|",
      "\"|>}",
      "<tool_call|>",
    ]
    var parser = Gemma4Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["content"] as? String == "Buy milk")
    // The partial `<|` delimiter must not leak into the final args text.
    #expect(!f.arguments.contains("<|"))
  }

  // M2: vLLM test_streaming_html_argument_does_not_duplicate_tag_prefixes
  // (test_gemma4_tool_parser.py:630-659).
  @Test
  func `HTML payload chunked at < boundaries reconstructs verbatim`() throws {
    let chunks = [
      "<|tool_call>",
      "call:write_file{",
      "path:<|\"|>index.html<|\"|>,",
      "content:<|\"|><!DOCTYPE html>\n<",
      "html lang=\"zh-CN\">\n<",
      "head>\n    <",
      "meta charset=\"UTF-8\">\n    <",
      "meta name=\"viewport\" content=\"width=device-width\">\n",
      "<|\"|>}",
      "<tool_call|>",
    ]
    var parser = Gemma4Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["path"] as? String == "index.html")
    let expectedContent = (
      "<!DOCTYPE html>\n"
        + "<html lang=\"zh-CN\">\n"
        + "<head>\n"
        + "    <meta charset=\"UTF-8\">\n"
        + "    <meta name=\"viewport\" content=\"width=device-width\">\n",
    )
    #expect(decoded["content"] as? String == expectedContent)
  }

  // M6: SGLang test_streaming_self_label_split_across_chunks
  // (test_reasoning_parser.py:775-786).
  @Test
  func `Self-label <|channel> / thought-newline / reasoning across chunks transitions correctly`() {
    let chunks = [
      "<|channel>",
      "thought\n",
      "reasoning here",
      "<channel|>",
    ]
    var parser = Gemma4Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let deltas = events.compactMap { ev -> String? in
      if case let .reasoningTextDelta(e) = ev { return e.delta } else { return nil }
    }
    #expect(deltas.joined() == "reasoning here")
  }
}
