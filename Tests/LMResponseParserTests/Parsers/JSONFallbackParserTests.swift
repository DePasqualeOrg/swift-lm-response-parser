// Copyright © Anthony DePasquale

@testable import LMResponseParser
import Testing

@Suite("JSONFallbackParser — plain text")
struct JSONFallbackPlainTextTests {
  @Test
  func `Single chunk of text emits open + delta then finalize closes cleanly`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
    #expect(m.content.count == 1)
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(t.text == "hello world")
  }

  @Test
  func `Multiple chunks accumulate correctly via deltas`() {
    var parser = JSONFallbackParser()
    var events = parser.process(ParserInput(text: "hel"))
    events += parser.process(ParserInput(text: "lo"))
    events += parser.process(ParserInput(text: " world"))
    events += parser.finalize()

    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message with outputText"); return
    }
    #expect(t.text == "hello world")
  }

  @Test
  func `Leading whitespace then text opens a message correctly`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(text: "   hello")) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "   hello")
  }

  @Test
  func `Text-mode event sequence: added → content_part.added → delta → text.done → content_part.done → item.done`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(text: "hi")) + parser.finalize()

    let kinds = events.map { eventKind($0) }
    #expect(kinds == [
      "outputItemAdded",
      "contentPartAdded",
      "outputTextDelta",
      "outputTextDone",
      "contentPartDone",
      "outputItemDone",
    ])
  }
}

@Suite("JSONFallbackParser — single tool call")
struct JSONFallbackSingleCallTests {
  @Test
  func `Object with name + arguments object is recognized as a function call`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "get_weather")
    #expect(f.status == .completed)
    // JSONSerialization with sortedKeys: only one key here, but format is stable.
    #expect(f.arguments.contains("\"city\""))
    #expect(f.arguments.contains("\"Paris\""))
  }

  @Test
  func `Object with parameters key (some models) is also recognized`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"{"name": "fn", "parameters": {"x": 1}}"#,
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "fn")
    #expect(f.arguments.contains("\"x\""))
  }

  @Test
  func `Arguments encoded as a JSON string are echoed verbatim`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"{"name": "fn", "arguments": "{\"x\": 1}"}"#,
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.arguments == "{\"x\": 1}")
  }

  @Test
  func `Non-object argument values serialize as JSON fragments`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"""
      [
      {"name": "array_arg", "arguments": [1, 2]},
      {"name": "number_arg", "arguments": 42},
      {"name": "bool_arg", "arguments": true},
      {"name": "null_arg", "arguments": null}
      ]
      """#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    let calls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(call) = item { return call }
      return nil
    }

    #expect(calls.map { $0.name } == ["array_arg", "number_arg", "bool_arg", "null_arg"])
    #expect(calls.map { $0.arguments } == ["[1,2]", "42", "true", "null"])
  }

  @Test
  func `Function-call event sequence: added → arguments.delta → arguments.done → item.done (no content_part envelope)`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"{"name": "fn", "arguments": {}}"#,
    )) + parser.finalize()
    let kinds = events.map { eventKind($0) }
    #expect(kinds == [
      "outputItemAdded",
      "functionCallArgumentsDelta",
      "functionCallArgumentsDone",
      "outputItemDone",
    ])
  }

  @Test
  func `Function call has distinct fc_ item ID and call_ call_id`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"{"name": "fn", "arguments": {}}"#,
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
    #expect(f.id != f.callId, "item id and call_id must be distinct so downstream tool-output items can reference call_id stably")
  }
}

@Suite("JSONFallbackParser — multiple tool calls")
struct JSONFallbackMultipleCallsTests {
  @Test
  func `Top-level array yields one function_call item per element`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"[{"name": "f1", "arguments": {}}, {"name": "f2", "arguments": {"x": 1}}]"#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0] else { Issue.record("Expected first function_call"); return }
    guard case let .functionCall(b) = items[1] else { Issue.record("Expected second function_call"); return }
    #expect(a.name == "f1")
    #expect(b.name == "f2")
    #expect(a.id != b.id)
  }

  @Test
  func `Each call gets its own monotonically increasing output_index`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"[{"name": "f1", "arguments": {}}, {"name": "f2", "arguments": {}}]"#,
    )) + parser.finalize()

    let addedIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes == [0, 1])
  }
}

@Suite("JSONFallbackParser — finalize edge cases")
struct JSONFallbackFinalizeTests {
  @Test
  func `finalize on an empty stream emits nothing`() {
    var parser = JSONFallbackParser()
    #expect(parser.finalize().isEmpty)
  }

  @Test
  func `finalize after only whitespace emits nothing`() {
    var parser = JSONFallbackParser()
    _ = parser.process(ParserInput(text: "   \n"))
    #expect(parser.finalize().isEmpty)
  }

  @Test
  func `Truncated JSON object falls through to message with status=incomplete`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(
      text: #"{"name": "fn", "arguments": {"loca"#,
    )) + parser.finalize()

    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0] else { Issue.record("Expected message fallback"); return }
    #expect(m.status == .incomplete)
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(t.text.hasPrefix("{\"name\""), "Original buffer should be replayed verbatim")
  }

  @Test
  func `Valid JSON that isn't a tool call falls through to message with status=completed`() {
    var parser = JSONFallbackParser()
    let events = parser.process(ParserInput(text: #"{"foo": "bar"}"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0] else { Issue.record("Expected message fallback"); return }
    #expect(m.status == .completed)
  }

  @Test
  func `Streaming chunks that build up to a valid tool call are batched correctly`() {
    var parser = JSONFallbackParser()
    var events = parser.process(ParserInput(text: "{"))
    #expect(events.isEmpty, "Buffering during JSON mode should emit nothing")
    events += parser.process(ParserInput(text: #""name": "fn", "arguments": "#))
    #expect(events.isEmpty)
    events += parser.process(ParserInput(text: "{}}"))
    #expect(events.isEmpty)
    events += parser.finalize()

    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "fn")
    #expect(f.arguments == "{}")
  }
}

@Suite("JSONFallbackParser — via dispatch")
struct JSONFallbackDispatchIntegrationTests {
  @Test
  func `ResponseFormat.json.makeParser returns a working JSONFallbackParser`() {
    let parser = ResponseFormat.json.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hi")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `ResponseFormat.infer falls back to JSON when neither name nor type matches`() {
    let format = ResponseFormat.infer(
      modelName: "totally/unknown-model-9000",
      modelType: "unknown_type",
      modelConfig: [:],
    ) ?? .json
    #expect(format == .json)
    var parser = format.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: "hi")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `parseResponse one-shot wrapper works for plain text`() {
    let items = parseResponse(
      "hello",
      format: .json,
      tokenizer: StubTokenizer(),
    )
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "hello")
  }

  @Test
  func `parseResponse one-shot wrapper works for tool call`() {
    let items = parseResponse(
      #"{"name": "fn", "arguments": {"x": 1}}"#,
      format: .json,
      tokenizer: StubTokenizer(),
    )
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function_call"); return }
    #expect(f.name == "fn")
  }
}
