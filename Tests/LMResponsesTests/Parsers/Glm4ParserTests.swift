// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

@Suite("Glm4Parser — basics")
struct Glm4BasicsTests {
  @Test
  func `Plain text emits a single message`() {
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: "hello")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record(""); return }
  }

  @Test
  func `Single tool call without schema falls back to string for unparseable values`() throws {
    let input = (
      "<tool_call>get_weather\n"
        + "<arg_key>city</arg_key>\n"
        + "<arg_value>Beijing</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Beijing")
  }

  @Test
  func `Without schema, JSON-parseable values are coerced (mirrors vLLM _deserialize)`() throws {
    // vLLM's `Glm4MoeModelToolParser._deserialize` (and sglang's
    // `parse_arguments`) tries `json.loads` first when no schema
    // entry is found, so `5` becomes the integer 5, `true` becomes
    // the boolean true, and `{"a":1}` stays as an object. Bare
    // tokens that aren't valid JSON fall back to a JSON string.
    let input = (
      "<tool_call>fn\n"
        + "<arg_key>count</arg_key>\n<arg_value>5</arg_value>\n"
        + "<arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n"
        + "<arg_key>cfg</arg_key>\n<arg_value>{\"a\": 1}</arg_value>\n"
        + "<arg_key>label</arg_key>\n<arg_value>Beijing</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["count"] as? Int == 5)
    #expect(decoded["flag"] as? Bool == true)
    #expect((decoded["cfg"] as? [String: Any])?["a"] as? Int == 1)
    #expect(decoded["label"] as? String == "Beijing")
  }

  @Test
  func `Schema-driven coercion of numeric and boolean parameters`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "get_weather",
        "parameters": [
          "properties": [
            "city": ["type": "string"] as [String: any Sendable],
            "days": ["type": "integer"] as [String: any Sendable],
            "active": ["type": "boolean"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>get_weather\n"
        + "<arg_key>city</arg_key>\n<arg_value>NYC</arg_value>\n"
        + "<arg_key>days</arg_key>\n<arg_value>5</arg_value>\n"
        + "<arg_key>active</arg_key>\n<arg_value>true</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "NYC")
    #expect(decoded["days"] as? Int == 5)
    #expect(decoded["active"] as? Bool == true)
  }

  @Test
  func `Multiple tool calls in sequence`() {
    let input = (
      "<tool_call>f1\n"
        + "<arg_key>x</arg_key>\n<arg_value>1</arg_value>\n"
        + "</tool_call>"
        + "<tool_call>f2\n"
        + "<arg_key>y</arg_key>\n<arg_value>2</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser()
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
  func `Content before tool call is emitted as a normal message`() {
    let input = (
      "Let me check.\n"
        + "<tool_call>f\n"
        + "<arg_key>x</arg_key>\n<arg_value>1</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record(""); return }
    guard case let .outputText(part) = m.content[0] else { Issue.record(""); return }
    #expect(part.text == "Let me check.\n")
    guard case .functionCall = items[1] else { Issue.record(""); return }
  }

  @Test
  func `Char-by-char streaming reconstructs the same items as one-shot`() {
    let input = (
      "<tool_call>f\n"
        + "<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n"
        + "</tool_call>",
    )

    var streaming = Glm4Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = Glm4Parser()
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

  @Test
  func `Parametric stream intervals [1, 2, 4, 8] reconstruct the same args`() throws {
    // Mirrors vLLM's `@pytest.mark.parametrize("stream_interval", [...])`
    // pattern from `tests/tool_parsers/test_hermes_tool_parser.py:190-322`.
    // Catches state-machine bugs that only fire at specific chunk sizes.
    let input = (
      "<tool_call>get_weather\n"
        + "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
        + "<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n"
        + "</tool_call>",
    )
    for interval in [1, 2, 4, 8] {
      let items = streamItems(text: input, interval: interval) { Glm4Parser() }
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
  func `Fixed chunks preserve exact text and string argument deltas`() {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "fn",
        "parameters": [
          "properties": [
            "text": ["type": "string"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let chunks = [
      "prefix ",
      "<tool",
      "_call>fn\n<arg_key>text</arg_key>\n<arg_value>he",
      "llo",
      "</arg_value>\n</tool_call>",
      " suffix",
    ]

    var parser = Glm4Parser(tools: tools)
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
    #expect(glm4OutputTextDeltas(from: events) == ["prefix ", " suffix"])
    #expect(glm4ArgumentDeltas(from: events) == [#"{"text": "he"#, "llo", "\"}"])
  }

  @Test
  func `Closed call followed by text and later call preserves exact deltas`() {
    let chunks = [
      "<tool_call>first\n<arg_key>a</arg_key>\n<arg_value>1</arg_value>\n</tool_call>",
      " gap ",
      "<tool_call>second\n<arg_key>b</arg_key>\n<arg_value>2</arg_value>\n</tool_call>",
    ]

    var parser = Glm4Parser()
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
    #expect(glm4OutputTextDeltas(from: events) == [" gap "])
    #expect(glm4ArgumentDeltas(from: events) == [#"{"a": 1}"#, #"{"b": 2}"#])
  }
}

@Suite("Glm4Parser — GLM 4.7 zero-arg form")
struct Glm47ZeroArgTests {
  // Mirrors vLLM `Glm47MoeModelToolParser`'s regex
  // `<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>`: the function
  // name may appear directly inside `<tool_call>...</tool_call>` with
  // no newline and no `<arg_key>` block.
  @Test
  func `Zero-arg tool call extracts name with empty arguments`() {
    var parser = Glm4Parser()
    let input = "<tool_call>fetch_status</tool_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls.first?.name == "fetch_status")
    #expect(toolCalls.first?.arguments == "{}")
  }

  @Test
  func `Zero-arg tool call with whitespace inside the envelope`() {
    var parser = Glm4Parser()
    let input = "<tool_call>  fetch_status  </tool_call>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls.first?.name == "fetch_status")
    #expect(toolCalls.first?.arguments == "{}")
  }
}

@Suite("Glm4Parser — dispatch")
struct Glm4DispatchTests {
  @Test
  func `Dispatch via ResponseFormat.glm4.makeParser`() {
    let parser = ResponseFormat.glm4.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "hello")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }

  @Test
  func `Name prefix glm-4 routes to .glm4`() {
    let format = ResponseFormat.infer(modelName: "glm-4-9b", modelType: "", modelConfig: [:])
    #expect(format == .glm4)
  }

  @Test
  func `Name prefix GLM-4_5 routes to thinking-enabled GLM parser`() {
    let format = ResponseFormat.infer(modelName: "zai-org/GLM-4.5", modelType: "", modelConfig: [:])
    #expect(format == .glm4Thinking)
  }

  @Test
  func `GLM thinking factory extracts implicit reasoning and tool calls`() {
    let parser = ResponseFormat.glm4Thinking.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let input = """
    need weather<tool_call>get_weather
    <arg_key>city</arg_key>
    <arg_value>Paris</arg_value>
    </tool_call>
    """
    let events = p.process(ParserInput(text: input)) + p.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.first == "need weather")
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }
}

@Suite("Glm4Parser — adversarial ports")
struct Glm4AdversarialTests {
  // H8: vLLM test_streaming_json_escape_in_string
  // (test_glm4_moe_tool_parser.py:624-653).
  @Test
  func `Streaming arg_value with embedded " and newline produces valid JSON`() throws {
    let chunks = [
      "<tool_call>send_message\n",
      "<arg_key>message</arg_key>",
      "<arg_value>Hello \"world\"\nNew line</arg_value>",
      "</tool_call>",
    ]
    var parser = Glm4Parser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let message = decoded["message"] as? String
    #expect(message?.contains("\"world\"") == true)
    #expect(message?.contains("\nNew line") == true)
  }

  // H7: vLLM test_streaming_long_content_incremental
  // (test_glm4_moe_tool_parser.py:656-762).
  //
  // The parser uses the rebuild-and-diff approach from vLLM's
  // `glm4_moe_tool_parser.py` so a long string `<arg_value>` body
  // reaches the consumer in many fragments as it arrives, not in one
  // burst at `</tool_call>`. Asserts ≥10 fragments for the multi-line
  // code fixture (matching vLLM's test threshold).
  @Test
  func `Long content streams as ≥10 incremental fragments and final JSON is valid`() throws {
    let bubbleSortCode = (
      "#!/usr/bin/env python3\n"
        + "# -*- coding: utf-8 -*-\n"
        + "\"\"\"\nBubble Sort Implementation\n\"\"\"\n\n"
        + "def bubble_sort(arr):\n"
        + "    n = len(arr)\n"
        + "    for i in range(n):\n"
        + "        swapped = False\n"
        + "        for j in range(0, n - i - 1):\n"
        + "            if arr[j] > arr[j + 1]:\n"
        + "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
        + "                swapped = True\n"
        + "        if not swapped:\n"
        + "            break\n"
        + "    return arr",
    )
    let tools: [ToolSpec] = [[
      "function": [
        "name": "write_to_file",
        "parameters": [
          "type": "object",
          "properties": [
            "file_path": ["type": "string"] as [String: any Sendable],
            "content": ["type": "string"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    var chunks: [String] = [
      "<tool_call>",
      "write_to_file\n",
      "<arg_key>file_path</arg_key>",
      "<arg_value>/tmp/bubble_sort.py</arg_value>",
      "<arg_key>content</arg_key>",
      "<arg_value>",
    ]
    for line in bubbleSortCode.split(separator: "\n", omittingEmptySubsequences: false) {
      chunks.append(String(line) + "\n")
    }
    chunks.append("</arg_value>")
    chunks.append("</tool_call>")

    var parser = Glm4Parser(tools: tools)
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let argsDeltas = events.compactMap { ev -> String? in
      if case let .functionCallArgumentsDelta(e) = ev { return e.delta } else { return nil }
    }
    // True incremental streaming: ≥10 fragments for the multi-line code
    // value, matching vLLM's threshold for the same fixture.
    #expect(argsDeltas.count >= 10)

    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["file_path"] as? String == "/tmp/bubble_sort.py")
    #expect((decoded["content"] as? String)?.contains("def bubble_sort") == true)
  }

  @Test
  func `Schema with properties but no top-level type infers object`() throws {
    // Sglang's `infer_type_from_json_schema` returns `object` for a
    // schema that has `properties` but no top-level `type`. The Swift
    // port now follows suit, so a model emitting `{"a": 1}` for this
    // param inlines as JSON object instead of being string-quoted.
    let tools: [ToolSpec] = [[
      "function": [
        "name": "f",
        "parameters": [
          "properties": [
            "config": [
              "properties": [
                "a": ["type": "integer"] as [String: any Sendable],
              ] as [String: any Sendable],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>f\n"
        + #"<arg_key>config</arg_key>"#
        + "\n"
        + #"<arg_value>{"a": 1}</arg_value>"#
        + "\n</tool_call>",
    )
    var parser = Glm4Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let cfg = decoded["config"] as? [String: Any]
    #expect(cfg?["a"] as? Int == 1, "object-typed param inlines as JSON object, not stringified")
  }

  @Test
  func `Schema with enum infers type from enum value runtime types`() throws {
    // Enum of integers — should infer `integer`, so `5` parses as Int.
    let tools: [ToolSpec] = [[
      "function": [
        "name": "f",
        "parameters": [
          "properties": [
            "level": ["enum": [1, 2, 3] as [Int]] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>f\n"
        + "<arg_key>level</arg_key>\n<arg_value>2</arg_value>\n</tool_call>",
    )
    var parser = Glm4Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["level"] as? Int == 2)
  }

  @Test
  func `CRLF line endings between tool name and first arg are tolerated`() throws {
    let input = "<tool_call>get_weather\r\n<arg_key>city</arg_key>\r\n<arg_value>NYC</arg_value>\r\n</tool_call>"
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "NYC")
  }

  @Test
  func `Numeric coercion does not trap on integers that exceed Int64 range`() {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "fn",
        "parameters": [
          "properties": [
            "count": ["type": "number"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>fn\n"
        + "<arg_key>count</arg_key>\n<arg_value>100000000000000000000</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    // No trap, and the value emits as a JSON number, not a string.
    #expect(f.arguments.contains("100000000000000000000"))
    #expect(!f.arguments.contains("\"100000000000000000000\""))
  }

  @Test
  func `Truncation mid-string-value emits parseable JSON with status=incomplete`() throws {
    // Stream stops mid-arg-value with no `</arg_value>` and no
    // `</tool_call>`. The defensive close in `closeToolCall` makes
    // the args field parseable so consumers can still read what was
    // streamed; truncation is signalled via `status: .incomplete`.
    let input = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Bei"
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.status == .incomplete)
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Bei")
  }

  // Pins the M10 invariant: a stream truncated after `<tool_call>fn\n`
  // (name parsed, no args) closes with `arguments` being parseable JSON
  // (`{}`), not an empty string. The `scan(isEnd: true)` in `finalize`
  // emits a `{}` delta before the truncation close loop, so cumulative
  // deltas equal the final arguments.
  @Test
  func `Truncation after name-only emits parseable empty-object args`() throws {
    let input = "<tool_call>fn\n"
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.status == .incomplete)
    // Final arguments must be parseable JSON (an empty object).
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded.isEmpty)
    // Cumulative deltas must equal the final arguments.
    let deltas = events.compactMap { ev -> String? in
      if case let .functionCallArgumentsDelta(d) = ev { return d.delta } else { return nil }
    }
    #expect(deltas.joined() == f.arguments)
  }

  // H4: the boolean-coercion path used to aggressively map
  // `yes/no/on/off/1/0` to JSON booleans. Both vLLM (`_deserialize`)
  // and sglang (`parse_arguments`) emit those as strings or numbers
  // instead — only `true`/`false` (and Python's `True`/`False` after
  // case folding) become booleans. Pin the new behavior.
  @Test
  func `Boolean type does not coerce yes/no/on/off — stays as string`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "fn",
        "parameters": [
          "properties": [
            "flag": ["type": "boolean"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>fn\n"
        + "<arg_key>flag</arg_key>\n<arg_value>yes</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    // sglang's `parse_arguments("yes", "boolean")` falls through to
    // strategy 4 (quoted JSON) → `"yes"` (string). vLLM's
    // `_deserialize("yes")` similarly fails json.loads and
    // ast.literal_eval → returns `"yes"`. Either way, not a bool.
    #expect(decoded["flag"] as? String == "yes")
  }

  @Test
  func `Boolean type with 1/0 emits as JSON number, not boolean`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "fn",
        "parameters": [
          "properties": [
            "a": ["type": "boolean"] as [String: any Sendable],
            "b": ["type": "boolean"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>fn\n"
        + "<arg_key>a</arg_key>\n<arg_value>1</arg_value>\n"
        + "<arg_key>b</arg_key>\n<arg_value>0</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    // vLLM/sglang both run `json.loads("1") → 1` (int). Stays as int.
    #expect(decoded["a"] as? Int == 1)
    #expect(decoded["b"] as? Int == 0)
  }

  @Test
  func `Boolean type with capitalized Python literals (True/False) does coerce`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "fn",
        "parameters": [
          "properties": [
            "a": ["type": "boolean"] as [String: any Sendable],
            "b": ["type": "boolean"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>fn\n"
        + "<arg_key>a</arg_key>\n<arg_value>True</arg_value>\n"
        + "<arg_key>b</arg_key>\n<arg_value>False</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    // vLLM's `_deserialize("True")` falls through to `ast.literal_eval`
    // which returns Python `True` → `json.dumps(True) == "true"`.
    #expect(decoded["a"] as? Bool == true)
    #expect(decoded["b"] as? Bool == false)
  }
}

private func glm4OutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .outputTextDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

private func glm4ArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .functionCallArgumentsDelta(e) = event {
      return e.delta
    }
    return nil
  }
}
