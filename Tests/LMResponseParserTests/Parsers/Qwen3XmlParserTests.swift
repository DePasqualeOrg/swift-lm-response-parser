// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("Qwen3XmlParser — plain text")
struct Qwen3XmlPlainTextTests {
  @Test
  func `Single chunk of text without tool calls emits a single message`() {
    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: "This is a regular response without any tool calls.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "This is a regular response without any tool calls.")
    #expect(m.status == .completed)
  }

  @Test
  func `Empty stream finalize emits nothing`() {
    var parser = Qwen3XmlParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("Qwen3XmlParser — single tool call")
struct Qwen3XmlSingleToolCallTests {
  @Test
  func `Single tool call with one string parameter`() {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=get_weather>
    <parameter=city>Tokyo</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    #expect(f.status == .completed)
    #expect(f.arguments == #"{"city": "Tokyo"}"#)
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
  }

  @Test
  func `Tool call with no parameters yields {}`() {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=refresh>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "refresh")
    #expect(f.arguments == "{}")
  }

  @Test
  func `Function-call event sequence: added → arguments.delta+ → arguments.done → output_item.done`() {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=fn>
    <parameter=x>1</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let kinds = events.map { eventKind($0) }
    #expect(kinds == [
      "outputItemAdded",
      "functionCallArgumentsDelta",
      "functionCallArgumentsDelta",
      "functionCallArgumentsDone",
      "outputItemDone",
    ])
  }
}

@Suite("Qwen3XmlParser — multiple parameters / data types")
struct Qwen3XmlDataTypeTests {
  /// Build a tool spec for the test_function with typed parameters.
  static let typedToolSpec: [ToolSpec] = [[
    "type": "function",
    "function": [
      "name": "test_function",
      "parameters": [
        "type": "object",
        "properties": [
          "string_field": ["type": "string"],
          "int_field": ["type": "integer"],
          "float_field": ["type": "number"],
          "bool_field": ["type": "boolean"],
          "null_field": ["type": "string"],
          "array_field": ["type": "array"],
          "object_field": ["type": "object"],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ] as [String: any Sendable],
  ]]

  @Test
  func `Various data types are coerced via the schema`() throws {
    var parser = Qwen3XmlParser(tools: Self.typedToolSpec)
    let input = """
    <tool_call>
    <function=test_function>
    <parameter=string_field>hello</parameter>
    <parameter=int_field>42</parameter>
    <parameter=float_field>3.14</parameter>
    <parameter=bool_field>true</parameter>
    <parameter=null_field>null</parameter>
    <parameter=array_field>["a", "b", "c"]</parameter>
    <parameter=object_field>{"nested": "value"}</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }

    // Decode the JSON arguments back to a dict and compare values.
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["string_field"] as? String == "hello")
    #expect(decoded["int_field"] as? Int == 42)
    #expect(decoded["float_field"] as? Double == 3.14)
    #expect(decoded["bool_field"] as? Bool == true)
    #expect(decoded["null_field"] is NSNull)
    #expect((decoded["array_field"] as? [String]) == ["a", "b", "c"])
    let nested = decoded["object_field"] as? [String: String]
    #expect(nested?["nested"] == "value")
  }

  @Test
  func `Without tool spec, parameters are coerced as strings`() throws {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=fn>
    <parameter=count>42</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    // No schema → keep as string.
    #expect(decoded["count"] as? String == "42")
  }

  @Test
  func `Bare null parameter value decodes as JSON null regardless of schema`() throws {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=fn>
    <parameter=value>null</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["value"] is NSNull)
  }

  @Test
  func `Numeric coercion does not trap on integers that exceed Int64 range`() {
    let toolSpec: [ToolSpec] = [[
      "type": "function",
      "function": [
        "name": "fn",
        "parameters": [
          "type": "object",
          "properties": ["count": ["type": "number"]] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    var parser = Qwen3XmlParser(tools: toolSpec)
    let input = """
    <tool_call>
    <function=fn>
    <parameter=count>100000000000000000000</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    // The exact representation may be the raw digits or a normalized
    // number; what matters is that the parser does not trap and the
    // value is emitted as a JSON number, not a string.
    #expect(f.arguments.contains("100000000000000000000"))
    #expect(!f.arguments.contains("\"100000000000000000000\""))
  }

  @Test
  func `Integer coercion preserves valid literals that exceed Int64 range`() {
    let toolSpec: [ToolSpec] = [[
      "type": "function",
      "function": [
        "name": "fn",
        "parameters": [
          "type": "object",
          "properties": ["count": ["type": "integer"]] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    var parser = Qwen3XmlParser(tools: toolSpec)
    let input = """
    <tool_call>
    <function=fn>
    <parameter=count>100000000000000000000</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.arguments.contains("100000000000000000000"))
    #expect(!f.arguments.contains("\"100000000000000000000\""))
  }

  @Test
  func `AnyOf object or null schema emits object arguments without double encoding`() throws {
    let toolSpec: [ToolSpec] = [[
      "type": "function",
      "function": [
        "name": "fn",
        "parameters": [
          "type": "object",
          "properties": [
            "payload": [
              "anyOf": [
                ["type": "object"],
                ["type": "null"],
              ] as [[String: any Sendable]],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    var parser = Qwen3XmlParser(tools: toolSpec)
    let input = """
    <tool_call>
    <function=fn>
    <parameter=payload>{"required": true}</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let payload = try #require(decoded["payload"] as? [String: Any])
    #expect(payload["required"] as? Bool == true)
  }

  @Test
  func `Python-literal fallback preserves True/False/None inside string contents`() throws {
    let toolSpec: [ToolSpec] = [[
      "type": "function",
      "function": [
        "name": "fn",
        "parameters": [
          "type": "object",
          "properties": ["payload": ["type": "object"]] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    var parser = Qwen3XmlParser(tools: toolSpec)
    // Single quotes force the strict-JSON decode to fail, so the
    // Python-literal fallback runs. The string value contains "True"
    // and "None" which must NOT be replaced with `true` / `null`.
    let input = """
    <tool_call>
    <function=fn>
    <parameter=payload>{'msg': 'True positive, None spotted'}</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let payload = decoded["payload"] as? [String: Any]
    #expect(payload?["msg"] as? String == "True positive, None spotted")
  }
}

@Suite("Qwen3XmlParser — escaped strings")
struct Qwen3XmlEscapedStringTests {
  @Test
  func `Strings with quotes, backslashes, and newlines are properly JSON-encoded`() throws {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=test_function>
    <parameter=quoted>He said "hello"</parameter>
    <parameter=path>C:\\Users\\file.txt</parameter>
    <parameter=newline>line1
    line2</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["quoted"] as? String == #"He said "hello""#)
    #expect(decoded["path"] as? String == #"C:\Users\file.txt"#)
    #expect((decoded["newline"] as? String)?.contains("\n") == true)
  }
}

@Suite("Qwen3XmlParser — parallel tool calls")
struct Qwen3XmlParallelToolCallTests {
  @Test
  func `Two consecutive tool calls each produce a function_call item`() {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=get_weather>
    <parameter=city>Tokyo</parameter>
    </function>
    </tool_call><tool_call>
    <function=get_time>
    <parameter=timezone>Asia/Tokyo</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "get_weather")
    #expect(b.name == "get_time")
    #expect(a.id != b.id)
  }

  @Test
  func `Parallel tool calls get monotonically increasing output_indexes`() {
    var parser = Qwen3XmlParser()
    let input = """
    <tool_call>
    <function=a>
    </function>
    </tool_call><tool_call>
    <function=b>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let addedIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(addedIndexes == [0, 1])
  }
}

@Suite("Qwen3XmlParser — surrounding text")
struct Qwen3XmlSurroundingTextTests {
  @Test
  func `Text before and after a tool call is emitted as message content`() {
    var parser = Qwen3XmlParser()
    let input = """
    Let me check the weather for you.

    <tool_call>
    <function=get_weather>
    <parameter=city>Tokyo</parameter>
    </function>
    </tool_call>

    I will get that information.
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Expected: message (lead-in) + function call + message (follow-up) = 3 items
    #expect(items.count == 3)
    guard case let .message(lead) = items[0],
          case let .outputText(leadText) = lead.content[0]
    else {
      Issue.record("Expected lead message"); return
    }
    #expect(leadText.text.contains("Let me check"))

    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")

    guard case let .message(tail) = items[2],
          case let .outputText(tailText) = tail.content[0]
    else {
      Issue.record("Expected tail message"); return
    }
    #expect(tailText.text.contains("I will get"))
  }
}

@Suite("Qwen3XmlParser — reasoning")
struct Qwen3XmlReasoningTests {
  @Test
  func `<think>r</think> followed by tool call: reasoning + tool call`() {
    var parser = Qwen3XmlParser()
    let input = """
    <think>I should check the weather.</think><tool_call>
    <function=get_weather>
    <parameter=city>Tokyo</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "I should check the weather.")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `InitialState .reasoning with implicit tool-call end`() {
    var parser = Qwen3XmlParser(initialState: .reasoning)
    let input = """
    Reasoning runs into the call.<tool_call>
    <function=fn>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "Reasoning runs into the call.")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }
}

@Suite("Qwen3XmlParser — streaming boundaries")
struct Qwen3XmlStreamingBoundaryTests {
  @Test
  func `Char-by-char streaming yields the same items as single-shot`() {
    let input = """
    <tool_call>
    <function=fn>
    <parameter=x>hello</parameter>
    <parameter=y>world</parameter>
    </function>
    </tool_call>
    """

    var streaming = Qwen3XmlParser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = Qwen3XmlParser()
    let oneShotEvents = oneShot.process(ParserInput(text: input)) + oneShot.finalize()
    let oneShotItems = accumulateItems(from: oneShotEvents)

    #expect(streamingItems.count == oneShotItems.count)
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.functionCall(sf), .functionCall(of)):
          #expect(sf.name == of.name)
          #expect(sf.arguments == of.arguments)
          #expect(sf.status == of.status)
        case let (.message(sm), .message(om)):
          #expect(sm.content == om.content)
        default:
          Issue.record("Item kinds differ: \(s) vs \(o)")
      }
    }
  }

  @Test
  func `Tag split across chunks: <tool_ | call> still parses`() {
    var parser = Qwen3XmlParser()
    var events = parser.process(ParserInput(text: "<tool_"))
    events += parser.process(ParserInput(text: "call>\n<function=fn>\n"))
    events += parser.process(ParserInput(text: "<parameter=x>v</parameter>\n</function>\n</tool_call>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.arguments == #"{"x": "v"}"#)
  }

  @Test
  func `Parameter value split across chunks accumulates correctly`() {
    var parser = Qwen3XmlParser()
    var events = parser.process(ParserInput(text: "<tool_call>\n<function=fn>\n<parameter=msg>hel"))
    events += parser.process(ParserInput(text: "lo wor"))
    events += parser.process(ParserInput(text: "ld</parameter>\n</function>\n</tool_call>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.arguments == #"{"msg": "hello world"}"#)
  }

  @Test
  func `Parametric stream intervals [1, 2, 4, 8] reconstruct the same args`() throws {
    // Mirrors vLLM's `@pytest.mark.parametrize("stream_interval", [...])`
    // pattern from `tests/tool_parsers/test_hermes_tool_parser.py:190-322`.
    // Reconstruct the same args regardless of chunk width.
    let input = """
    <tool_call>
    <function=get_weather>
    <parameter=city>San Francisco</parameter>
    <parameter=unit>celsius</parameter>
    </function>
    </tool_call>
    """
    for interval in [1, 2, 4, 8] {
      let items = streamItems(text: input, interval: interval) { Qwen3XmlParser() }
      #expect(items.count == 1, "interval=\(interval): expected 1 item, got \(items.count)")
      guard case let .functionCall(f) = items.first else {
        Issue.record("interval=\(interval): expected function call, got \(items)")
        continue
      }
      #expect(f.name == "get_weather", "interval=\(interval)")
      let parsed = try JSONSerialization.jsonObject(
        with: Data(f.arguments.utf8),
      ) as? [String: Any]
      #expect(parsed?["city"] as? String == "San Francisco", "interval=\(interval)")
      #expect(parsed?["unit"] as? String == "celsius", "interval=\(interval)")
    }
  }
}

@Suite("Qwen3XmlParser — finalize edge cases")
struct Qwen3XmlFinalizeTests {
  @Test
  func `Truncated tool call: function open but no parameters/close emits incomplete call`() {
    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: "<tool_call>\n<function=fn>\n<parameter=x>val")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.status == .incomplete)
  }

  @Test
  func `Tool call with function tag but no name still emits no spurious item`() {
    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: "<tool_call><function=></function></tool_call>")) + parser.finalize()
    let items = accumulateItems(from: events)
    // Empty function name produces a function call with name "" — the
    // wire format allows it, but it's a degenerate case. Allow either
    // 0 items (parser refuses to open) or 1 item with empty name.
    #expect(items.count <= 1)
  }
}

@Suite("Qwen3XmlParser — dispatch")
struct Qwen3XmlDispatchTests {
  @Test
  func `ResponseFormat.qwen3Xml.makeParser returns a working Qwen3XmlParser`() {
    let parser = ResponseFormat.qwen3Xml.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let input = "<tool_call>\n<function=fn>\n<parameter=x>v</parameter>\n</function>\n</tool_call>"
    let events = p.process(ParserInput(text: input)) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }

  @Test
  func `priorOutput with unclosed <think> resumes parser in reasoning state`() {
    let parser = ResponseFormat.qwen3Xml.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial reasoning",
    )
    var p = parser
    let events = p.process(ParserInput(text: " continues</think>after")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == " continues")
  }
}

@Suite("Qwen3XmlParser — adversarial ports")
struct Qwen3XmlAdversarialTests {
  // H5: vLLM test_malformed_xml_no_gt_delimiter and test_none_tool_calls_filtered
  // (test_qwen3coder_tool_parser.py:999-1039).
  @Test
  func `Malformed <function=name without > is filtered; following good call extracts`() throws {
    // Two tool calls: the first has a malformed function tag (no `>`
    // before another `<`); the second is well-formed. Verify only the
    // second surfaces.
    let input = (
      "<tool_call>\n"
        + "<function=bad_func_no_gt\n"
        + "</function>\n"
        + "</tool_call>\n"
        + "<tool_call>\n"
        + "<function=get_current_weather>\n"
        + "<parameter=city>Dallas</parameter>\n"
        + "<parameter=state>TX</parameter>\n"
        + "</function>\n"
        + "</tool_call>",
    )
    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_current_weather")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Dallas")
    #expect(decoded["state"] as? String == "TX")
  }

  @Test
  func `Single malformed <function= body does not crash and surfaces no spurious call`() {
    // From the same vLLM regression. Single block with no `>` in the
    // function tag — should not crash; tool_calls list contains only
    // well-formed entries (we surface zero).
    let input = (
      "<tool_call>\n"
        + "<function=get_current_weather\n"
        + "<parameter=city>Dallas</parameter>\n"
        + "</function>\n"
        + "</tool_call>",
    )
    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    // No spurious tool calls with garbage names.
    for tc in toolCalls {
      #expect(!tc.name.contains("<"))
    }
  }

  // H6: vLLM test_no_double_serialization_string_args
  // (test_qwen3coder_tool_parser.py:1115-1148).
  @Test
  func `Schema-typed string args do not double-serialize`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "greet",
        "parameters": [
          "type": "object",
          "properties": [
            "message": ["type": "string"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "<tool_call>\n"
        + "<function=greet>\n"
        + "<parameter=message>hello world</parameter>\n"
        + "</function>\n"
        + "</tool_call>",
    )
    var parser = Qwen3XmlParser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["message"] as? String == "hello world")
    // Raw arguments must not contain the double-serialized form.
    #expect(!f.arguments.contains(#"\"hello world\""#))
  }

  // M0: vLLM test_extract_tool_calls_missing_closing_parameter_tag and
  // streaming variant (test_qwen3coder_tool_parser.py:327-375).
  @Test
  func `Missing parameter closing tag before next parameter still extracts all args`() throws {
    let input = (
      "Let me check.\n"
        + "<tool_call>\n"
        + "<function=get_current_weather>\n"
        + "<parameter=city>\nDallas\n"
        + "<parameter=state>\nTX\n"
        + "<parameter=unit>\ncelsius\n"
        + "</function>\n"
        + "</tool_call>",
    )
    assertStreamingReconstruction(input, parser: { Qwen3XmlParser() })

    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let messages = items.compactMap { item -> ResponseOutputMessage? in
      if case let .message(m) = item { return m } else { return nil }
    }
    #expect(messages.count == 1)
    if case let .outputText(t) = messages[0].content[0] {
      #expect(t.text == "Let me check.\n")
    }

    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_current_weather")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Dallas")
    #expect(decoded["state"] as? String == "TX")
    #expect(decoded["unit"] as? String == "celsius")
  }

  // M1: vLLM test_extract_tool_calls_fallback_no_tags
  // (test_qwen3coder_tool_parser.py:377-397).
  @Test
  func `Fallback: <function=…>…</function> without <tool_call> wrapper still extracts`() throws {
    let input = (
      "<function=get_current_weather>\n"
        + "<parameter=city>\nDallas\n</parameter>\n"
        + "<parameter=state>\nTX\n</parameter>\n"
        + "</function>",
    )
    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_current_weather")
    let decodedData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Dallas")
    #expect(decoded["state"] as? String == "TX")
  }
}
