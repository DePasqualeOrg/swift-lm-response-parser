// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Adversarial fixtures lifted from vLLM's `tests/tool_parsers/common_tests.py`
// `test_escaped_strings`. Each parser receives a tool-call argument value
// containing escaped quotes, backslashes, and newlines; the JSON-encoded
// arguments string must round-trip through `JSONSerialization` to the
// expected literal values.

private let expectedTextValue = #"He said "hello""#
private let expectedPathValue = #"C:\Users\file"#
private let expectedNewlineValue = "line1\nline2"

private func decodeArgs(_ args: String) throws -> [String: Any] {
  let data = args.data(using: .utf8)!
  return try JSONSerialization.jsonObject(with: data) as! [String: Any]
}

@Suite("Escaped strings — Hermes")
struct HermesEscapedStringsTests {
  @Test
  func `Argument values with escaped quotes, backslashes, and newlines round-trip`() throws {
    let input = #"<tool_call>{"name": "send_message", "arguments": {"text": "He said \"hello\"", "path": "C:\\Users\\file", "newline": "line1\nline2"}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "send_message")
    let decoded = try decodeArgs(f.arguments)
    #expect(decoded["text"] as? String == expectedTextValue)
    #expect(decoded["path"] as? String == expectedPathValue)
    #expect(decoded["newline"] as? String == expectedNewlineValue)
  }
}

@Suite("Escaped strings — DeepSeek-R1")
struct DeepSeekR1EscapedStringsTests {
  @Test
  func `CJK envelope wraps escaped JSON values intact`() throws {
    let input = (
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>send_message\n"
        + "```json\n"
        + #"{"text": "He said \"hello\"", "path": "C:\\Users\\file", "newline": "line1\nline2"}"#
        + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "send_message")
    let decoded = try decodeArgs(f.arguments)
    #expect(decoded["text"] as? String == expectedTextValue)
    #expect(decoded["path"] as? String == expectedPathValue)
    #expect(decoded["newline"] as? String == expectedNewlineValue)
  }
}

@Suite("Escaped strings — DeepSeek-V3")
struct DeepSeekV3EscapedStringsTests {
  @Test
  func `CJK envelope wraps escaped JSON values intact (V3 fence)`() throws {
    // Direct port of vllm/tests/tool_parsers/test_deepseekv3_tool_parser.py
    // `escaped_strings_output` (V3 base format).
    let input = (
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>send_message\n"
        + "```json\n"
        + #"{"text": "He said \"hello\"", "path": "C:\\Users\\file", "newline": "line1\nline2"}"#
        + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "send_message")
    let decoded = try decodeArgs(f.arguments)
    #expect(decoded["text"] as? String == expectedTextValue)
    #expect(decoded["path"] as? String == expectedPathValue)
    #expect(decoded["newline"] as? String == expectedNewlineValue)
  }
}

@Suite("Escaped strings — DeepSeek-V3.1")
struct DeepSeekV31EscapedStringsTests {
  @Test
  func `V3.1 no-fence envelope wraps escaped JSON values intact`() throws {
    // V3.1 drops the `function\n` literal and the ````json` fence.
    // Function name appears directly after `<｜tool▁call▁begin｜>` and JSON
    // appears directly after `<｜tool▁sep｜>`.
    let input = (
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>send_message<｜tool▁sep｜>"
        + #"{"text": "He said \"hello\"", "path": "C:\\Users\\file", "newline": "line1\nline2"}"#
        + "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "send_message")
    let decoded = try decodeArgs(f.arguments)
    #expect(decoded["text"] as? String == expectedTextValue)
    #expect(decoded["path"] as? String == expectedPathValue)
    #expect(decoded["newline"] as? String == expectedNewlineValue)
  }
}

@Suite("Escaped strings — DeepSeek-V3.2")
struct DeepSeekV32EscapedStringsTests {
  @Test
  func `DSML JSON-body invoke carries escaped values intact`() throws {
    // V3.2 supports a JSON body directly inside `<｜DSML｜invoke>`.
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="send_message">"#
        + #"{"text": "He said \"hello\"", "path": "C:\\Users\\file", "newline": "line1\nline2"}"#
        + "</｜DSML｜invoke>"
        + "</｜DSML｜function_calls>",
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "send_message")
    let decoded = try decodeArgs(f.arguments)
    #expect(decoded["text"] as? String == expectedTextValue)
    #expect(decoded["path"] as? String == expectedPathValue)
    #expect(decoded["newline"] as? String == expectedNewlineValue)
  }
}

@Suite("Escaped strings — MiniMax M2")
struct MiniMaxM2EscapedStringsTests {
  @Test
  func `Parameter-tag values containing quotes, backslashes, and newlines round-trip`() throws {
    // MiniMax M2's wire format puts each parameter value inside a
    // `<parameter name="K">VALUE</parameter>` element verbatim. The
    // parser is responsible for JSON-encoding the value into the args
    // object so quotes/backslashes/newlines reach the consumer correctly.
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="send_message">"#
        + #"<parameter name="text">He said "hello"</parameter>"#
        + #"<parameter name="path">C:\Users\file</parameter>"#
        + "<parameter name=\"newline\">line1\nline2</parameter>"
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "send_message")
    let decoded = try decodeArgs(f.arguments)
    #expect(decoded["text"] as? String == expectedTextValue)
    #expect(decoded["path"] as? String == expectedPathValue)
    #expect(decoded["newline"] as? String == expectedNewlineValue)
  }
}

@Suite("Escaped strings — GLM 4")
struct Glm4EscapedStringsTests {
  @Test
  func `arg_value tag contents with quotes, backslashes, and newlines round-trip`() throws {
    // GLM 4's wire format puts each value inside `<arg_value>VALUE</arg_value>`
    // verbatim. The parser renders each value into the JSON args object.
    let input = (
      "<tool_call>send_message\n"
        + "<arg_key>text</arg_key>\n"
        + "<arg_value>He said \"hello\"</arg_value>\n"
        + "<arg_key>path</arg_key>\n"
        + #"<arg_value>C:\Users\file</arg_value>"#
        + "\n<arg_key>newline</arg_key>\n"
        + "<arg_value>line1\nline2</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "send_message")
    let decoded = try decodeArgs(f.arguments)
    #expect(decoded["text"] as? String == expectedTextValue)
    #expect(decoded["path"] as? String == expectedPathValue)
    #expect(decoded["newline"] as? String == expectedNewlineValue)
  }
}
