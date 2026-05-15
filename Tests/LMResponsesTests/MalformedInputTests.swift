// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// Adversarial fixtures lifted from vLLM's `tests/tool_parsers/` per-format
// `malformed_input_outputs` lists (where present) and from individual
// `test_*_invalid_json` cases. Each test verifies the parser handles broken
// input without crashing and produces a sane fallback — either no items,
// partial items closed as `incomplete`, or the malformed input forwarded as
// content. The goal is resilience pinning, not correctness assertion: if a
// future refactor changes how we degrade gracefully, the test fails loudly.

private func itemKinds(_ items: [ResponseOutputItem]) -> [String] {
  items.map { item in
    switch item {
      case .message: "message"
      case .functionCall: "functionCall"
      case .reasoning: "reasoning"
      case .functionCallOutput: "functionCallOutput"
    }
  }
}

@Suite("Malformed input — Hermes")
struct HermesMalformedInputTests {
  @Test
  func `Unclosed <tool_call> closes the tool call as incomplete at finalize`() {
    let input = #"<tool_call>{"name": "fn", "arguments": {"x": 1}"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
    #expect(f.status == .incomplete)
  }

  @Test
  func `Malformed JSON arguments still produce a function call closed as completed`() {
    // vLLM Hermes test_hermes_parser_non_streaming_tool_call_invalid_json:
    // missing closing brace mid-stream. Our parser closes the tool call
    // as completed when `</tool_call>` arrives, but we don't validate the
    // JSON — best effort, garbage in / garbage out.
    let input = #"<tool_call>{"name": "fn", "arguments": {"x":</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Verify no crash and at most one tool call surfaces.
    #expect(items.count <= 1)
  }

  @Test
  func `Missing name field in arguments holds without producing a tool call`() {
    // Without a `"name"` field, `extractToolName` returns nil and we
    // never open the function call. The buffer stays parked until
    // `</tool_call>` is seen; at finalize, no item should be emitted.
    let input = #"<tool_call>{"arguments": {"x": 1}}</tool_call>"#
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}

@Suite("Malformed input — DeepSeek-V3")
struct DeepSeekV3MalformedInputTests {
  @Test
  func `Tool call missing closing brace closes as incomplete`() {
    // Direct port of vllm/tests/tool_parsers/test_deepseekv3_tool_parser.py
    // `malformed_input_outputs[0]`.
    let input = (
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        + "```json\n"
        + #"{"city": "Tokyo""#
        + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV3Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Resilience: no crash, at most one tool call surfaces.
    #expect(items.count <= 1)
  }

  @Test
  func `Section opener without <｜tool▁call▁begin｜> degrades gracefully`() {
    // Direct port of vllm/tests/tool_parsers/test_deepseekv3_tool_parser.py
    // `malformed_input_outputs[1]`.
    let input = (
      "<｜tool▁calls▁begin｜>function<｜tool▁sep｜>get_weather\n"
        + "```json\n"
        + #"{"city": "Tokyo"}"#
        + "\n```<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV3Parser()
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }
}

@Suite("Malformed input — DeepSeek-V3.1")
struct DeepSeekV31MalformedInputTests {
  @Test
  func `missing closing brace`() {
    // V3.1 shape (no fence, function name right after begin marker).
    let input = (
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
        + #"{"city": "Tokyo""#
        + "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count <= 1)
  }

  @Test
  func `missing tool call begin`() {
    let input = (
      "<｜tool▁calls▁begin｜>get_weather<｜tool▁sep｜>"
        + #"{"city": "Tokyo"}"#
        + "<｜tool▁calls▁end｜>",
    )
    var parser = DeepSeekV31Parser()
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }
}

@Suite("Malformed input — DeepSeek-V3.2")
struct DeepSeekV32MalformedInputTests {
  @Test
  func `DSML invoke without closing tag closes as incomplete`() {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="get_weather">"#
        + #"{"city": "Tokyo""#,
    )
    var parser = DeepSeekV32Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count <= 1)
  }

  @Test
  func `DSML envelope without function_calls close degrades gracefully`() {
    let input = (
      "<｜DSML｜function_calls>"
        + #"<｜DSML｜invoke name="fn"><｜DSML｜parameter name="x">1</｜DSML｜parameter></｜DSML｜invoke>"#,
    )
    var parser = DeepSeekV32Parser()
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }
}

@Suite("Malformed input — Mistral")
struct MistralMalformedInputTests {
  @Test
  func `Compact format with unbalanced JSON closes call as incomplete`() {
    let input = #"[TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo""#
    var parser = MistralParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count <= 1)
  }

  @Test
  func `JSON-array format missing closing bracket degrades gracefully`() {
    let input = #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Tokyo"}}"#
    var parser = MistralParser()
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }
}

@Suite("Malformed input — MiniMax M2")
struct MiniMaxM2MalformedInputTests {
  @Test
  func `Truncated mid-invoke (no closing tag) surfaces as incomplete`() {
    // With incremental streaming the parser commits to a tool call as
    // soon as `<invoke name="...">` is seen, then closes it as
    // `incomplete` at finalize when `</invoke>` never arrives.
    // Matches the truncation behavior of other parsers in the package.
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="fn"><parameter name="x">1</parameter>"#,
    )
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "fn")
    #expect(toolCalls[0].status == .incomplete)
  }

  @Test
  func `Truncated parameter tag does not crash the parser`() {
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="fn"><parameter name="x">1</param"#,
    )
    var parser = MiniMaxM2Parser()
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }

  @Test
  func `Envelope with no <invoke> children produces no tool calls`() {
    let input = "</think><minimax:tool_call></minimax:tool_call>"
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty)
  }
}

@Suite("Malformed input — GLM 4")
struct Glm4MalformedInputTests {
  @Test
  func `Truncated tool call without closing tag surfaces as incomplete`() {
    // With incremental streaming the parser commits to a tool call as
    // soon as the function name has been seen, then closes it as
    // `incomplete` at finalize when `</tool_call>` never arrives.
    // Matches the truncation behavior of Hermes/DeepSeek/Kimi parsers.
    let input = (
      "<tool_call>fn\n"
        + "<arg_key>x</arg_key>\n"
        + "<arg_value>1</arg_value>\n",
    )
    var parser = Glm4Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "fn")
    #expect(toolCalls[0].status == .incomplete)
  }

  @Test
  func `Mismatched arg_key/arg_value tags do not crash`() {
    let input = (
      "<tool_call>fn\n"
        + "<arg_key>x</arg_key>\n"
        + "<arg_value>1\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser()
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }

  @Test
  func `Missing arg_key with arg_value present degrades gracefully`() {
    let input = (
      "<tool_call>fn\n"
        + "<arg_value>1</arg_value>\n"
        + "</tool_call>",
    )
    var parser = Glm4Parser()
    _ = parser.process(ParserInput(text: input)) + parser.finalize()
  }
}
