// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

@Suite("MiniMaxM2Parser — reasoning")
struct MiniMaxM2ReasoningTests {
  @Test
  func `Default initial state is reasoning; content before </think> is reasoning`() {
    let input = "Thinking out loud.</think>The answer."
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record(""); return }
    guard case let .reasoningText(rPart) = r.content[0] else { Issue.record(""); return }
    #expect(rPart.text == "Thinking out loud.")
    guard case let .message(m) = items[1] else { Issue.record(""); return }
    guard case let .outputText(mPart) = m.content[0] else { Issue.record(""); return }
    #expect(mPart.text == "The answer.")
  }

  @Test
  func `Plain text only emits a reasoning item if it never sees </think>`() {
    let input = "Just thinking, no closer."
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record(""); return }
    #expect(r.status == .incomplete)
  }

  @Test
  func `Initial state .normal skips reasoning`() {
    var parser = MiniMaxM2Parser(initialState: .normal)
    let events = parser.process(ParserInput(text: "Just an answer.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record(""); return }
  }
}

@Suite("MiniMaxM2Parser — tool calls")
struct MiniMaxM2ToolCallTests {
  @Test
  func `Single tool call with one parameter (no schema → string)`() throws {
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="get_weather">"#
        + #"<parameter name="city">Seattle</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Seattle")
  }

  @Test
  func `Schema-driven coercion turns string 5 into integer 5`() throws {
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
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="get_weather">"#
        + #"<parameter name="city">NYC</parameter>"#
        + #"<parameter name="days">5</parameter>"#
        + #"<parameter name="active">true</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
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
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="fn">"#
        + #"<parameter name="count">100000000000000000000</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    #expect(f.arguments.contains("100000000000000000000"))
    #expect(!f.arguments.contains("\"100000000000000000000\""))
  }

  @Test
  func `Without schema, all parameter values are emitted as JSON strings`() throws {
    // vLLM and SGLang's `_get_param_types_from_config` default to
    // `["string"]` when the parameter is not in the tool's schema, so
    // the value is always emitted as a JSON string regardless of
    // whether it parses as JSON. Pinned by vLLM's
    // `test_header_and_params_in_separate_chunks` which asserts
    // `"days": "5"` for an unschema'd numeric parameter.
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="fn">"#
        + #"<parameter name="count">5</parameter>"#
        + #"<parameter name="flag">true</parameter>"#
        + #"<parameter name="items">[1, 2, 3]</parameter>"#
        + #"<parameter name="label">Beijing</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["count"] as? String == "5")
    #expect(decoded["flag"] as? String == "true")
    #expect(decoded["items"] as? String == "[1, 2, 3]")
    #expect(decoded["label"] as? String == "Beijing")
  }

  @Test
  func `Two invokes in one envelope`() {
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="f1"><parameter name="x">1</parameter></invoke>"#
        + #"<invoke name="f2"><parameter name="y">2</parameter></invoke>"#
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser()
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
      "</think>"
        + "Let me check. "
        + "<minimax:tool_call>"
        + #"<invoke name="f"><parameter name="x">1</parameter></invoke>"#
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record(""); return }
    guard case let .outputText(mPart) = m.content[0] else { Issue.record(""); return }
    #expect(mPart.text == "Let me check. ")
    guard case .functionCall = items[1] else { Issue.record(""); return }
  }

  @Test
  func `null value is preserved as JSON null`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "fn",
        "parameters": [
          "properties": [
            "nick": ["type": "string"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="fn"><parameter name="nick">null</parameter></invoke>"#
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["nick"] is NSNull)
  }

  @Test
  func `Object-typed parameter accepts JSON inline`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "fn",
        "parameters": [
          "properties": [
            "config": ["type": "object"] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="fn"><parameter name="config">{"theme":"dark"}</parameter></invoke>"#
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record(""); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let config = decoded["config"] as? [String: Any]
    #expect(config?["theme"] as? String == "dark")
  }
}

@Suite("MiniMaxM2Parser — streaming")
struct MiniMaxM2StreamingTests {
  @Test
  func `Char-by-char streaming reconstructs the same items as one-shot`() {
    let input = (
      "Reasoning."
        + "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="f"><parameter name="x">1</parameter></invoke>"#
        + "</minimax:tool_call>",
    )

    var streaming = MiniMaxM2Parser()
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    var oneShot = MiniMaxM2Parser()
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
          Issue.record(""); return
      }
    }
  }

  @Test
  func `Fixed chunks preserve exact reasoning text and string argument deltas`() {
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
      "think",
      "ing</think>prefix ",
      "<minimax",
      ":tool_call>",
      #"<invoke name="fn"><parameter name="text">he"#,
      "llo",
      "</parameter></invoke></minimax:tool_call>",
      " suffix",
    ]

    var parser = MiniMaxM2Parser(tools: tools)
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let outputIndexes = events.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }
    #expect(outputIndexes == [0, 1, 2, 3])
    #expect(miniMaxM2ReasoningDeltas(from: events) == ["think", "ing"])
    #expect(miniMaxM2OutputTextDeltas(from: events) == ["prefix ", " suffix"])
    #expect(miniMaxM2ArgumentDeltas(from: events) == [#"{"text": "he"#, "llo", "\"}"])
  }

  @Test
  func `Closed invoke followed by text and later invoke preserves exact deltas`() {
    let chunks = [
      "</think><minimax:tool_call>"
        + #"<invoke name="first"><parameter name="a">1</parameter></invoke>"#
        + "</minimax:tool_call>",
      " gap ",
      "<minimax:tool_call>"
        + #"<invoke name="second"><parameter name="b">2</parameter></invoke>"#
        + "</minimax:tool_call>",
    ]

    var parser = MiniMaxM2Parser()
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
    #expect(miniMaxM2OutputTextDeltas(from: events) == [" gap "])
    #expect(miniMaxM2ArgumentDeltas(from: events) == [#"{"a": "1"}"#, #"{"b": "2"}"#])
  }
}

@Suite("MiniMaxM2Parser — incremental streaming")
struct MiniMaxM2IncrementalStreamingTests {
  // Mirrors the H7 test for GLM 4: a long string parameter value fed
  // line-by-line should reach the consumer in many fragments, not in one
  // burst at `</invoke>`. Asserts ≥10 fragments for the multi-line code
  // fixture (matching vLLM's threshold for the same pattern).
  @Test
  func `Long string parameter streams as ≥10 incremental fragments`() throws {
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
      "</think>",
      "<minimax:tool_call>",
      #"<invoke name="write_to_file">"#,
      #"<parameter name="file_path">/tmp/bubble_sort.py</parameter>"#,
      #"<parameter name="content">"#,
    ]
    for line in bubbleSortCode.split(separator: "\n", omittingEmptySubsequences: false) {
      chunks.append(String(line) + "\n")
    }
    chunks.append("</parameter>")
    chunks.append("</invoke>")
    chunks.append("</minimax:tool_call>")

    var parser = MiniMaxM2Parser(tools: tools)
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let argsDeltas = events.compactMap { ev -> String? in
      if case let .functionCallArgumentsDelta(e) = ev { return e.delta } else { return nil }
    }
    #expect(argsDeltas.count >= 10)

    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["file_path"] as? String == "/tmp/bubble_sort.py")
    #expect((decoded["content"] as? String)?.contains("def bubble_sort") == true)
  }

  @Test
  func `Two invokes in one envelope each open and close in order`() {
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="f1"><parameter name="x">1</parameter></invoke>"#
        + #"<invoke name="f2"><parameter name="y">2</parameter></invoke>"#
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let outputItemAddedNames = events.compactMap { ev -> String? in
      if case let .outputItemAdded(e) = ev,
         case let .functionCall(f) = e.item { return f.name } else { return nil }
    }
    let outputItemDoneNames = events.compactMap { ev -> String? in
      if case let .outputItemDone(e) = ev,
         case let .functionCall(f) = e.item { return f.name } else { return nil }
    }
    #expect(outputItemAddedNames == ["f1", "f2"])
    #expect(outputItemDoneNames == ["f1", "f2"])
  }
}

@Suite("MiniMaxM2Parser — dispatch")
struct MiniMaxM2DispatchTests {
  @Test
  func `Dispatch via ResponseFormat.miniMaxM2.makeParser`() {
    let parser = ResponseFormat.miniMaxM2.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: "thinking</think>")) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .reasoning = items[0] else { Issue.record(""); return }
  }

  @Test
  func `Name prefix minimax-m2 routes to .miniMaxM2`() {
    let format = ResponseFormat.infer(modelName: "minimax-m2-instruct", modelType: "", modelConfig: [:])
    #expect(format == .miniMaxM2)
  }

  @Test
  func `priorOutput with </think> starts in normal phase`() {
    let parser = ResponseFormat.miniMaxM2.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "previous reasoning</think>previous answer",
    )
    var p = parser
    let events = p.process(ParserInput(text: "more answer")) + p.finalize()
    let items = accumulateItems(from: events)
    // No reasoning item: we resume in normal phase.
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }
}

@Suite("MiniMaxM2Parser — anyOf nullable parameters")
struct MiniMaxM2AnyOfNullableTests {
  // H2: vLLM TestAnyOfNullableParam
  // (test_minimax_m2_tool_parser.py:451-549). Verifies that a parameter
  // declared `anyOf: [{type: T}, {type: null}]` coerces correctly for a
  // non-null T-typed value, the literal `null`, and (for object T) an
  // inline JSON value.

  @Test
  func `anyOf [string, null] preserves a non-null string value`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "update_profile",
        "parameters": [
          "type": "object",
          "properties": [
            "nickname": [
              "anyOf": [
                ["type": "string"] as [String: any Sendable],
                ["type": "null"] as [String: any Sendable],
              ] as [any Sendable],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="update_profile">"#
        + #"<parameter name="nickname">Alice</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["nickname"] as? String == "Alice")
  }

  @Test
  func `anyOf [string, null] collapses literal null to JSON null`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "update_profile",
        "parameters": [
          "type": "object",
          "properties": [
            "nickname": [
              "anyOf": [
                ["type": "string"] as [String: any Sendable],
                ["type": "null"] as [String: any Sendable],
              ] as [any Sendable],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="update_profile">"#
        + #"<parameter name="nickname">null</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["nickname"] is NSNull)
  }

  @Test
  func `anyOf [object, null] parses an inline JSON value as a dict, not a string`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "update_settings",
        "parameters": [
          "type": "object",
          "properties": [
            "config": [
              "anyOf": [
                ["type": "object"] as [String: any Sendable],
                ["type": "null"] as [String: any Sendable],
              ] as [any Sendable],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="update_settings">"#
        + #"<parameter name="config">{"theme": "dark", "fontSize": 14}</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    let config = decoded["config"] as? [String: Any]
    #expect(config?["theme"] as? String == "dark")
    #expect(config?["fontSize"] as? Int == 14)
  }
}

@Suite("MiniMaxM2Parser — schema type priority")
struct MiniMaxM2TypePriorityTests {
  // sglang's `_convert_param_value_with_types` tries integer/number/
  // boolean/object/array before falling back to string. For mixed
  // schemas where both string and a non-string type are valid, this
  // determines whether `"5"` ends up as `5` (integer) or `"5"` (string).
  //
  // Pre-fix the Swift parser collapsed the schema to a single type via
  // `inferTypeFromJsonSchema`, which preferred `string` for mixed
  // anyOf branches and so emitted `"5"` instead of `5`. Now it
  // honors sglang's priority order.
  @Test
  func `anyOf [string, integer] coerces numeric value to integer (priority over string)`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "lookup",
        "parameters": [
          "type": "object",
          "properties": [
            "id": [
              "anyOf": [
                ["type": "string"] as [String: any Sendable],
                ["type": "integer"] as [String: any Sendable],
              ] as [any Sendable],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="lookup">"#
        + #"<parameter name="id">5</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["id"] as? Int == 5)
  }

  @Test
  func `anyOf [string, integer] keeps non-numeric value as string (integer fails, falls back)`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "lookup",
        "parameters": [
          "type": "object",
          "properties": [
            "id": [
              "anyOf": [
                ["type": "string"] as [String: any Sendable],
                ["type": "integer"] as [String: any Sendable],
              ] as [any Sendable],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="lookup">"#
        + #"<parameter name="id">abc</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["id"] as? String == "abc")
  }

  @Test
  func `type: [integer, null] array coerces numeric value to integer`() throws {
    let tools: [ToolSpec] = [[
      "function": [
        "name": "lookup",
        "parameters": [
          "type": "object",
          "properties": [
            "n": [
              "type": ["integer", "null"] as [any Sendable],
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ] as [String: any Sendable],
    ]]
    let input = (
      "</think>"
        + "<minimax:tool_call>"
        + #"<invoke name="lookup">"#
        + #"<parameter name="n">42</parameter>"#
        + "</invoke>"
        + "</minimax:tool_call>",
    )
    var parser = MiniMaxM2Parser(tools: tools)
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["n"] as? Int == 42)
  }
}

private func miniMaxM2ReasoningDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .reasoningDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

private func miniMaxM2OutputTextDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .outputTextDelta(e) = event {
      return e.delta
    }
    return nil
  }
}

private func miniMaxM2ArgumentDeltas(from events: [ResponseStreamingEvent]) -> [String] {
  events.compactMap { event in
    if case let .functionCallArgumentsDelta(e) = event {
      return e.delta
    }
    return nil
  }
}
