// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Granite 4 reuses the Hermes parser configured with the
// `argumentsMayBeJSONString` flag. These tests focus on the variant
// behavior:
//
// 1. Object-shaped `arguments` flow through as standard Hermes (the
//    flag is permissive, not exclusive).
// 2. String-encoded `arguments` are decoded so the emitted args text
//    matches the canonical object form.
// 3. Streaming defers emission for string-encoded args until the
//    region closes, since the wire bytes don't match the canonical
//    bytes mid-stream.

@Suite("Granite4Parser — variant behavior")
struct Granite4VariantTests {
  @Test
  func `Object-shaped arguments behave as standard Hermes`() throws {
    let input = #"<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>"#
    let parser = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: input)) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }

  @Test
  func `String-encoded arguments are decoded to canonical object form`() throws {
    // `arguments` is a JSON-encoded string containing the canonical
    // object. Granite 4 emits this form when its template
    // pre-stringifies arguments. The parser must decode the string so
    // downstream consumers see `{"city":"Tokyo"}`, not the
    // doubly-encoded `"{\"city\":\"Tokyo\"}"`.
    let inner = #"{"city":"Tokyo"}"#
    let outer: [String: Any] = ["name": "get_weather", "arguments": inner]
    let outerData = try JSONSerialization.data(withJSONObject: outer, options: [])
    let outerJSON = try #require(String(data: outerData, encoding: .utf8))
    let input = "<tool_call>" + outerJSON + "</tool_call>"

    var p = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    let events = p.process(ParserInput(text: input)) + p.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["city"] as? String == "Tokyo")
  }

  @Test
  func `String-encoded arguments containing nested arrays decode correctly`() throws {
    // Mirrors vLLM's `test_granite4_tool_parser.py` complex fixture:
    // `coord_arg` is a dict-with-nested-array that gets json.dumped
    // when the test enables `create_string_args`.
    let coord: [String: Any] = [
      "coordinates": [[23.54, 43.1], [-12.2, 54.3], [4, 5]],
      "coordinate_type": "latlong",
    ]
    let coordData = try JSONSerialization.data(withJSONObject: coord, options: [.sortedKeys])
    let coordJSON = try #require(String(data: coordData, encoding: .utf8))
    let outer: [String: Any] = ["name": "find_bbox", "arguments": coordJSON]
    let outerData = try JSONSerialization.data(withJSONObject: outer, options: [])
    let outerJSON = try #require(String(data: outerData, encoding: .utf8))
    let input = "<tool_call>" + outerJSON + "</tool_call>"

    var p = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    let events = p.process(ParserInput(text: input)) + p.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let data = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(decoded["coordinate_type"] as? String == "latlong")
    let coords = try #require(decoded["coordinates"] as? [[Double]])
    #expect(coords.count == 3)
    #expect(coords[0] == [23.54, 43.1])
  }

  @Test
  func `Mixed object and string-encoded args across parallel calls`() throws {
    // The complex fixture has alternating string and object-shaped
    // args within one response. Both should resolve to the canonical
    // object form.
    let stringEncoded = #"{"name":"a","arguments":"{\"x\":1}"}"#
    let objectShaped = #"{"name":"b","arguments":{"y":2}}"#
    let input = """
    Some prose. <tool_call>\(stringEncoded)</tool_call>
    More prose. <tool_call>\(objectShaped)</tool_call>
    Trailing.
    """

    var p = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    let events = p.process(ParserInput(text: input)) + p.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    let argsAData = try #require(toolCalls[0].arguments.data(using: .utf8))
    let argsA = try #require(JSONSerialization.jsonObject(with: argsAData) as? [String: Any])
    #expect(argsA["x"] as? Int == 1)
    let argsBData = try #require(toolCalls[1].arguments.data(using: .utf8))
    let argsB = try #require(JSONSerialization.jsonObject(with: argsBData) as? [String: Any])
    #expect(argsB["y"] as? Int == 2)
  }

  @Test
  func `Complex mixed content and tool calls preserve all text segments`() throws {
    // Mirrors vLLM's `test_granite4_tool_parser.py` complex fixture,
    // including text before, between, and after tool calls.
    let coord: [String: Any] = [
      "coordinates": [[23.54, 43.1], [-12.2, 54.3], [4, 5]],
      "coordinate_type": "latlong",
    ]
    let coordData = try JSONSerialization.data(withJSONObject: coord, options: [.sortedKeys])
    let coordJSON = try #require(String(data: coordData, encoding: .utf8))
    let inputDicts: [[String: Any]] = [
      ["name": "find_bbox", "arguments": coordJSON],
      [
        "name": "get_stock_price",
        "arguments": [
          "symbol": "AAPL",
          "start_date": "2021-01-01",
          "end_date": "2021-12-31",
        ],
      ],
      ["name": "find_bbox", "arguments": coordJSON],
    ]
    let formattedCalls = try inputDicts.map { call in
      let data = try JSONSerialization.data(withJSONObject: call, options: [])
      return "<tool_call> " + (String(data: data, encoding: .utf8) ?? "") + " </tool_call>"
    }
    let textMessages = [
      "Here goes the bbox call: \n",
      " Now the stock price call: \n ",
      " Now another bbox call: \n ",
      " See? I'm a helpful assistant.",
    ]
    let input = textMessages[0] + formattedCalls[0]
      + textMessages[1] + formattedCalls[1]
      + textMessages[2] + formattedCalls[2]
      + textMessages[3]

    var p = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    let events = p.process(ParserInput(text: input)) + p.finalize()
    let items = accumulateItems(from: events)

    let content = items.compactMap { item -> String? in
      guard case let .message(message) = item,
            case let .outputText(text) = message.content.first
      else { return nil }
      return text.text
    }.joined()
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }

    #expect(content == textMessages.joined())
    #expect(toolCalls.map(\.name) == ["find_bbox", "get_stock_price", "find_bbox"])
    let firstArgsData = try #require(toolCalls.first?.arguments.data(using: .utf8))
    let firstArgs = try #require(JSONSerialization.jsonObject(with: firstArgsData) as? [String: Any])
    #expect(firstArgs["coordinate_type"] as? String == "latlong")
    let secondArgsData = try #require(toolCalls.dropFirst().first?.arguments.data(using: .utf8))
    let secondArgs = try #require(JSONSerialization.jsonObject(with: secondArgsData) as? [String: Any])
    #expect(secondArgs["symbol"] as? String == "AAPL")
  }
}

@Suite("Granite4Parser — streaming")
struct Granite4StreamingTests {
  @Test
  func `Char-by-char string-encoded args reconstructs to canonical object`() throws {
    let inner = #"{"city":"Tokyo"}"#
    let outer: [String: Any] = ["name": "get_weather", "arguments": inner]
    let outerData = try JSONSerialization.data(withJSONObject: outer, options: [])
    let outerJSON = try #require(String(data: outerData, encoding: .utf8))
    let input = "<tool_call>" + outerJSON + "</tool_call>"

    var oneShot = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    guard case let .functionCall(a) = oneShotItems[0],
          case let .functionCall(b) = streamedItems[0]
    else {
      Issue.record("Expected function calls"); return
    }
    #expect(a.arguments == b.arguments)
    let aData = try #require(a.arguments.data(using: .utf8))
    let aDict = try #require(JSONSerialization.jsonObject(with: aData) as? [String: Any])
    #expect(aDict["city"] as? String == "Tokyo")
  }

  @Test
  func `String-encoded args do not leak wire bytes mid-stream`() {
    // The wire form contains escaped quotes (`\"`); the canonical form
    // doesn't. If the parser emitted intermediate wire bytes as deltas
    // before close, downstream consumers would see escaped output then
    // the canonical, which is broken. Verify the only delta is the
    // canonical text.
    let input = #"<tool_call>{"name":"f","arguments":"{\"x\":1}"}</tool_call>"#

    var streamed = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    var deltaTexts: [String] = []
    for ch in input {
      let evs = streamed.process(ParserInput(text: String(ch)))
      for ev in evs {
        if case let .functionCallArgumentsDelta(d) = ev {
          deltaTexts.append(d.delta)
        }
      }
    }
    let evs = streamed.finalize()
    for ev in evs {
      if case let .functionCallArgumentsDelta(d) = ev {
        deltaTexts.append(d.delta)
      }
    }
    let combined = deltaTexts.joined()
    #expect(combined == #"{"x":1}"#, "Expected only the canonical decoded args")
    #expect(!combined.contains("\\\""), "Escaped quotes from the wire form must not leak")
  }

  @Test
  func `Fixed chunks preserve canonical string-encoded argument delta`() {
    let chunks = [
      "Before ",
      "<tool_call>",
      #"{"name":"f","arguments":"{\"x\":"#,
      #"1}"#,
      #""}"#,
      "</tool_call>",
      " after",
    ]
    var parser = ResponseFormat.granite4.makeParser(tokenizer: StubTokenizer())
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()

    let argsDeltas = events.compactMap {
      if case let .functionCallArgumentsDelta(e) = $0 { return e.delta }
      return nil
    }
    let textDeltas = events.compactMap {
      if case let .outputTextDelta(e) = $0 { return e.delta }
      return nil
    }
    #expect(argsDeltas == [#"{"x":1}"#])
    #expect(textDeltas == ["Before ", " after"])
  }
}

@Suite("ResponseFormat dispatch — Granite 4")
struct Granite4DispatchTests {
  @Test
  func `Granite 4 routes to .granite4 by name`() {
    let f = ResponseFormat.infer(
      modelName: "ibm-granite/granite-4.0-h-tiny",
      modelType: "granitemoehybrid",
      modelConfig: [:],
    )
    #expect(f == .granite4)
  }

  @Test
  func `Bare granite-4 prefix also routes to .granite4`() {
    let f = ResponseFormat.infer(
      modelName: "granite-4.0-h-small",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .granite4)
  }

  @Test
  func `Granite 3.x and Granite 4 dispatch to distinct formats`() {
    let f3 = ResponseFormat.infer(
      modelName: "ibm-granite/granite-3.1-8b-instruct",
      modelType: "",
      modelConfig: [:],
    )
    let f4 = ResponseFormat.infer(
      modelName: "ibm-granite/granite-4.0-h-tiny",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f3 == .granite)
    #expect(f4 == .granite4)
    #expect(f3 != f4)
  }

  @Test
  func `Granite 4 hybrid routes to .granite4 by model_type when name is unhelpful`() {
    // Covers the case where the loader has rewritten the model
    // configuration to a directory-based id (e.g. `snapshots/<hash>`),
    // so name-prefix matching can't fire and we have to fall through to
    // model_type. `granitemoehybrid` is the HF arch for Granite 4.0 H
    // Tiny / H Small.
    let f = ResponseFormat.infer(
      modelName: "snapshots/a892ded1552d6d4089fa644bbff6ccbc54dddc67",
      modelType: "granitemoehybrid",
      modelConfig: [:],
    )
    #expect(f == .granite4)
  }

  @Test
  func `Granite 3.x routes to .granite by model_type when name is unhelpful`() {
    let granite = ResponseFormat.infer(
      modelName: "snapshots/abc",
      modelType: "granite",
      modelConfig: [:],
    )
    let graniteMoE = ResponseFormat.infer(
      modelName: "snapshots/abc",
      modelType: "granitemoe",
      modelConfig: [:],
    )
    #expect(granite == .granite)
    #expect(graniteMoE == .granite)
  }
}
