// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("PythonicParser — plain text")
struct PythonicPlainTextTests {
  @Test
  func `Plain text without tool calls emits a single message`() {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "hello world")
  }

  @Test
  func `Empty stream emits nothing`() {
    var parser = PythonicParser()
    #expect(parser.finalize().isEmpty)
  }
}

@Suite("PythonicParser — single tool call")
struct PythonicSingleToolCallTests {
  @Test
  func `Single call with string argument`() throws {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"[get_weather(city="Paris")]"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Paris")
  }

  @Test
  func `Mixed argument types: int, float, bool, None, string`() throws {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(
      text: #"[fn(i=42, f=3.14, b=True, n=None, s='hello')]"#,
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["i"] as? Int == 42)
    #expect(decoded["f"] as? Double == 3.14)
    #expect(decoded["b"] as? Bool == true)
    #expect(decoded["n"] is NSNull)
    #expect(decoded["s"] as? String == "hello")
  }

  @Test
  func `Hex, octal, binary, and underscored integer literals match ast.literal_eval`() throws {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(
      text: #"[fn(hex=0xFF, oct=0o17, bin=0b1010, sep=1_000_000, neg=-0x10)]"#,
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["hex"] as? Int == 255)
    #expect(decoded["oct"] as? Int == 15)
    #expect(decoded["bin"] as? Int == 10)
    #expect(decoded["sep"] as? Int == 1_000_000)
    #expect(decoded["neg"] as? Int == -16)
  }

  @Test
  func `Decimal integer literals exceeding Int range preserve exact digits`() {
    let big = "123456789012345678901234567890"
    var parser = PythonicParser()
    let events = parser.process(ParserInput(
      text: "[fn(id=\(big), neg=-\(big))]",
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.arguments.contains(#""id": \#(big)"#))
    #expect(f.arguments.contains(#""neg": -\#(big)"#))
    #expect(isValidJSON(f.arguments))
  }

  @Test
  func `List argument value`() throws {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"[fn(items=[1, 2, 3])]"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["items"] as? [Int] == [1, 2, 3])
  }

  @Test
  func `Tuple argument value serializes as a JSON array`() throws {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"[fn(coords=(1, 2), wrapped=("x"))]"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["coords"] as? [Int] == [1, 2])
    #expect(decoded["wrapped"] as? String == "x")
  }

  @Test
  func `Tool call IDs are distinct fc_/call_ pairs`() {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"[fn()]"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.id.hasPrefix("fc_"))
    #expect(f.callId.hasPrefix("call_"))
  }
}

@Suite("PythonicParser — multiple tool calls")
struct PythonicMultipleToolCallTests {
  @Test
  func `Parallel calls in one bracket list`() {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"[f1(), f2(x=1)]"#)) + parser.finalize()
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
}

@Suite("PythonicParser — wrapper tokens")
struct PythonicWrapperTokenTests {
  @Test
  func `<|python_start|>...<|python_end|> wrappers are stripped`() {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(
      text: #"<|python_start|>[fn(x=1)]<|python_end|>"#,
    )) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "fn")
  }
}

@Suite("PythonicParser — surrounding text")
struct PythonicSurroundingTextTests {
  @Test
  func `Text before bracket list is emitted as message`() {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"Let me check. [fn(x=1)]"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
    guard case .functionCall = items[1] else { Issue.record("Expected function call"); return }
  }
}

@Suite("PythonicParser — finalize edge cases")
struct PythonicFinalizeTests {
  @Test
  func `Truncated bracket list is emitted as message text`() {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: "[fn(x=1, y=")) + parser.finalize()
    let items = accumulateItems(from: events)
    // No closing bracket — treated as plain text.
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }
}

@Suite("PythonicParser — dispatch")
struct PythonicDispatchTests {
  @Test
  func `ResponseFormat.pythonic.makeParser returns a working PythonicParser`() {
    let parser = ResponseFormat.pythonic.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let events = p.process(ParserInput(text: #"[fn(x=1)]"#)) + p.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
  }
}

@Suite("PythonicParser — adversarial ports")
struct PythonicAdversarialTests {
  // M3: vLLM test_streaming_tool_call_with_large_steps
  // (test_llama4_pythonic_tool_parser.py:227-245). Three calls in a single
  // delta, including a parameterless call and a call whose only argument
  // is an empty list.
  @Test
  func `Three calls in one delta with empty-list and parameterless variants`() throws {
    let input = (
      "<|python_start|>"
        + "[get_weather(city='LA', metric='C'), "
        + "get_weather(), "
        + "do_something_cool(steps=[])]"
        + "<|python_end|>",
    )
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 3)
    #expect(toolCalls[0].name == "get_weather")
    let args0Data = try #require(toolCalls[0].arguments.data(using: .utf8))
    let args0 = try #require(JSONSerialization.jsonObject(with: args0Data) as? [String: Any])
    #expect(args0["city"] as? String == "LA")
    #expect(args0["metric"] as? String == "C")
    #expect(toolCalls[1].name == "get_weather")
    // Parameterless call has empty args.
    let args1Data = try #require(toolCalls[1].arguments.data(using: .utf8))
    let args1 = try #require(JSONSerialization.jsonObject(with: args1Data) as? [String: Any])
    #expect(args1.isEmpty)
    #expect(toolCalls[2].name == "do_something_cool")
    let args2Data = try #require(toolCalls[2].arguments.data(using: .utf8))
    let args2 = try #require(JSONSerialization.jsonObject(with: args2Data) as? [String: Any])
    let steps = args2["steps"] as? [Any]
    #expect(steps?.isEmpty == true)
  }
}

@Suite("PythonicParser — string escapes")
struct PythonicStringEscapeTests {
  private func decodeArgs(_ raw: String) throws -> [String: Any] {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: raw)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call, got: \(items)")
      return [:]
    }
    return try JSONSerialization.jsonObject(with: f.arguments.data(using: .utf8)!) as! [String: Any]
  }

  @Test
  func `Common backslash escapes for n, t, r, backslash, double-quote, apostrophe`() throws {
    let args = try decodeArgs(#"[fn(s="a\nb\tc\rd\\e\"f")]"#)
    #expect(args["s"] as? String == "a\nb\tc\rd\\e\"f")
  }

  @Test
  func `Single-quoted string with escaped apostrophe (vLLM oracle)`() throws {
    let args = try decodeArgs(#"[get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')]"#)
    #expect(args["city"] as? String == "Martha's Vineyard")
    #expect(args["metric"] as? String == "\"cool units\"")
  }

  @Test
  func `Hex escape xHH form`() throws {
    let args = try decodeArgs(#"[fn(s="\x41\x42\x43")]"#)
    #expect(args["s"] as? String == "ABC")
  }

  @Test
  func `Unicode escape uHHHH form`() throws {
    let args = try decodeArgs(#"[fn(s="écafé")]"#)
    #expect(args["s"] as? String == "écafé")
  }

  @Test
  func `Long unicode escape UHHHHHHHH form`() throws {
    let args = try decodeArgs(#"[fn(s="\U0001F600")]"#)
    #expect(args["s"] as? String == "\u{1F600}")
  }

  @Test
  func `Octal escape (1-3 digits)`() throws {
    // \101 = 0o101 = 65 = 'A', \7 = bell, \101 'A'
    let args = try decodeArgs(#"[fn(s="\101\7\101")]"#)
    #expect(args["s"] as? String == "A\u{07}A")
  }

  @Test
  func `Bell, backspace, form feed, vertical tab`() throws {
    let args = try decodeArgs(#"[fn(s="\a\b\f\v")]"#)
    #expect(args["s"] as? String == "\u{07}\u{08}\u{0C}\u{0B}")
  }

  @Test
  func `Unknown escape preserves both characters (Python semantics)`() throws {
    // \d is not a recognized escape; Python keeps both chars.
    let args = try decodeArgs(#"[fn(s="\d\q")]"#)
    #expect(args["s"] as? String == "\\d\\q")
  }

  @Test
  func `Line continuation drops backslash and newline`() throws {
    let args = try decodeArgs("[fn(s=\"a\\\nb\")]")
    #expect(args["s"] as? String == "ab")
  }

  @Test
  func `Malformed u escape with too few hex digits is rejected`() {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"[fn(s="\u12")]"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    // Reference parsers raise on malformed escapes and the call falls
    // through to plain content; the Swift port matches that.
    for item in items {
      if case .functionCall = item {
        Issue.record("Malformed \\u should not produce a function call")
      }
    }
  }
}

@Suite("PythonicParser — bare numerics produce valid JSON")
struct PythonicNumericTests {
  private func decodeArgs(_ raw: String) throws -> [String: Any] {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: raw)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call, got: \(items)")
      return [:]
    }
    return try JSONSerialization.jsonObject(with: f.arguments.data(using: .utf8)!) as! [String: Any]
  }

  @Test
  func `Bare leading-decimal .5 re-serializes as JSON 0.5`() throws {
    let args = try decodeArgs(#"[fn(x=.5)]"#)
    // Either `0.5` or `0.5e+0` etc. — JSONSerialization round-trip works.
    #expect(args["x"] as? Double == 0.5)
  }

  @Test
  func `Scientific notation 1e10 round-trips through JSONSerialization`() throws {
    let args = try decodeArgs(#"[fn(x=1e10)]"#)
    #expect(args["x"] as? Double == 1e10)
  }

  @Test
  func `Lowercase JSON literals (true/false/null) are accepted alongside Python`() throws {
    // vLLM's `_JSON_NAME_LITERALS` accepts these mixed in pythonic
    // syntax. Some models emit them; the Swift port follows.
    let args = try decodeArgs(#"[fn(a=true, b=false, c=null)]"#)
    #expect(args["a"] as? Bool == true)
    #expect(args["b"] as? Bool == false)
    #expect(args["c"] is NSNull)
  }
}

@Suite("PythonicParser — streaming")
struct PythonicStreamingTests {
  @Test
  func `Char-by-char streaming reconstructs the same items as one-shot`() {
    assertStreamingReconstruction(
      #"<|python_start|>[get_weather(city="Paris", unit="celsius")]<|python_end|>"#,
      parser: { PythonicParser() },
    )
  }

  @Test
  func `Parametric stream intervals [1, 2, 4, 8] reconstruct the same args`() throws {
    let text = #"<|python_start|>[get_weather(city="San Francisco", unit="celsius")]<|python_end|>"#
    for interval in [1, 2, 4, 8] {
      let items = streamItems(text: text, interval: interval) { PythonicParser() }
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

  @Test
  func `Boolean args reconstruct correctly across stream intervals (regression-style)`() throws {
    // Mirrors the vLLM #19056 regression test for Hermes; pythonic
    // emits `True` / `False` but the JSON re-serialization and chunk
    // boundaries are still failure modes worth pinning.
    let text = #"<|python_start|>[final_answer(trigger=True)]<|python_end|>"#
    for interval in [1, 2, 5] {
      let items = streamItems(text: text, interval: interval) { PythonicParser() }
      guard case let .functionCall(f) = items.first else {
        Issue.record("interval=\(interval): expected function call, got \(items)")
        continue
      }
      let parsed = try JSONSerialization.jsonObject(
        with: Data(f.arguments.utf8),
      ) as? [String: Any]
      #expect(parsed?["trigger"] as? Bool == true, "interval=\(interval)")
    }
  }
}

@Suite("PythonicParser — chunk boundary cases")
struct PythonicBoundaryTests {
  @Test
  func `<|python_start|> marker split across chunks does not leak`() {
    var parser = PythonicParser()
    var events = parser.process(ParserInput(text: "<|py"))
    events += parser.process(ParserInput(text: "thon_start|>[fn(x=1)]<|python_end|>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1, "Expected one function_call after marker split, got \(items)")
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call, got \(items)"); return
    }
    #expect(f.name == "fn")
  }

  @Test
  func `<|python_end|> marker split across chunks does not leak`() {
    var parser = PythonicParser()
    var events = parser.process(ParserInput(text: "<|python_start|>[fn(x=1)]<|python_"))
    events += parser.process(ParserInput(text: "end|>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1, "Expected one function_call after end-marker split, got \(items)")
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call, got \(items)"); return
    }
    #expect(f.name == "fn")
  }

  @Test
  func `Nested bracket argument value parses across chunk boundaries`() throws {
    var parser = PythonicParser()
    let events = parser.process(ParserInput(text: #"[fn(items=[1, 2, 3])]"#)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call"); return
    }
    let parsed = try JSONSerialization.jsonObject(
      with: Data(f.arguments.utf8),
    ) as? [String: Any]
    #expect(parsed?["items"] as? [Int] == [1, 2, 3])
  }

  @Test
  func `Nested bracket argument value split mid-bracket reconstructs correctly`() throws {
    var parser = PythonicParser()
    var events = parser.process(ParserInput(text: "[fn(items=[1, 2,"))
    events += parser.process(ParserInput(text: " 3])]"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1, "Expected one function_call across nested-bracket split, got \(items)")
    guard case let .functionCall(f) = items.first else {
      Issue.record("Expected function call, got \(items)"); return
    }
    let parsed = try JSONSerialization.jsonObject(
      with: Data(f.arguments.utf8),
    ) as? [String: Any]
    #expect(parsed?["items"] as? [Int] == [1, 2, 3])
  }
}
