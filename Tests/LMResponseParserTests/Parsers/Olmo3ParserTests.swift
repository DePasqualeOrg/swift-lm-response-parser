// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// OLMo 3 reuses PythonicParser with `<function_calls>` / `</function_calls>`
// wrapper tokens and newline-separated calls (no enclosing `[...]`).
// Argument values may use either Python literals (`True`/`False`/`None`)
// or JSON literals (`true`/`false`/`null`). These tests pin behavior
// against vLLM's `Olmo3PythonicToolParser` fixtures.

private func makeOlmo3Parser() -> PythonicParser {
  PythonicParser(
    startTag: "<function_calls>",
    endTag: "</function_calls>",
    newlineSeparated: true,
  )
}

@Suite("Olmo3 — wrapper and inner shape")
struct Olmo3WrapperTests {
  @Test
  func `Single call inside <function_calls>...</function_calls>`() throws {
    var parser = makeOlmo3Parser()
    let input = "<function_calls>get_weather(city='San Francisco', metric='celsius')</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "San Francisco")
    #expect(decoded["metric"] as? String == "celsius")
  }

  @Test
  func `Parallel calls separated by newlines`() {
    var parser = makeOlmo3Parser()
    let input = """
    <function_calls>get_weather(city='San Francisco', metric='celsius')
    register_user(name='John Doe', age=37)</function_calls>
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
    #expect(b.name == "register_user")
  }

  @Test
  func `Parameterless call returns empty arguments object`() {
    var parser = makeOlmo3Parser()
    let input = "<function_calls>get_weather()</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
    #expect(f.arguments == "{}")
  }

  @Test
  func `Plain text without the wrapper passes through as a message`() {
    var parser = makeOlmo3Parser()
    let events = parser.process(ParserInput(text: "How can I help you today?")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "How can I help you today?")
  }

  @Test
  func `Bare bracket-list without wrapper is NOT recognized as tool call`() {
    var parser = makeOlmo3Parser()
    let input = "[get_weather(city='Paris')]"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == input)
  }

  @Test
  func `Leading text before envelope emits as message then tool call`() {
    var parser = makeOlmo3Parser()
    let input = "Sure, calling: <function_calls>get_weather(city='Paris')</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .message(m) = items[0] else { Issue.record("Expected leading message"); return }
    guard case let .outputText(t) = m.content[0] else { Issue.record("Expected text content"); return }
    #expect(t.text == "Sure, calling: ")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }
}

@Suite("Olmo3 — argument literal kinds")
struct Olmo3LiteralKindTests {
  @Test
  func `Python literals: True, False, None`() throws {
    var parser = makeOlmo3Parser()
    let input = "<function_calls>register_user(passed_test=True, role=None, archived=False)</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["passed_test"] as? Bool == true)
    #expect(decoded["role"] is NSNull)
    #expect(decoded["archived"] as? Bool == false)
  }

  @Test
  func `JSON literals: true, false, null are accepted`() throws {
    var parser = makeOlmo3Parser()
    let input = "<function_calls>register_user(passed_test=true, role=null, archived=false)</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["passed_test"] as? Bool == true)
    #expect(decoded["role"] is NSNull)
    #expect(decoded["archived"] as? Bool == false)
  }

  @Test
  func `Mixed Python and JSON literals in the same call`() throws {
    var parser = makeOlmo3Parser()
    let input = "<function_calls>fn(a=True, b=null, c=False, d=None, e=true)</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["a"] as? Bool == true)
    #expect(decoded["b"] is NSNull)
    #expect(decoded["c"] as? Bool == false)
    #expect(decoded["d"] is NSNull)
    #expect(decoded["e"] as? Bool == true)
  }

  @Test
  func `Nested dict and list arguments`() throws {
    var parser = makeOlmo3Parser()
    let input = "<function_calls>register_user(name='John Doe', age=37, address={'city': 'San Francisco', 'state': 'CA'}, aliases=['John', 'Johnny'])</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["name"] as? String == "John Doe")
    #expect(decoded["age"] as? Int == 37)
    let addr = try #require(decoded["address"] as? [String: Any])
    #expect(addr["city"] as? String == "San Francisco")
    let aliases = try #require(decoded["aliases"] as? [String])
    #expect(aliases == ["John", "Johnny"])
  }

  @Test
  func `Escaped strings inside arguments`() throws {
    var parser = makeOlmo3Parser()
    let input = #"<function_calls>get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')</function_calls>"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    let decodedData = try #require(f.arguments.data(using: .utf8))
    let decoded = try #require(JSONSerialization.jsonObject(with: decodedData) as? [String: Any])
    #expect(decoded["city"] as? String == "Martha's Vineyard")
    #expect(decoded["metric"] as? String == #""cool units""#)
  }
}

@Suite("Olmo3 — streaming")
struct Olmo3StreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = """
    <function_calls>get_weather(city='San Francisco', metric='celsius')
    register_user(name='John Doe', age=37)</function_calls>
    """

    var oneShot = makeOlmo3Parser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = makeOlmo3Parser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    #expect(oneShotItems.count == 2)
    for (a, b) in zip(oneShotItems, streamedItems) {
      guard case let .functionCall(fa) = a,
            case let .functionCall(fb) = b
      else {
        Issue.record("Expected function call pair"); continue
      }
      #expect(fa.name == fb.name)
      #expect(fa.arguments == fb.arguments)
    }
  }

  @Test
  func `Split <function_calls> opener across chunks does not leak as content`() {
    // Split the opener mid-tag. The buffer must hold the partial
    // prefix until the next chunk completes the marker.
    var parser = makeOlmo3Parser()
    var events = parser.process(ParserInput(text: "<func"))
    events += parser.process(ParserInput(text: "tion_calls>get_weather(city='Paris')</function_calls>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Split </function_calls> closer across chunks does not leak`() {
    var parser = makeOlmo3Parser()
    var events = parser.process(ParserInput(text: "<function_calls>get_weather(city='Paris')</func"))
    events += parser.process(ParserInput(text: "tion_calls>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCall(f) = items[0] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Unclosed envelope at end-of-stream falls back to message text`() {
    var parser = makeOlmo3Parser()
    // Open the wrapper but never close it.
    let events = parser.process(ParserInput(text: "<function_calls>get_weather(city='Paris')")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message fallback"); return
    }
    #expect(t.text == "<function_calls>get_weather(city='Paris')")
  }

  @Test
  func `Streaming with split arguments inside the envelope`() {
    // Mirrors vLLM's `test_streaming_tool_call_with_large_steps`: the
    // arguments to the first call are split mid-string across two
    // process() chunks. The parser holds back until `</function_calls>`
    // arrives in the second chunk.
    var parser = makeOlmo3Parser()
    var events = parser.process(ParserInput(text: "<function_calls>get_weather(city='San"))
    events += parser.process(ParserInput(text: " Francisco', metric='celsius')\nget_weather()</function_calls>"))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .functionCall(a) = items[0],
          case let .functionCall(b) = items[1]
    else {
      Issue.record("Expected two function calls"); return
    }
    #expect(a.name == "get_weather")
    #expect(b.name == "get_weather")
    #expect(b.arguments == "{}")
  }
}

// OLMo 3 reasoning is opt-in via `acceptThink: true`. The shape is
// `<think>...</think>` where the markers are vocabulary tokens. Some
// chat templates inject `<think>` into the prompt so the model emits
// only `</think>` — the parser tolerates both. Fixtures mirror vLLM's
// `test_olmo3_reasoning_parser.py`.

private func makeOlmo3ThinkParser(
  initialState: PythonicParser.InitialState = .reasoning,
) -> PythonicParser {
  PythonicParser(
    startTag: "<function_calls>",
    endTag: "</function_calls>",
    newlineSeparated: true,
    acceptThink: true,
    initialState: initialState,
  )
}

@Suite("Olmo3 — reasoning preamble")
struct Olmo3ReasoningTests {
  @Test
  func `Reasoning then content (with explicit opener) emits a reasoning item then a message`() {
    var parser = makeOlmo3ThinkParser()
    let input = "<think>This is a reasoning section</think>This is the rest"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "This is a reasoning section")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "This is the rest")
  }

  @Test
  func `Content before explicit reasoning marker is preserved`() {
    var parser = makeOlmo3ThinkParser()
    let input = "Before <think>thinking</think>After"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 3)
    guard case let .message(before) = items[0],
          case let .outputText(beforeText) = before.content[0]
    else {
      Issue.record("Expected leading message"); return
    }
    #expect(beforeText.text == "Before ")
    guard case let .reasoning(r) = items[1] else {
      Issue.record("Expected reasoning"); return
    }
    #expect(r.text == "thinking")
    guard case let .message(after) = items[2],
          case let .outputText(afterText) = after.content[0]
    else {
      Issue.record("Expected trailing message"); return
    }
    #expect(afterText.text == "After")
  }

  @Test
  func `Reasoning closing only (no opener; chat template injected it)`() {
    var parser = makeOlmo3ThinkParser()
    let input = "The user is asking me not to think.</think>No thoughts!"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "The user is asking me not to think.")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "No thoughts!")
  }

  @Test
  func `Empty reasoning with only closing marker emits content with leading newlines`() {
    var parser = makeOlmo3ThinkParser()
    let input = "</think>\n\nNo thoughts, head empty!"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message only"); return
    }
    #expect(t.text == "\n\nNo thoughts, head empty!")
  }

  @Test
  func `Empty reasoning between markers emits no reasoning item`() {
    var parser = makeOlmo3ThinkParser()
    let input = "<think></think>No thoughts, head empty!"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message only"); return
    }
    #expect(t.text == "No thoughts, head empty!")
  }

  @Test
  func `Newlines around the closer are preserved as content (matches vLLM)`() {
    var parser = makeOlmo3ThinkParser()
    let input = "<think>\n</think>\n\nNo thoughts, head empty!"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "\n")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "\n\nNo thoughts, head empty!")
  }

  @Test
  func `Multiple newlines around reasoning close are preserved`() {
    var parser = makeOlmo3ThinkParser()
    let input = "<think>\nLook!\nI'm thinking...\n\n</think>\n\n\nThis is the rest"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "\nLook!\nI'm thinking...\n\n")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "\n\n\nThis is the rest")
  }

  @Test
  func `Reasoning followed by tool call emits both`() {
    var parser = makeOlmo3ThinkParser()
    let input = "<think>I should call get_weather.</think><function_calls>get_weather(city='Paris')</function_calls>"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text == "I should call get_weather.")
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    #expect(f.name == "get_weather")
  }

  @Test
  func `Char-by-char reconstruction matches one-shot for reasoning + tool call`() {
    let input = "<think>plan</think><function_calls>get_weather(city='Paris')</function_calls>"

    var oneShot = makeOlmo3ThinkParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = makeOlmo3ThinkParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    #expect(oneShotItems.count == 2)
    guard case let .reasoning(r1) = oneShotItems[0],
          case let .reasoning(r2) = streamedItems[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(r1.text == r2.text)
    guard case let .functionCall(f1) = oneShotItems[1],
          case let .functionCall(f2) = streamedItems[1]
    else {
      Issue.record("Expected function call"); return
    }
    #expect(f1.name == f2.name)
    #expect(f1.arguments == f2.arguments)
  }

  @Test
  func `InitialState .normal skips reasoning extraction (continuation path)`() {
    // Continuation request: prior output already contained `</think>`,
    // so the new parser starts in normal phase and treats `<think>` as
    // ordinary content. (We expect message text in that case.)
    var parser = makeOlmo3ThinkParser(initialState: .normal)
    let input = "Now back to work."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Now back to work.")
  }

  @Test
  func `Default off: acceptThink not set means <think> leaks as content`() {
    // Verifies the V3-family default-off semantics. Without `acceptThink`,
    // `<think>` markers are not extracted.
    var parser = makeOlmo3Parser()
    let input = "<think>plan</think>response"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == input)
  }
}

@Suite("Olmo3 — format dispatch")
struct Olmo3DispatchTests {
  @Test
  func `Name-prefix infer routes to .olmo3`() {
    let f = ResponseFormat.infer(
      modelName: "allenai/Olmo-3-7B-Instruct",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .olmo3)
  }

  @Test
  func `Think name-prefix infer routes to .olmo3Thinking`() {
    let f = ResponseFormat.infer(
      modelName: "allenai/Olmo-3-32B-Think",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .olmo3Thinking)
  }

  @Test
  func `model_type olmo3 routes to .olmo3`() {
    let f = ResponseFormat.resolveByType("olmo3", config: [:])
    #expect(f == .olmo3)
  }

  @Test
  func `Factory olmo3 keeps reasoning markers as message content`() {
    var parser = ResponseFormat.olmo3.makeParser(tokenizer: StubTokenizer())
    let input = "<think>plan</think>response"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == input)
  }

  @Test
  func `Factory olmo3Thinking extracts implicit reasoning preamble`() {
    var parser = ResponseFormat.olmo3Thinking.makeParser(tokenizer: StubTokenizer())
    let events =
      parser.process(ParserInput(text: "<think>plan</think>response")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else {
      Issue.record("Expected reasoning"); return
    }
    #expect(r.text == "plan")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "response")
  }

  @Test
  func `Factory olmo3Thinking resumes normal after prior reasoning close`() {
    var parser = ResponseFormat.olmo3Thinking.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>plan</think>",
    )
    let events = parser.process(ParserInput(text: "response")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "response")
  }
}
