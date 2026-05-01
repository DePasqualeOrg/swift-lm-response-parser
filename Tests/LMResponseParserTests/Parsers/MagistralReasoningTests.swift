// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

// Magistral mirrors vLLM's `MistralReasoningParser`: `[THINK]...[/THINK]`
// reasoning preamble layered on top of the standard Mistral tool-call
// shape. Tests focus on the reasoning extraction; the existing
// `MistralParserTests` cover the tool-call body.

@Suite("MistralParser — Magistral reasoning preamble")
struct MagistralReasoningTests {
  @Test
  func `Reasoning preamble emits a reasoning item`() {
    var parser = MistralParser(acceptThink: true)
    let input = "[THINK]Working it out.[/THINK]Final answer."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(reasoning.first == "Working it out.")
    #expect(messages.first == "Final answer.")
  }

  @Test
  func `Content before reasoning marker is preserved around Magistral reasoning`() {
    // Mirrors vLLM's `new_line` reasoning fixture: content may precede
    // `[THINK]`, with more content after `[/THINK]`.
    var parser = MistralParser(acceptThink: true)
    let input = "Before\n[THINK]This is a reasoning section[/THINK]\nThis is the rest"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)

    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messages == ["Before\n", "\nThis is the rest"])
    #expect(reasoning == ["This is a reasoning section"])
  }

  @Test
  func `Streaming content before reasoning marker is preserved`() {
    var parser = MistralParser(acceptThink: true)
    var events: [ResponseStreamingEvent] = []
    for chunk in ["Before\n", "[TH", "INK]R", "[/THINK]", "After"] {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let items = accumulateItems(from: events)

    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messages == ["Before\n", "After"])
    #expect(reasoning == ["R"])
  }

  @Test
  func `Reasoning preamble followed by a tool call`() {
    var parser = MistralParser(acceptThink: true)
    let input = #"[THINK]Need to call the tool.[/THINK][TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(reasoning.first == "Need to call the tool.")
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }

  @Test
  func `Stray [/THINK] without [THINK] is stripped from emitted content`() {
    // vLLM `MistralReasoningParser.extract_reasoning` Case 3: when
    // `[/THINK]` appears without a preceding `[THINK]`, the marker is
    // stripped and surrounding text is concatenated as content.
    var parser = MistralParser(acceptThink: true)
    let input = "Hello world [/THINK] more text"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(reasoning.isEmpty, "No [THINK] opener means no reasoning item")
    #expect(messages.first == "Hello world  more text", "[/THINK] marker must be stripped")
  }

  @Test
  func `acceptThink off — [THINK] is treated as plain content`() {
    // Default Mistral parser doesn't extract reasoning; the markers
    // flow through as plain text.
    var parser = MistralParser()
    let input = "[THINK]ignored[/THINK]rest"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> Bool? in
      if case .reasoning = item { return true } else { return nil }
    }
    #expect(reasoning.isEmpty, "Base Mistral must not extract reasoning")
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] { return t.text } else { return nil }
    }
    #expect(messages.first?.contains("[THINK]") == true)
  }

  @Test
  func `Truncated reasoning at EOS surfaces as incomplete`() {
    var parser = MistralParser(acceptThink: true)
    let input = "[THINK]still thinking when truncated"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text.contains("still thinking when truncated"))
    #expect(r.status == .incomplete)
  }

  @Test
  func `Char-by-char streaming preserves reasoning vs content split`() {
    let input = "[THINK]A B C[/THINK]Done."

    var oneShot = MistralParser(acceptThink: true)
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = MistralParser(acceptThink: true)
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    let oneShotReasoning = oneShotItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    let streamedReasoning = streamedItems.compactMap { item -> String? in
      if case let .reasoning(r) = item { return r.text } else { return nil }
    }
    #expect(oneShotReasoning == streamedReasoning)
  }

  @Test
  func `No [THINK] preamble — falls through to content immediately`() {
    var parser = MistralParser(acceptThink: true)
    let input = "Hello, world!"
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "Hello, world!")
  }

  @Test
  func `Continuation: priorOutput inside [THINK] starts in reasoning phase`() {
    let prior = "[THINK]I started"
    var parser = ResponseFormat.magistral.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: prior,
    )
    let events = parser.process(ParserInput(text: " and continued.[/THINK]Done."))
      + parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.text.contains("and continued."))
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else { Issue.record("Expected message"); return }
    #expect(t.text == "Done.")
  }
}

@Suite("ResponseFormat dispatch — Magistral")
struct MagistralDispatchTests {
  @Test
  func `Magistral-Small-2506 routes to .magistral by name`() {
    let f = ResponseFormat.infer(
      modelName: "mistralai/Magistral-Small-2506",
      modelType: "mistral",
      modelConfig: [:],
    )
    #expect(f == .magistral)
  }

  @Test
  func `Magistral-Small-1.1 routes to .magistral by name`() {
    let f = ResponseFormat.infer(
      modelName: "mistralai/Magistral-Small-1.1",
      modelType: "mistral",
      modelConfig: [:],
    )
    #expect(f == .magistral)
  }

  @Test
  func `Base Mistral routes to .mistral, not .magistral`() {
    let f = ResponseFormat.infer(
      modelName: "mistralai/Mistral-7B-Instruct-v0.3",
      modelType: "mistral",
      modelConfig: [:],
    )
    #expect(f == .mistral)
  }
}
