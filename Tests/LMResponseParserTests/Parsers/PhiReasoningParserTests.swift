// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("PhiReasoningParser — basics")
struct PhiReasoningBasicTests {
  @Test
  func `Plain text without <think> emits a single message`() {
    var parser = PhiReasoningParser()
    let events = parser.process(ParserInput(text: "The answer is 42.")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "The answer is 42.")
  }

  @Test
  func `Empty stream emits nothing`() {
    var parser = PhiReasoningParser()
    #expect(parser.finalize().isEmpty)
  }

  @Test
  func `Reasoning then answer`() {
    var parser = PhiReasoningParser()
    let input = "<think>Let me work through this.</think>The answer is 42."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    guard case let .reasoningText(rt) = r.content[0] else { Issue.record("Expected reasoning text"); return }
    #expect(rt.text == "Let me work through this.")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "The answer is 42.")
  }

  @Test
  func `Empty <think></think> emits no reasoning item`() {
    var parser = PhiReasoningParser()
    let events = parser.process(ParserInput(text: "<think></think>The answer.")) + parser.finalize()
    let items = accumulateItems(from: events)
    // Empty reasoning should not produce a reasoning item — only the
    // trailing message. Matches our SGLang-aligned behavior for other
    // think-tag parsers.
    let reasonings = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasonings.isEmpty)
    let messages = items.compactMap { item -> String? in
      if case let .message(m) = item, case let .outputText(t) = m.content[0] {
        return t.text
      }
      return nil
    }
    #expect(messages == ["The answer."])
  }

  @Test
  func `Tool-call literal in reasoning is treated as content, not a marker`() {
    // Phi reasoning models have no tool-call channel. Any
    // `<tool_call>` literal in the output is plain text.
    var parser = PhiReasoningParser()
    let input = "Use <tool_call>{\"name\": \"foo\"}</tool_call> if available."
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

@Suite("PhiReasoningParser — streaming")
struct PhiReasoningStreamingTests {
  @Test
  func `Char-by-char reconstruction matches one-shot`() {
    let input = "<think>Step 1: identify. Step 2: compute.</think>The answer is 42."

    var oneShot = PhiReasoningParser()
    let oneShotItems = accumulateItems(
      from: oneShot.process(ParserInput(text: input)) + oneShot.finalize(),
    )

    var streamed = PhiReasoningParser()
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    #expect(oneShotItems.count == streamedItems.count)
    #expect(oneShotItems.count == 2)
    guard case let .reasoning(oR) = oneShotItems[0],
          case let .reasoning(sR) = streamedItems[0],
          case let .reasoningText(oRT) = oR.content[0],
          case let .reasoningText(sRT) = sR.content[0]
    else {
      Issue.record("Expected reasoning blocks"); return
    }
    #expect(oRT.text == sRT.text)
  }

  @Test
  func `Split <think> across chunks doesn't leak partial marker`() {
    var parser = PhiReasoningParser()
    var events = parser.process(ParserInput(text: "<thi"))
    events += parser.process(ParserInput(text: "nk>thinking...</think>final."))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "thinking...")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "final.")
  }

  @Test
  func `Split </think> across chunks doesn't leak partial marker into reasoning`() {
    var parser = PhiReasoningParser()
    var events = parser.process(ParserInput(text: "<think>thinking</thi"))
    events += parser.process(ParserInput(text: "nk>final."))
    events += parser.finalize()
    let items = accumulateItems(from: events)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == "thinking")
  }
}

@Suite("PhiReasoningParser — finalize edge cases")
struct PhiReasoningFinalizeEdgeCases {
  @Test
  func `Truncated <think> without </think> closes reasoning as incomplete`() {
    var parser = PhiReasoningParser()
    let events = parser.process(ParserInput(text: "<think>partial reasoning")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .incomplete)
  }

  @Test
  func `Buffer holding partial-marker prefix at finalize emits as content`() {
    var parser = PhiReasoningParser()
    // `<thi` could grow into `<think>` but EOS arrives first. The
    // partial bytes surface as plain content, not reasoning.
    let events = parser.process(ParserInput(text: "<thi")) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "<thi")
  }
}

@Suite("PhiReasoningParser — continuation")
struct PhiReasoningContinuationTests {
  @Test
  func `Initial state .reasoning resumes mid-think without expecting an opener`() {
    var parser = PhiReasoningParser(initialState: .reasoning)
    // Continuation: prior request ended mid-`<think>`, this stream
    // continues with reasoning content and the closing `</think>`.
    let input = " continuing thought.</think>final answer."
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 2)
    guard case let .reasoning(r) = items[0],
          case let .reasoningText(rt) = r.content[0]
    else {
      Issue.record("Expected reasoning"); return
    }
    #expect(rt.text == " continuing thought.")
    guard case let .message(m) = items[1],
          case let .outputText(t) = m.content[0]
    else {
      Issue.record("Expected message"); return
    }
    #expect(t.text == "final answer.")
  }
}

@Suite("PhiReasoningParser — dispatch")
struct PhiReasoningDispatchTests {
  @Test
  func `Phi-4-reasoning routes to phiReasoning`() {
    let f = ResponseFormat.infer(
      modelName: "microsoft/Phi-4-reasoning",
      modelType: "phi3",
      modelConfig: [:],
    )
    #expect(f == .phiReasoning)
  }

  @Test
  func `Phi-4-reasoning-plus routes to phiReasoning (more specific prefix)`() {
    let f = ResponseFormat.infer(
      modelName: "microsoft/Phi-4-reasoning-plus",
      modelType: "phi3",
      modelConfig: [:],
    )
    #expect(f == .phiReasoning)
  }

  @Test
  func `Phi-4-mini-reasoning routes to phiReasoning`() {
    let f = ResponseFormat.infer(
      modelName: "microsoft/Phi-4-mini-reasoning",
      modelType: "phi3",
      modelConfig: [:],
    )
    #expect(f == .phiReasoning)
  }
}
