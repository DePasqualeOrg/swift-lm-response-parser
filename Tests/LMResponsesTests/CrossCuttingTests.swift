// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

// MARK: Token-ID propagation

@Suite("Token-ID propagation")
struct TokenIDPropagationTests {
  @Test
  func `ResponseStreamEmitter forwards both text and pending tokenIds to the parser`() {
    let captured = TokenCapture()
    let parser = ScriptedParser(
      onProcess: { input in
        captured.received.append(input)
        return []
      },
    )
    let emitter = ResponseStreamEmitter(
      parser: parser,
      config: ResponseStreamConfig(model: "m", createdAt: 0),
    )
    _ = emitter.start()
    _ = emitter.process(text: "hello", tokenIds: [101, 102, 103])
    _ = emitter.finalize(info: FinishInfo(finishReason: .stop, inputTokens: 1, outputTokens: 3))

    #expect(captured.received.count == 1)
    #expect(captured.received[0].text == "hello")
    #expect(captured.received[0].tokenIds == [101, 102, 103])
  }

  /// Use a class to capture parser inputs from inside a `Sendable` closure.
  final class TokenCapture: @unchecked Sendable {
    var received: [ParserInput] = []
  }
}

// MARK: Per-item-type event ordering

@Suite("Event ordering — message item")
struct MessageEventOrderingTests {
  @Test
  func `Plain message: added → content_part.added → text.delta+ → text.done → content_part.done → item.done`() {
    var parser = HermesParser()
    let events = parser.process(ParserInput(text: "hello world")) + parser.finalize()
    let kinds = events.map { eventKind($0) }
    #expect(kinds == [
      "outputItemAdded",
      "contentPartAdded",
      "outputTextDelta",
      "outputTextDone",
      "contentPartDone",
      "outputItemDone",
    ])
  }
}

@Suite("Event ordering — reasoning item")
struct ReasoningEventOrderingTests {
  @Test
  func `Reasoning: added → content_part.added (reasoning_text part) → reasoning delta+ → reasoning done → content_part.done → item.done`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: "<think>thinking...</think>")) + parser.finalize()
    let kinds = events.map { eventKind($0) }
    #expect(kinds == [
      "outputItemAdded",
      "contentPartAdded",
      "reasoningDelta",
      "reasoningDone",
      "contentPartDone",
      "outputItemDone",
    ])
  }

  @Test
  func `Reasoning content part is reasoning_text (not output_text or refusal)`() {
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: "<think>x</think>")) + parser.finalize()
    let added = events.compactMap { event -> ResponseContentPartAddedEvent? in
      if case let .contentPartAdded(e) = event { return e }
      return nil
    }
    #expect(added.count == 1)
    guard case .reasoningText = added[0].part else {
      Issue.record("Expected reasoning_text part"); return
    }
  }
}

@Suite("Event ordering — function call item")
struct FunctionCallEventOrderingTests {
  @Test
  func `Function call: added → arguments.delta+ → arguments.done → item.done (no content_part envelope)`() {
    var parser = HermesParser()
    let events = parser.process(ParserInput(
      text: #"<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#,
    )) + parser.finalize()
    let kinds = events.map { eventKind($0) }
    // No content_part envelope: function_call_arguments are on the
    // item itself, not in a content part.
    #expect(!kinds.contains("contentPartAdded"))
    #expect(!kinds.contains("contentPartDone"))
    #expect(kinds.first == "outputItemAdded")
    #expect(kinds.last == "outputItemDone")
  }
}

// MARK: accumulateItems

@Suite("accumulateItems")
struct AccumulateItemsTests {
  @Test
  func `Empty stream yields no items`() {
    let items = accumulateItems(from: [])
    #expect(items.isEmpty)
  }

  @Test
  func `Output items accumulate in order`() {
    var hermes = HermesParser()
    let events = hermes.process(ParserInput(
      text: #"hello<tool_call>{"name": "fn", "arguments": {}}</tool_call>world"#,
    )) + hermes.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 3)
    guard case let .message(m1) = items[0] else { Issue.record("Expected message"); return }
    guard case let .functionCall(f) = items[1] else { Issue.record("Expected function call"); return }
    guard case let .message(m2) = items[2] else { Issue.record("Expected message"); return }
    if case let .outputText(t) = m1.content[0] {
      #expect(t.text == "hello")
    }
    #expect(f.name == "fn")
    if case let .outputText(t) = m2.content[0] {
      #expect(t.text == "world")
    }
  }

  @Test
  func `output_item.done overrides slot with the canonical item`() {
    let id = "msg_test"
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(
        item: .message(.init(id: id, status: .inProgress)),
        outputIndex: 0,
        sequenceNumber: 0,
      )),
      .contentPartAdded(.init(
        itemId: id, outputIndex: 0, contentIndex: 0,
        part: .outputText(.init(text: "")),
        sequenceNumber: 1,
      )),
      .outputTextDelta(.init(
        itemId: id, outputIndex: 0, contentIndex: 0,
        delta: "hi", sequenceNumber: 2,
      )),
      .outputItemDone(.init(
        item: .message(.init(id: id, content: [.outputText(.init(text: "hi"))], status: .completed)),
        outputIndex: 0,
        sequenceNumber: 3,
      )),
    ]
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    #expect(m.status == .completed)
  }

  @Test
  func `Gap-indexed output_item events are ignored instead of fabricating placeholder items`() {
    let item = ResponseOutputItem.message(.init(id: "msg_gap", status: .completed))
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(item: item, outputIndex: 2, sequenceNumber: 0)),
      .outputItemDone(.init(item: item, outputIndex: 2, sequenceNumber: 1)),
    ]

    #expect(accumulateItems(from: events).isEmpty)
  }
}

@Suite("ResponseItemsAccumulator")
struct ResponseItemsAccumulatorTests {
  @Test
  func `Empty accumulator yields no items`() {
    let accumulator = ResponseItemsAccumulator()
    #expect(accumulator.items.isEmpty)
  }

  // The whole point of the accumulator: ingesting events one chunk at a time
  // produces the same item array as a single one-shot accumulateItems call
  // over the concatenated stream. This is the equivalence the perf
  // optimization rests on.
  @Test
  func `Incremental ingest of split chunks matches one-shot accumulation`() {
    var hermes = HermesParser()
    let events = hermes.process(ParserInput(
      text: #"hello<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>world"#,
    )) + hermes.finalize()

    let oneShot = accumulateItems(from: events)

    // Split events into three arbitrary chunks to exercise mid-event ingest seams.
    let third = max(1, events.count / 3)
    let chunkA = Array(events.prefix(third))
    let chunkB = Array(events.dropFirst(third).prefix(third))
    let chunkC = Array(events.dropFirst(third * 2))

    var accumulator = ResponseItemsAccumulator()
    accumulator.ingest(chunkA)
    accumulator.ingest(chunkB)
    accumulator.ingest(chunkC)

    #expect(accumulator.items == oneShot)
  }

  @Test
  func `Per-event ingest matches one-shot accumulation`() {
    var hermes = HermesParser()
    let events = hermes.process(ParserInput(
      text: #"plain text<tool_call>{"name": "fn", "arguments": {}}</tool_call>more text"#,
    )) + hermes.finalize()

    let oneShot = accumulateItems(from: events)

    var accumulator = ResponseItemsAccumulator()
    for event in events {
      accumulator.ingest(event)
    }

    #expect(accumulator.items == oneShot)
  }

  // Pin the chunk-boundary case the parser-generated tests above sample
  // only by accident: a final delta arrives in chunk N, and its matching
  // `output_item.done` arrives in chunk N+1. The mid-stream snapshot must
  // expose the in-progress status; after the done event lands, status
  // flips to completed. Accumulator state must persist across the seam.
  @Test
  func `Chunk boundary between final delta and outputItemDone preserves status transition`() {
    let id = "msg_x"
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(
        item: .message(.init(id: id, status: .inProgress)),
        outputIndex: 0,
        sequenceNumber: 0,
      )),
      .contentPartAdded(.init(
        itemId: id, outputIndex: 0, contentIndex: 0,
        part: .outputText(.init(text: "")),
        sequenceNumber: 1,
      )),
      .outputTextDelta(.init(
        itemId: id, outputIndex: 0, contentIndex: 0,
        delta: "hi", sequenceNumber: 2,
      )),
      .outputItemDone(.init(
        item: .message(.init(id: id, content: [.outputText(.init(text: "hi"))], status: .completed)),
        outputIndex: 0,
        sequenceNumber: 3,
      )),
    ]

    var accumulator = ResponseItemsAccumulator()
    accumulator.ingest(Array(events.prefix(3)))
    guard case let .message(mid) = accumulator.items[0] else {
      Issue.record("Expected message after first chunk"); return
    }
    #expect(mid.status == .inProgress)
    if case let .outputText(t) = mid.content[0] {
      #expect(t.text == "hi")
    }

    accumulator.ingest(Array(events.suffix(1)))
    #expect(accumulator.items == accumulateItems(from: events))
    guard case let .message(final) = accumulator.items[0] else {
      Issue.record("Expected message after second chunk"); return
    }
    #expect(final.status == .completed)
  }

  // Multi-item interleaving: reasoning + message + function call all open
  // and produce deltas in the same response. Per-chunk ingest must
  // match one-shot regardless of where chunk boundaries fall.
  @Test
  func `Multi-item interleaved stream matches one-shot under chunked ingest`() {
    var qwen = QwenParser()
    // Produces three items: reasoning, message ("before"), function call.
    let events = qwen.process(ParserInput(text: "<think>r</think>before<tool_"))
      + qwen.process(ParserInput(text: #"call>{"name": "fn", "arguments": {}}</tool_call>"#))
      + qwen.finalize()

    let oneShot = accumulateItems(from: events)
    #expect(oneShot.count == 3, "fixture should yield reasoning + message + function call")

    // Slide a chunk boundary across every event index and assert
    // equivalence at each split. Catches state-leak bugs that depend
    // on which event happens to land at the boundary.
    for split in 0 ... events.count {
      var accumulator = ResponseItemsAccumulator()
      accumulator.ingest(Array(events.prefix(split)))
      accumulator.ingest(Array(events.dropFirst(split)))
      #expect(accumulator.items == oneShot, "split at index \(split) diverged from one-shot")
    }
  }
}

// MARK: Continuation tests for all reasoning-capable parsers

@Suite("DelimitedReasoningBoundary")
struct DelimitedReasoningBoundaryTests {
  @Test
  func `Open delimiter returns suffix from the last opener`() {
    let boundary = DelimitedReasoningBoundary.think()
    let suffix = boundary.suffixIfOpen(in: "old <think>done</think> new <think>partial")
    #expect(suffix == "<think>partial")
  }

  @Test
  func `Explicit end marker closes the boundary`() {
    let boundary = DelimitedReasoningBoundary.think()
    #expect(!boundary.isOpen(in: "<think>done</think>"))
  }

  @Test
  func `Implicit end marker closes the boundary`() {
    let boundary = DelimitedReasoningBoundary.think(implicitEndTokens: ["<tool_call>"])
    #expect(!boundary.isOpen(in: "<think>partial<tool_call>"))
  }

  @Test
  func `Unpaired implicit end marker closes the boundary`() {
    let boundary = DelimitedReasoningBoundary.think(unpairedImplicitEnds: [.toolCall])
    #expect(!boundary.isOpen(in: "<think>partial<tool_call>"))
  }

  @Test
  func `Paired implicit end marker keeps the boundary open`() {
    let boundary = DelimitedReasoningBoundary.think(unpairedImplicitEnds: [.toolCall])
    let suffix = boundary.suffixIfOpen(in: "<think>example<tool_call></tool_call>")
    #expect(suffix == "<think>example<tool_call></tool_call>")
  }
}

@Suite("ImplicitReasoningPreamble")
struct ImplicitReasoningPreambleTests {
  @Test
  func `Starts in reasoning when no prior end marker exists`() {
    let preamble = ImplicitReasoningPreamble.think()
    #expect(preamble.startsInReasoning(after: nil))
    #expect(preamble.startsInReasoning(after: "partial reasoning"))
  }

  @Test
  func `Explicit end marker resumes normal`() {
    let preamble = ImplicitReasoningPreamble.think()
    #expect(!preamble.startsInReasoning(after: "reasoning</think>answer"))
  }

  @Test
  func `Implicit end marker also resumes normal`() {
    let preamble = ImplicitReasoningPreamble.think(implicitEndTokens: ["<tool_call>"])
    #expect(!preamble.startsInReasoning(after: "reasoning<tool_call>"))
  }
}

@Suite("Prompt-boundary parser state")
struct PromptBoundaryParserStateTests {
  @Test
  func `Qwen XML starts in reasoning when rendered prompt leaves think open`() {
    let prior = ResponseFormat.qwen3Xml.promptBoundaryPriorText(
      fromRenderedPrompt: "system\nuser\nassistant\n<think>\n",
    )
    #expect(prior == "<think>")
  }

  @Test
  func `Qwen XML stays normal when rendered prompt already closed think`() {
    let prior = ResponseFormat.qwen3Xml.promptBoundaryPriorText(
      fromRenderedPrompt: "system\nuser\nassistant\n<think>\n\n</think>\n\n",
    )
    #expect(prior == nil)
  }

  @Test
  func `Qwen XML treats unpaired prompt tool call as implicit reasoning end`() {
    let prior = ResponseFormat.qwen3Xml.promptBoundaryPriorText(
      fromRenderedPrompt: "<think>old reasoning<tool_call>",
    )
    #expect(prior == nil)
  }

  @Test
  func `Qwen XML ignores paired prompt tool call examples when deciding prompt boundary`() {
    let prior = ResponseFormat.qwen3Xml.promptBoundaryPriorText(
      fromRenderedPrompt: "<think>example<tool_call></tool_call>",
    )
    #expect(prior == "<think>")
  }

  @Test
  func `Paired prompt tool call examples keep generated suffix in reasoning`() {
    let prior = ResponseFormat.qwen3Xml.combinedPriorOutput(
      fromRenderedPrompt: "<think>example<tool_call></tool_call>",
      generatedPriorOutput: nil,
    )
    var parser = ResponseFormat.qwen3Xml.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: prior,
    )

    let events = parser.process(ParserInput(text: "new reasoning</think>answer"))
      + parser.finalize()
    let items = accumulateItems(from: events)

    guard items.count == 2 else {
      Issue.record("Expected reasoning + message, got \(items)")
      return
    }
    guard case let .reasoning(reasoning) = items[0] else {
      Issue.record("Expected first item to be reasoning, got \(items[0])")
      return
    }
    #expect(reasoning.text == "new reasoning")
    guard case let .message(message) = items[1] else {
      Issue.record("Expected second item to be message, got \(items[1])")
      return
    }
    guard case let .outputText(text) = message.content.first else {
      Issue.record("Expected message output text, got \(message.content)")
      return
    }
    #expect(text.text == "answer")
  }

  @Test
  func `Non Qwen formats ignore prompt think markers`() {
    let prior = ResponseFormat.hermes.promptBoundaryPriorText(
      fromRenderedPrompt: "assistant\n<think>\n",
    )
    #expect(prior == nil)
  }

  @Test
  func `Generated prior output is appended after prompt boundary context`() {
    let prior = ResponseFormat.qwen3Xml.combinedPriorOutput(
      fromRenderedPrompt: "system\nuser\nassistant\n<think>\n",
      generatedPriorOutput: "partial reasoning",
    )
    #expect(prior == "<think>partial reasoning")
  }

  @Test
  func `Generated prior output passes through when prompt has no relevant boundary`() {
    let prior = ResponseFormat.hermes.combinedPriorOutput(
      fromRenderedPrompt: "assistant\n<think>\n",
      generatedPriorOutput: "partial generated text",
    )
    #expect(prior == "partial generated text")
  }

  @Test
  func `Prompt injected think makes Qwen3 XML generated suffix parse as reasoning then tool call`() throws {
    let prior = ResponseFormat.qwen3Xml.promptBoundaryPriorText(
      fromRenderedPrompt: "system\nuser\nassistant\n<think>\n",
    )
    var parser = ResponseFormat.qwen3Xml.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: prior,
    )

    let output = """
    I should call the weather tool.</think><tool_call>
    <function=get_weather>
    <parameter=city>Paris</parameter>
    </function>
    </tool_call>
    """
    let events = parser.process(ParserInput(text: output)) + parser.finalize()
    let items = accumulateItems(from: events)

    guard items.count == 2 else {
      Issue.record("Expected reasoning + function call, got \(items)")
      return
    }
    guard case let .reasoning(reasoning) = items[0] else {
      Issue.record("Expected first item to be reasoning, got \(items[0])")
      return
    }
    #expect(reasoning.text == "I should call the weather tool.")

    guard case let .functionCall(call) = items[1] else {
      Issue.record("Expected second item to be function call, got \(items[1])")
      return
    }
    #expect(call.name == "get_weather")
    #expect(call.status == .completed)
    let decoded = try call.decodedArguments(as: WeatherArgs.self)
    #expect(decoded == WeatherArgs(city: "Paris"))
  }

  private struct WeatherArgs: Codable, Equatable {
    var city: String
  }
}

@Suite("Continuation — priorOutput sets initial reasoning state")
struct ContinuationTests {
  @Test
  func `Qwen: priorOutput with unclosed <think> resumes in reasoning`() {
    let parser = ResponseFormat.qwen.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial",
    )
    var p = parser
    let items = accumulateItems(from:
      p.process(ParserInput(text: " continues</think>final")) + p.finalize())
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
  }

  @Test
  func `Qwen: priorOutput with tool call after think starts normal`() {
    let parser = ResponseFormat.qwen.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial<tool_call></tool_call>",
    )
    var p = parser
    let items = accumulateItems(from: p.process(ParserInput(text: "final")) + p.finalize())
    guard items.count == 1 else {
      Issue.record("Expected one message, got \(items)")
      return
    }
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Qwen3-Xml: priorOutput with unclosed <think> resumes in reasoning`() {
    let parser = ResponseFormat.qwen3Xml.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial",
    )
    var p = parser
    let items = accumulateItems(from:
      p.process(ParserInput(text: " continues</think>final")) + p.finalize())
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
  }

  @Test
  func `Qwen3-Xml: priorOutput with tool call after think starts normal`() {
    let parser = ResponseFormat.qwen3Xml.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial<tool_call></tool_call>",
    )
    var p = parser
    let items = accumulateItems(from: p.process(ParserInput(text: "final")) + p.finalize())
    guard items.count == 1 else {
      Issue.record("Expected one message, got \(items)")
      return
    }
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `DeepSeek-R1: priorOutput with unclosed <think> resumes in reasoning`() {
    let parser = ResponseFormat.deepseekR1.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "<think>partial",
    )
    var p = parser
    let items = accumulateItems(from:
      p.process(ParserInput(text: " continues</think>final")) + p.finalize())
    #expect(items.count == 2)
    guard case .reasoning = items[0] else { Issue.record("Expected reasoning"); return }
  }

  @Test
  func `No priorOutput starts in normal state`() {
    let parser = ResponseFormat.qwen.makeParser(tokenizer: StubTokenizer())
    var p = parser
    let items = accumulateItems(from:
      p.process(ParserInput(text: "plain")) + p.finalize())
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Qwen: closed prior <think> + fresh <think> opener re-enters reasoning (SGLang divergence)`() {
    // Adversarial fixture from
    // sglang/test/registered/unit/parser/test_reasoning_parser.py
    // (`test_continue_end_in_previous_new_text_has_start_but_no_end`).
    //
    // Prior had a closed `<think>...</think>` block and trailing content.
    // New tokens emit a fresh `<think>` opener with no `</think>` yet.
    //
    // SGLang treats the new tokens as `normal_text` ("continuing
    // reasoning") and `reasoning_text == ""`. We deliberately treat the
    // new `<think>` as a fresh reasoning block — `QwenParser.scanNormal`
    // transitions back to reasoning when it sees `<think>` before any
    // content or tool call has been opened. Our position: an unclosed
    // `<think>` opener is most naturally interpreted as a real reasoning
    // block in progress, regardless of what the prior response contained.
    let parser = ResponseFormat.qwen.makeParser(
      tokenizer: StubTokenizer(),
      priorOutput: "earlier <think>old</think>old answer",
    )
    var p = parser
    let items = accumulateItems(from:
      p.process(ParserInput(text: "<think>continuing reasoning")) + p.finalize())
    #expect(items.count == 1)
    guard case let .reasoning(r) = items[0] else { Issue.record("Expected reasoning"); return }
    #expect(r.status == .incomplete)
    guard case let .reasoningText(part) = r.content[0] else { Issue.record("Expected reasoningText"); return }
    #expect(part.text == "continuing reasoning")
  }
}

// MARK: One-shot vs streaming agreement

@Suite("One-shot vs streaming — all parsers")
struct OneShotVsStreamingTests {
  @Test
  func `Hermes: one-shot and char-by-char streaming yield matching items`() {
    let input = #"hello<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>"#
    compareOneShotVsStreaming(format: .hermes, input: input)
  }

  @Test
  func `Qwen: one-shot and streaming agree (with reasoning)`() {
    let input = #"<think>r</think>before<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#
    compareOneShotVsStreaming(format: .qwen, input: input)
  }

  @Test
  func `Mistral: one-shot and streaming agree (compact)`() {
    let input = #"[TOOL_CALLS]fn[ARGS]{"x": 1}"#
    compareOneShotVsStreaming(format: .mistral, input: input)
  }

  @Test
  func `Llama 3: one-shot and streaming agree`() {
    let input = #"<|python_tag|>{"name": "fn", "arguments": {"x": 1}}"#
    compareOneShotVsStreaming(format: .llama3, input: input)
  }

  @Test
  func `JSON fallback: one-shot and streaming agree`() {
    let input = #"{"name": "fn", "arguments": {"x": 1}}"#
    compareOneShotVsStreaming(format: .json, input: input)
  }

  private func compareOneShotVsStreaming(format: ResponseFormat, input: String) {
    var oneShot = format.makeParser(tokenizer: StubTokenizer())
    let oneShotItems = accumulateItems(from:
      oneShot.process(ParserInput(text: input)) + oneShot.finalize())

    var streaming = format.makeParser(tokenizer: StubTokenizer())
    var streamingEvents: [ResponseStreamingEvent] = []
    for ch in input {
      streamingEvents += streaming.process(ParserInput(text: String(ch)))
    }
    streamingEvents += streaming.finalize()
    let streamingItems = accumulateItems(from: streamingEvents)

    #expect(oneShotItems.count == streamingItems.count, "Item count differs for \(format)")
    for (s, o) in zip(streamingItems, oneShotItems) {
      switch (s, o) {
        case let (.message(sm), .message(om)):
          #expect(sm.content == om.content)
        case let (.functionCall(sf), .functionCall(of)):
          #expect(sf.name == of.name)
          #expect(sf.arguments == of.arguments)
        case let (.reasoning(sr), .reasoning(or)):
          #expect(sr.content == or.content)
        default:
          Issue.record("Item kinds differ for \(format)"); return
      }
    }
  }
}

// MARK: Reasoning adversarial ports

@Suite("Reasoning — adversarial ports")
struct ReasoningAdversarialTests {
  // H12: SGLang test_partial_end_tag_buffer_loss_bug
  // (test_reasoning_parser.py:1097-1127). Regression for a real bug.
  @Test
  func `Partial </ then non-matching answer flushes as </answer (no buffer loss)`() {
    let chunks = ["</", "answer"]
    var parser = QwenParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let deltas = events.compactMap { ev -> String? in
      if case let .outputTextDelta(e) = ev { return e.delta } else { return nil }
    }
    // Total accumulated content is the original input — no fragment lost.
    #expect(deltas.joined() == "</answer")
  }

  // M4: SGLang test_multiple_partial_fragments
  // (test_reasoning_parser.py:1166-1183).
  @Test
  func `Multiple sub-character fragments <, /, random> accumulate to </random>`() {
    let chunks = ["<", "/", "random>"]
    var parser = QwenParser()
    var events: [ResponseStreamingEvent] = []
    for chunk in chunks {
      events += parser.process(ParserInput(text: chunk))
    }
    events += parser.finalize()
    let deltas = events.compactMap { ev -> String? in
      if case let .outputTextDelta(e) = ev { return e.delta } else { return nil }
    }
    // SGLang holds the partial fragments and emits the whole string at
    // chunk 3. Our parser emits more eagerly (chunks 2 and 3 combined),
    // but the total accumulated content is identical.
    #expect(deltas.joined() == "</random>")
  }

  // M5: SGLang test_empty_reasoning_blocks
  // (test_reasoning_parser.py:1050-1057). Verify across each
  // <think>-based parser.
  @Test
  func `Empty <think></think> followed by content yields no reasoning item (Qwen)`() {
    let input = "<think></think>Just the answer."
    var parser = QwenParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .message(m) = items[0] else { Issue.record("Expected message"); return }
    guard case let .outputText(part) = m.content[0] else { Issue.record("Expected outputText"); return }
    #expect(part.text == "Just the answer.")
  }

  @Test
  func `Empty <think></think> followed by content yields no reasoning item (Qwen3-Xml)`() {
    let input = "<think></think>Just the answer."
    var parser = Qwen3XmlParser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }

  @Test
  func `Empty <think></think> followed by content yields no reasoning item (DeepSeek-R1)`() {
    let input = "<think></think>Just the answer."
    var parser = DeepSeekR1Parser()
    let events = parser.process(ParserInput(text: input)) + parser.finalize()
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case .message = items[0] else { Issue.record("Expected message"); return }
  }
}
