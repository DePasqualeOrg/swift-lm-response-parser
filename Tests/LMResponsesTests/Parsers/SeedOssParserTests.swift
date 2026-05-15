// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponses
import Testing

/// Seed-OSS reuses the Qwen 3 Coder XML parser, parameterized on its
/// `<seed:tool_call>` envelope and `<seed:think>` reasoning markers.
/// Coverage focuses on:
/// 1. The renamed envelope tokens are recognized end-to-end via the
///    factory (parity with the wire format vLLM's `SeedOssToolParser`
///    consumes).
/// 2. The vanilla `<tool_call>` / `<think>` tokens are *not* recognized
///    when the parser is configured for Seed-OSS – stray Qwen-shape
///    markers in Seed-OSS output flow through as content.
/// 3. Reasoning is extracted in the R1-style: works whether or not the
///    `<seed:think>` opener is emitted by the model.

@Suite("SeedOssParser — envelope swap")
struct SeedOssEnvelopeTests {
  @Test
  func `Single tool call wrapped in <seed:tool_call> is extracted`() {
    let text = """
    <seed:tool_call>
    <function=get_weather>
    <parameter=location>Barcelona, Spain</parameter>
    </function>
    </seed:tool_call>
    """
    var parser = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
    #expect(toolCalls[0].arguments == #"{"location": "Barcelona, Spain"}"#)
  }

  @Test
  func `Plain Qwen <tool_call> is not recognized as the Seed-OSS envelope`() {
    // In Seed-OSS the parser starts in `.reasoning` mode (the chat
    // template injects `<seed:think>` so the model emits reasoning
    // immediately). In reasoning mode, only `</seed:think>` and
    // `<seed:tool_call>` are exit markers – the unmatched Qwen
    // `<tool_call>` text is therefore treated as reasoning content,
    // not envelope. No tool calls are extracted.
    let text = #"""
    <tool_call>
    <function=get_weather>
    <parameter=city>Tokyo</parameter>
    </function>
    </tool_call>
    """#
    var parser = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.isEmpty, "Qwen-style <tool_call> must not be recognized as a Seed-OSS envelope")
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text.contains("<tool_call>"))
    } else {
      Issue.record("Expected reasoning text content")
    }
  }

  @Test
  func `Char-by-char streaming through <seed:tool_call> matches single-shot`() {
    let text = """
    <seed:tool_call>
    <function=get_weather>
    <parameter=location>Barcelona, Spain</parameter>
    </function>
    </seed:tool_call>
    """
    var single = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    let singleEvents = single.process(ParserInput(text: text)) + single.finalize()
    let singleItems = accumulateItems(from: singleEvents)

    var streamed = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    var streamedEvents: [ResponseStreamingEvent] = []
    for ch in text {
      streamedEvents += streamed.process(ParserInput(text: String(ch)))
    }
    streamedEvents += streamed.finalize()
    let streamedItems = accumulateItems(from: streamedEvents)

    let singleCalls = singleItems.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    let streamedCalls = streamedItems.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(singleCalls.count == streamedCalls.count)
    #expect(singleCalls.count == 1)
    #expect(singleCalls[0].name == streamedCalls[0].name)
    #expect(singleCalls[0].arguments == streamedCalls[0].arguments)
  }

  @Test
  func `Two consecutive <seed:tool_call> envelopes both extract`() {
    let text = """
    <seed:tool_call>
    <function=f>
    <parameter=x>1</parameter>
    </function>
    </seed:tool_call>
    <seed:tool_call>
    <function=g>
    <parameter=y>2</parameter>
    </function>
    </seed:tool_call>
    """
    var parser = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 2)
    #expect(toolCalls[0].name == "f")
    #expect(toolCalls[1].name == "g")
  }
}

@Suite("SeedOssParser — reasoning")
struct SeedOssReasoningTests {
  @Test
  func `Reasoning between <seed:think> and </seed:think> becomes a reasoning item`() {
    let text = """
    <seed:think>
    Let me figure out the weather query.
    </seed:think>
    <seed:tool_call>
    <function=get_weather>
    <parameter=location>Barcelona, Spain</parameter>
    </function>
    </seed:tool_call>
    """
    var parser = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.count == 1)
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text.contains("figure out the weather"))
    } else {
      Issue.record("Expected reasoning text content")
    }
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
    #expect(toolCalls[0].name == "get_weather")
  }

  @Test
  func `Reasoning without an explicit <seed:think> opener is still extracted (R1-style)`() {
    // Mirrors the behavior of vLLM's SeedOSSReasoningParser, whose
    // docstring promises "Similar to DeepSeek R1, it supports cases
    // where the model doesn't generate the start token." The chat
    // template injects `<seed:think>` so the model output begins
    // mid-thought.
    let text = """
    Thinking about the user's query.
    </seed:think>
    <seed:tool_call>
    <function=f>
    </function>
    </seed:tool_call>
    """
    var parser = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.count == 1, "Reasoning should be extracted even without a <seed:think> opener")
    if case let .reasoningText(r) = reasoning[0].content[0] {
      #expect(r.text.contains("Thinking about the user"))
    } else {
      Issue.record("Expected reasoning text content")
    }
  }

  @Test
  func `<seed:tool_call> arriving before any </seed:think> exits reasoning empty-handed`() {
    // No explicit reasoning, no `<seed:think>` opener in the model
    // output – this is the "0 thinking budget" Seed-OSS test fixture
    // from vLLM. The chat-template-injected opener was empty so the
    // model went straight to the tool call. The R1-style reasoning
    // mode treats the `<seed:tool_call>` marker as an exit boundary,
    // so the empty-prefix case yields no reasoning item.
    let text = """
    <seed:tool_call>
    <function=f>
    </function>
    </seed:tool_call>
    """
    var parser = ResponseFormat.seedOss.makeParser(tokenizer: StubTokenizer())
    let events = parser.process(ParserInput(text: text)) + parser.finalize()
    let items = accumulateItems(from: events)
    let reasoning = items.compactMap { item -> ResponseReasoningItem? in
      if case let .reasoning(r) = item { return r } else { return nil }
    }
    #expect(reasoning.isEmpty, "No reasoning should be emitted when the model goes straight to a tool call")
    let toolCalls = items.compactMap { item -> ResponseFunctionToolCall? in
      if case let .functionCall(f) = item { return f } else { return nil }
    }
    #expect(toolCalls.count == 1)
  }
}

@Suite("ResponseFormat dispatch — Seed-OSS")
struct SeedOssDispatchTests {
  @Test
  func `Seed-OSS-36B-Instruct name routes to .seedOss`() {
    let f = ResponseFormat.infer(
      modelName: "ByteDance-Seed/Seed-OSS-36B-Instruct",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .seedOss)
  }

  @Test
  func `Bare seed-oss prefix also routes to .seedOss`() {
    let f = ResponseFormat.infer(
      modelName: "seed-oss-7b",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .seedOss)
  }

  @Test
  func `model_type seed_oss routes to .seedOss`() {
    let f = ResponseFormat.infer(
      modelName: "",
      modelType: "seed_oss",
      modelConfig: [:],
    )
    #expect(f == .seedOss)
  }
}
