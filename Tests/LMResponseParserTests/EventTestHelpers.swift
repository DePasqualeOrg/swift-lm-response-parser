// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

/// Run `input` through `makeParser()` two ways — one shot (whole string in
/// one `process` call) and char-by-char (one `process` call per Character)
/// — then assert the accumulated `[ResponseOutputItem]` match (modulo
/// parser-minted IDs, which are random per invocation).
///
/// Mirrors vLLM's `test_streaming_reconstruction` from
/// `tests/tool_parsers/common_tests.py`: the streaming and non-streaming
/// paths must agree on what the model said. Most state-machine
/// regressions (lost prefix-hold byte, off-by-one on a marker boundary,
/// dropped final delta in `finalize()`) show up here as a divergence the
/// per-format tests can miss because they only exercise one path at a
/// time.
///
/// `makeParser` is a closure so each invocation gets a fresh state — the
/// helper builds two parsers, one per path. `mutating func process` means
/// the parser must be a value type, which all the shipped parsers are.
func assertStreamingReconstruction(
  _ input: String,
  parser makeParser: () -> some ResponseFormatParser,
  sourceLocation: SourceLocation = #_sourceLocation,
) {
  var oneShotParser = makeParser()
  let oneShotEvents =
    oneShotParser.process(ParserInput(text: input)) + oneShotParser.finalize()
  let oneShotItems = accumulateItems(from: oneShotEvents)

  var streamedParser = makeParser()
  var streamedEvents: [ResponseStreamingEvent] = []
  for ch in input {
    streamedEvents += streamedParser.process(ParserInput(text: String(ch)))
  }
  streamedEvents += streamedParser.finalize()
  let streamedItems = accumulateItems(from: streamedEvents)

  let oneShotNormalized = normalizeItemIDs(oneShotItems)
  let streamedNormalized = normalizeItemIDs(streamedItems)

  #expect(
    oneShotNormalized == streamedNormalized,
    """
    Streaming and one-shot paths produced different items.
    one-shot: \(oneShotNormalized)
    streamed: \(streamedNormalized)
    """,
    sourceLocation: sourceLocation,
  )
}

/// Replace each item's parser-minted IDs with a stable sentinel so two
/// independent parser runs (which produce different random IDs every
/// time) can be compared structurally.
func normalizeItemIDs(_ items: [ResponseOutputItem]) -> [ResponseOutputItem] {
  items.map { item in
    switch item {
      case var .message(m):
        m.id = "msg_TEST"
        return .message(m)
      case var .functionCall(f):
        f.id = "fc_TEST"
        f.callId = "call_TEST"
        return .functionCall(f)
      case var .reasoning(r):
        r.id = "rs_TEST"
        return .reasoning(r)
      case var .functionCallOutput(o):
        o.id = "fco_TEST"
        o.callId = "call_TEST"
        return .functionCallOutput(o)
    }
  }
}

/// Stream `text` through `makeParser()` in fixed-width character chunks
/// of size `interval` and return the accumulated `[ResponseOutputItem]`.
///
/// Mirrors vLLM's `@pytest.mark.parametrize("stream_interval", [1, 2, 4, 8])`
/// pattern from `tests/tool_parsers/test_hermes_tool_parser.py`. State-
/// machine bugs that only fire at specific chunk sizes (notably bug
/// #19056 in vLLM, where boolean `true`/`false` args broke at
/// `stream_interval > 1`) are caught by running a parser-format-specific
/// canonical fixture across `[1, 2, 4, 8]` and asserting on the
/// reconstructed items.
func streamItems(
  text: String,
  interval: Int,
  parser makeParser: () -> some ResponseFormatParser,
) -> [ResponseOutputItem] {
  var parser = makeParser()
  var events: [ResponseStreamingEvent] = []
  let chars = Array(text)
  var i = 0
  while i < chars.count {
    let end = min(i + interval, chars.count)
    events += parser.process(ParserInput(text: String(chars[i ..< end])))
    i = end
  }
  events += parser.finalize()
  return accumulateItems(from: events)
}

/// Tag of a streaming event by case name. Used in tests that assert on the
/// exact emit order without depending on the event payloads.
func eventKind(_ event: ResponseStreamingEvent) -> String {
  switch event {
    case .responseCreated: "responseCreated"
    case .responseInProgress: "responseInProgress"
    case .responseCompleted: "responseCompleted"
    case .outputItemAdded: "outputItemAdded"
    case .outputItemDone: "outputItemDone"
    case .contentPartAdded: "contentPartAdded"
    case .contentPartDone: "contentPartDone"
    case .outputTextDelta: "outputTextDelta"
    case .outputTextDone: "outputTextDone"
    case .functionCallArgumentsDelta: "functionCallArgumentsDelta"
    case .functionCallArgumentsDone: "functionCallArgumentsDone"
    case .reasoningTextDelta: "reasoningTextDelta"
    case .reasoningTextDone: "reasoningTextDone"
  }
}

/// Trivial `ParserTokenizer` for tests that only need to construct a parser
/// (none of the parsers under test today exercise the tokenizer methods).
struct StubTokenizer: ParserTokenizer {
  func convertTokenToId(_: String) -> Int? {
    nil
  }

  func encode(text _: String, addSpecialTokens _: Bool) -> [Int] {
    []
  }

  func decode(tokenIds _: [Int], skipSpecialTokens _: Bool) -> String {
    ""
  }
}

/// Trivial `ResponseFormatParser` that emits no events. Used in emitter
/// lifecycle tests where we want to exercise the envelope path without
/// any per-format work.
struct NoOpParser: ResponseFormatParser {
  mutating func process(_: ParserInput) -> [ResponseStreamingEvent] {
    []
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    []
  }
}
