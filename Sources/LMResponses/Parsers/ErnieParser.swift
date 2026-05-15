// Copyright © Anthony DePasquale

import Foundation

/// Parser for Baidu's ERNIE 4.5 tool-call and reasoning format.
///
/// **Wire shape.** ERNIE 4.5 emits Hermes-shaped tool calls
/// (`<tool_call>{json}</tool_call>`) plus two ERNIE-specific
/// envelopes:
///
/// 1. A `</think>` reasoning closer (the opener is typically injected
///    into the prompt by the chat template, so the model emits only
///    the closer).
/// 2. An optional `<response>...</response>` content envelope around
///    the assistant's user-visible response.
///
/// ```text
/// reasoning text</think>
///
/// <response>
/// I'll look that up.
/// <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
/// </response>
/// ```
///
/// The parser extracts reasoning before `</think>` as a `.reasoning`
/// item, strips the `<response>` / `</response>` envelope from content,
/// trims a leading `\n` after `</think>` / `<response>` / `</response>`,
/// and emits any `<tool_call>` envelopes found in the body.
///
/// **Reasoning is opt-in by default**, mirroring vLLM's stance for
/// V3-family thinking and OLMo 3: pass `acceptThink: true` (default
/// `false`) plus the appropriate `initialState` to enable reasoning.
/// The initial state defaults to `.reasoning` because ERNIE 4.5
/// Thinking variants typically have `<think>` injected by the chat
/// template.
///
/// **Reference**: `vllm/tool_parsers/ernie45_tool_parser.py` and
/// `vllm/reasoning/ernie45_reasoning_parser.py`. Reference models
/// include `baidu/ERNIE-4.5-21B-A3B-Thinking`,
/// `baidu/ERNIE-4.5-VL-28B-A3B-PT`.
struct ErnieParser: ResponseFormatParser {
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let responseStart = "<response>"
  private static let responseEnd = "</response>"

  private let acceptThink: Bool
  private var thinkPreamble: ThinkPreambleExtractor

  /// Buffer of bytes that haven't yet been forwarded to the inner
  /// Hermes parser. Used to strip the `<response>` envelope, trim
  /// stray leading newlines after `</think>` / `<response>` /
  /// `</response>`, and hold partial-marker overlap across chunks.
  private var buffer: String = ""

  /// Inner Hermes parser handles `<tool_call>{json}</tool_call>`
  /// envelopes plus all interleaved message text. The Ernie envelope
  /// stripping happens in `feedHermes` before bytes reach it.
  ///
  /// Constructed lazily on first feed so it can be told its starting
  /// output index – we need to know whether a reasoning item has
  /// already been emitted (which would have taken slot 0).
  private var hermes: HermesParser?

  /// Sequence/index counters owned by the helper. Used to keep the
  /// reasoning preamble's events numbered consistently with whatever
  /// the inner Hermes parser emits. The Hermes parser starts at
  /// output index `nextOutputIndex` after the helper drains.
  private var nextSequence: Int = 0
  private var nextOutputIndex: Int = 0

  /// Set after the first byte has been forwarded to the inner Hermes
  /// parser, so we can stop trimming leading newlines (which is only
  /// relevant right after `</think>` / `<response>`).
  private var hermesStarted: Bool = false

  /// True after we've seen a `</think>` close in the input. Until then,
  /// we treat the prefix as reasoning content.
  private var reasoningClosed: Bool = false

  init(
    acceptThink: Bool = false,
    initialState: InitialState = .reasoning,
  ) {
    self.acceptThink = acceptThink
    if acceptThink {
      let preambleState: ThinkPreambleExtractor.InitialState = switch initialState {
        case .normal: .normal
        case .reasoning: .reasoning
      }
      thinkPreamble = ThinkPreambleExtractor(initialState: preambleState)
    } else {
      thinkPreamble = ThinkPreambleExtractor(initialState: .normal)
    }
    reasoningClosed = !acceptThink || initialState == .normal
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return drain(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = drain(isEnd: true)
    if hermes != nil {
      events.append(contentsOf: hermes!.finalize())
    }
    if acceptThink {
      events.append(contentsOf: thinkPreamble.finalizeIfOpen(nextSequence: &nextSequence))
    }
    return events
  }

  // MARK: Drain pipeline

  private mutating func drain(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    if acceptThink, thinkPreamble.phase != .done {
      events.append(contentsOf: thinkPreamble.drain(
        buffer: &buffer,
        isEnd: isEnd,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
      if thinkPreamble.phase != .done { return events }
      reasoningClosed = true
    }

    // Strip stray leading `\n` characters that follow `</think>` /
    // `<response>` / `</response>` until the first non-newline byte
    // reaches the inner Hermes parser. This must run on every drain
    // (not only the one that closed reasoning) because in streaming
    // mode the trailing newlines may arrive in a later chunk.
    if reasoningClosed, !hermesStarted {
      while buffer.first == "\n" {
        buffer.removeFirst()
      }
    }

    // Strip `<response>` / `</response>` envelope tokens from buffer
    // and forward the remainder to the inner Hermes parser.
    let safeBuffer = stripResponseTokens(in: buffer, isEnd: isEnd)
    if !safeBuffer.consumable.isEmpty {
      if hermes == nil {
        // Tell Hermes its first output_index slot, after any reasoning
        // already emitted by the preamble extractor.
        hermes = HermesParser(startingOutputIndex: nextOutputIndex)
      }
      events.append(contentsOf: hermes!.process(ParserInput(text: safeBuffer.consumable)))
      hermesStarted = true
    }
    buffer = safeBuffer.held

    return events
  }

  /// Walk `text` and split it into `consumable` (passable to Hermes)
  /// and `held` (must wait for more bytes to disambiguate). Strips any
  /// complete `<response>` / `</response>` literal it encounters; trims
  /// a leading `\n` that follows them.
  private func stripResponseTokens(
    in text: String,
    isEnd: Bool,
  ) -> (consumable: String, held: String) {
    var consumable = ""
    var current = text
    while !current.isEmpty {
      let startRange = current.range(of: Self.responseStart)
      let endRange = current.range(of: Self.responseEnd)

      let nextRange: Range<String.Index>?
      let nextLength: Int
      switch (startRange, endRange) {
        case let (.some(s), .some(e)):
          if s.lowerBound < e.lowerBound {
            nextRange = s; nextLength = Self.responseStart.count
          } else {
            nextRange = e; nextLength = Self.responseEnd.count
          }
        case let (.some(s), .none):
          nextRange = s; nextLength = Self.responseStart.count
        case let (.none, .some(e)):
          nextRange = e; nextLength = Self.responseEnd.count
        case (.none, .none):
          nextRange = nil; nextLength = 0
      }

      if let r = nextRange {
        let pre = String(current[current.startIndex ..< r.lowerBound])
        consumable += pre
        // Skip the marker.
        current = String(current[current.index(r.lowerBound, offsetBy: nextLength)...])
        // Trim a single leading `\n` so the marker plus its trailing
        // newline are erased atomically.
        if current.first == "\n" {
          current.removeFirst()
        }
        continue
      }

      // No more markers in `current`. Hold back any partial-marker
      // suffix unless we're at end of stream.
      let chars = Array(current)
      let startOverlap = partialOverlap(suffixOf: chars, with: Array(Self.responseStart))
      let endOverlap = partialOverlap(suffixOf: chars, with: Array(Self.responseEnd))
      let overlap = isEnd ? 0 : Swift.max(startOverlap, endOverlap)
      let safeEnd = chars.count - overlap
      if safeEnd > 0 {
        consumable += String(chars[0 ..< safeEnd])
      }
      let held = String(chars[safeEnd ..< chars.count])
      return (consumable, held)
    }
    return (consumable, "")
  }
}
