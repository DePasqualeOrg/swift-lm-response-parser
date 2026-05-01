// Copyright © Anthony DePasquale

import Foundation

/// Parser for the GPT-OSS / Harmony reserved-token protocol.
///
/// **Wire shape.** Harmony interleaves reasoning, plain messages, and tool
/// calls inside a fixed envelope built from seven reserved structural
/// tokens: `<|start|>`, `<|channel|>`, `<|message|>`, `<|constrain|>`,
/// `<|end|>`, `<|call|>`, `<|return|>`. A typical exchange:
///
/// ```text
/// <|channel|>analysis<|message|>Need to fetch the weather.<|end|>
/// <|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}<|call|>
/// <|start|>assistant<|channel|>final<|message|>It's 18°C in Paris.<|return|>
/// ```
///
/// **Channel routing.**
/// - `analysis` blocks ending in `<|end|>` carry chain-of-thought reasoning.
/// - `analysis` blocks ending in `<|call|>` and a `to=` target are built-in
///   tool calls (e.g., `to=browser.search`).
/// - `commentary` blocks ending in `<|end|>` are user-visible preamble text
///   (``ResponseOutputMessage/Phase/commentary``).
/// - `commentary` blocks ending in `<|call|>` and a `to=functions.NAME`
///   target are function calls.
/// - `final` blocks ending in `<|return|>` (or end-of-stream) are the final
///   answer (``ResponseOutputMessage/Phase/finalAnswer``).
///
/// **Message phase.** Harmony is the only format that populates
/// ``ResponseOutputMessage/phase``. The field is `nil` for every other
/// parser and for plain text emitted outside any channel block.
///
/// **Commentary filler.** After a tool call closes with `<|call|>`, models
/// sometimes emit a standalone `commentary` word before the next channel
/// header. This is a known malformed-output pattern; the parser filters it
/// (across chunk boundaries when needed) so it never leaks into a message.
///
/// **Marker matching: text-based.** Harmony's seven structural tokens
/// are reserved special tokens whose decoded text is canonical and
/// unambiguous (`<|start|>`, `<|channel|>`, …). The parser matches
/// markers in the detokenized text rather than keying off token IDs.
/// SGLang's `harmony_parser.py` takes the same approach. vLLM's
/// `gptoss_reasoning_parser.py` keys off token IDs instead, but the
/// motivations there (speculative decoding delivering many tokens per
/// step, `skip_special_tokens=True` configurations stripping the
/// structural tokens before they reach the parser, server-batch
/// performance) don't apply at the parser-library layer.
///
/// The `ParserTokenizer` parameter at construction and the
/// `ParserInput.tokenIds` field on every chunk are preserved on the
/// protocol surface as forward-looking infrastructure – none of the
/// shipped parsers read either. Switch over if any of these become
/// real:
///
/// - A model that emits `<|channel|>` (or any other Harmony marker)
///   as literal text via regular tokens rather than the reserved
///   special token, in response to prompts that ask it to echo the
///   string. Decoded text is identical; only token IDs differ.
/// - A tokenizer that decodes the structural tokens to non-canonical
///   text.
/// - A consumer-side detokenizer that strips special tokens before
///   `process(_:)` sees them.
struct HarmonyParser: ResponseFormatParser {
  /// Initial parser phase. Default is ``idle`` (start fresh outside any
  /// block). Set to ``inReasoning`` when the parser should resume an
  /// unclosed `analysis` block from a prior response – typically because
  /// the factory found an unclosed `<|channel|>analysis<|message|>` …
  /// block in `priorOutput`.
  enum InitialState: Equatable {
    case idle
    case inReasoning
  }

  private static let mStart = "<|start|>"
  private static let mChannel = "<|channel|>"
  private static let mMessage = "<|message|>"
  private static let mConstrain = "<|constrain|>"
  private static let mEnd = "<|end|>"
  private static let mCall = "<|call|>"
  private static let mReturn = "<|return|>"

  private static let allMarkers: [String] = [
    mStart, mChannel, mMessage, mConstrain, mEnd, mCall, mReturn,
  ]

  // `<|refusal|>` is in Harmony's encoding registry but unspecified in
  // the format docs and unimplemented by vLLM and sglang (both have
  // open TODOs). If a model ever emits it, our parser surfaces it as
  // content text — same as sglang. Implementing the token without a
  // documented spec would invent semantics.

  private enum MarkerKind {
    case start, channel, message, constrain, end, call, ret
  }

  private struct MarkerHit {
    let kind: MarkerKind
    let startIdx: Int
    let endIdx: Int
  }

  private static let markerTable: [(MarkerKind, String)] = [
    (.start, mStart),
    (.channel, mChannel),
    (.message, mMessage),
    (.constrain, mConstrain),
    (.end, mEnd),
    (.call, mCall),
    (.ret, mReturn),
  ]

  private enum State {
    case idle
    case afterStart
    case inChannelHeader
    case inReasoning
    case inMessage(phase: ResponseOutputMessage.Phase?)
    case inToolCallArgs(name: String)
  }

  /// Some serving stacks decode gpt-oss with `skip_special_tokens=True`,
  /// stripping the `<|...|>` Harmony markers and leaving only literal
  /// channel labels: `<|start|>assistant<|channel|>final<|message|>`
  /// collapses to `assistantfinal`. The parser detects which mode to use
  /// from the first content it sees.
  private enum Mode {
    case undecided
    case canonical
    case text
  }

  private static let textLabels = ["assistantfinal", "analysis", "commentary"]
  /// Channel-only labels that may sit behind an optional `assistant`
  /// prefix in text-fallback mode. Mirrors sglang's
  /// `(?:assistant)?\s*(analysis|commentary|...)` regex tolerance.
  private static let assistantPrefix = "assistant"
  private static let postAssistantLabels = ["analysis", "commentary"]

  private var buffer: String = ""
  private var parsedIdx: Int = 0
  private var state: State = .idle
  private var mode: Mode = .undecided

  private var openMessage: OpenMessage?
  private var openReasoning: OpenReasoning?
  private var openFunctionCall: OpenFunctionCall?
  private var pendingRecipient: String?

  /// True after a `<|call|>` boundary, until the parser confirms that the
  /// next text either is or is not the standalone "commentary" filler.
  private var filterCommentaryFiller: Bool = false

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var phase: ResponseOutputMessage.Phase?
    var emittedText: String = ""
  }

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenFunctionCall {
    var id: String
    var callId: String
    var outputIndex: Int
    var name: String
    var argsEmitted: String = ""
  }

  init(initialState: InitialState = .idle) {
    switch initialState {
      case .idle: state = .idle
      case .inReasoning: state = .inReasoning
    }
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return scan(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = scan(isEnd: true)
    if openReasoning != nil {
      events.append(contentsOf: closeReasoning(status: .incomplete))
    }
    if openFunctionCall != nil {
      events.append(contentsOf: closeFunctionCall(status: .incomplete))
    }
    if let msg = openMessage {
      // Final-answer blocks can end at EOS as completed because Harmony
      // models may omit `<|return|>` at stop. Other open messages are
      // genuinely truncated.
      let status: ItemStatus = msg.phase == .finalAnswer ? .completed : .incomplete
      events.append(contentsOf: closeMessage(status: status))
    }
    return events
  }

  // MARK: Scan loop

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    while true {
      let preIdx = parsedIdx
      let preTag = stateTag()
      let stepEvents = scanStep(isEnd: isEnd)
      events.append(contentsOf: stepEvents)
      if parsedIdx == preIdx, stateTag() == preTag, stepEvents.isEmpty {
        break
      }
    }
    return events
  }

  private mutating func scanStep(isEnd: Bool) -> [ResponseStreamingEvent] {
    if mode == .undecided {
      decideMode(isEnd: isEnd)
      // If the buffer is still ambiguous, hold until more arrives so
      // we don't emit a partial label prefix as content.
      if mode == .undecided { return [] }
    }
    if mode == .text {
      switch state {
        case .idle: return scanTextIdle(isEnd: isEnd)
        case .inReasoning: return scanTextReasoning(isEnd: isEnd)
        case let .inMessage(phase): return scanTextMessage(phase: phase, isEnd: isEnd)
        default:
          // Text mode never enters tool-call/header/start states.
          return []
      }
    }
    switch state {
      case .idle: return scanIdle(isEnd: isEnd)
      case .afterStart: return scanAfterStart(isEnd: isEnd)
      case .inChannelHeader: return scanChannelHeader(isEnd: isEnd)
      case .inReasoning: return scanReasoning(isEnd: isEnd)
      case let .inMessage(phase): return scanMessage(phase: phase, isEnd: isEnd)
      case let .inToolCallArgs(name): return scanToolCall(name: name, isEnd: isEnd)
    }
  }

  /// Pick canonical vs text mode based on what's in the buffer past the
  /// current parse cursor. If the input mixes both, canonical wins because
  /// markers are unambiguous; text mode is only entered when no markers
  /// are present and a leading text label is recognized. The decision
  /// considers two ambiguities that must hold the buffer instead of
  /// committing: a partial Harmony marker at the suffix
  /// (e.g., `complete text <|ret`) and a partial text label at the
  /// leading non-whitespace position (e.g., `analy`).
  private mutating func decideMode(isEnd: Bool) {
    let chars = Array(buffer)
    guard parsedIdx < chars.count else { return }
    for marker in HarmonyParser.allMarkers {
      if chars.firstIndexOf(substring: marker, after: parsedIdx) != nil {
        mode = .canonical
        return
      }
    }
    // A partial marker at the buffer suffix is canonical-only –
    // commit to canonical so the suffix gets held back as a marker
    // prefix instead of being emitted as content.
    if !isEnd, maxLeadingMarkerOverlap(suffixOf: chars) > 0 {
      mode = .canonical
      return
    }
    // No marker, complete or partial. Skip leading whitespace and
    // check for a text label.
    var i = parsedIdx
    while i < chars.count, chars[i].isWhitespace {
      i += 1
    }
    if i >= chars.count {
      if isEnd { mode = .canonical }
      return
    }
    let tail = String(chars[i...]).lowercased()
    for label in HarmonyParser.textLabels {
      if tail.hasPrefix(label) {
        mode = .text
        return
      }
      if !isEnd, label.hasPrefix(tail) {
        // Buffer might still complete into a label.
        return
      }
    }
    // Optional `assistant` prefix (with or without intervening
    // whitespace) before `analysis`/`commentary`. Mirrors sglang's
    // `(?:^|\s)(?:assistant)?\s*(analysis|commentary|...)` regex
    // tolerance for `skip_special_tokens=True` decodes that produce
    // the role token's text alongside the channel header.
    if tail.hasPrefix(HarmonyParser.assistantPrefix) {
      var j = i + HarmonyParser.assistantPrefix.count
      while j < chars.count, chars[j].isWhitespace {
        j += 1
      }
      let postTail = j < chars.count ? String(chars[j...]).lowercased() : ""
      for label in HarmonyParser.postAssistantLabels {
        if postTail.hasPrefix(label) {
          mode = .text
          return
        }
        if !isEnd, label.hasPrefix(postTail) {
          return
        }
      }
    }
    // Leading non-whitespace is neither a marker nor a text label –
    // route as plain content via canonical mode.
    mode = .canonical
  }

  private func stateTag() -> Int {
    switch state {
      case .idle: 0
      case .afterStart: 1
      case .inChannelHeader: 2
      case .inReasoning: 3
      case .inMessage: 4
      case .inToolCallArgs: 5
    }
  }

  // MARK: Idle (text outside any block)

  private mutating func scanIdle(isEnd: Bool) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    var events: [ResponseStreamingEvent] = []

    let nextMarker = findNextMarker(in: chars, from: parsedIdx)
    let textEnd: Int = if let m = nextMarker {
      m.startIdx
    } else if isEnd {
      chars.count
    } else {
      chars.count - maxLeadingMarkerOverlap(suffixOf: chars)
    }

    if textEnd > parsedIdx {
      let textChunk = String(chars[parsedIdx ..< textEnd])
      if filterCommentaryFiller {
        let result = consumeCommentaryFiller(
          in: textChunk, isEnd: isEnd, hasMarkerAfter: nextMarker != nil,
        )
        switch result {
          case .holdAll:
            return events
          case let .consumed(n):
            parsedIdx += n
            filterCommentaryFiller = false
            let restEnd = textEnd
            if restEnd > parsedIdx {
              let rest = String(chars[parsedIdx ..< restEnd])
              events.append(contentsOf: emitMessageContent(rest))
              parsedIdx = restEnd
            }
          case .noMatch:
            filterCommentaryFiller = false
            events.append(contentsOf: emitMessageContent(textChunk))
            parsedIdx = textEnd
        }
      } else {
        events.append(contentsOf: emitMessageContent(textChunk))
        parsedIdx = textEnd
      }
    }

    guard let m = nextMarker, m.startIdx == parsedIdx else { return events }
    // Reaching a block-opening marker closes the post-call filler
    // window. If `<|call|>` was followed directly by `<|start|>` or
    // `<|channel|>` with no idle text between, `consumeCommentaryFiller`
    // never ran and the flag would otherwise stay armed for the next
    // idle region — a rare but real way to filter legitimate content
    // from a later block.
    switch m.kind {
      case .start:
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        parsedIdx = m.endIdx
        state = .afterStart
        filterCommentaryFiller = false
      case .channel:
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        parsedIdx = m.endIdx
        state = .inChannelHeader
        filterCommentaryFiller = false
      case .message, .end, .call, .ret, .constrain:
        // Stray structural token outside any block; silently consume.
        parsedIdx = m.endIdx
    }
    return events
  }

  private mutating func emitMessageContent(_ text: String) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem(phase: nil))
    }
    if var msg = openMessage {
      msg.emittedText += text
      openMessage = msg
      events.append(.outputTextDelta(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        delta: text,
        sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  // MARK: After-start (role text – discarded)

  private mutating func scanAfterStart(isEnd: Bool) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    if let m = findNextMarker(in: chars, from: parsedIdx) {
      let roleHeader = String(chars[parsedIdx ..< m.startIdx])
      pendingRecipient = extractToolName(from: roleHeader)
      // Skip the role-text bytes between <|start|> and the next marker.
      parsedIdx = m.startIdx
      switch m.kind {
        case .channel:
          parsedIdx = m.endIdx
          state = .inChannelHeader
        case .message:
          // Tool-response message: content streams to <|end|> as a
          // no-phase message item.
          pendingRecipient = nil
          parsedIdx = m.endIdx
          state = .inMessage(phase: nil)
        case .start:
          pendingRecipient = nil
          parsedIdx = m.endIdx
        default:
          pendingRecipient = nil
          parsedIdx = m.endIdx
      }
    } else if isEnd {
      pendingRecipient = nil
      parsedIdx = chars.count
    } else {
      parsedIdx = chars.count - maxLeadingMarkerOverlap(suffixOf: chars)
    }
    return []
  }

  // MARK: Channel header (between <|channel|> and <|message|>)

  private mutating func scanChannelHeader(isEnd _: Bool) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    guard let msgIdx = chars.firstIndexOf(substring: HarmonyParser.mMessage, after: parsedIdx) else {
      return []
    }
    let headerText = String(chars[parsedIdx ..< msgIdx])
    parsedIdx = msgIdx + Array(HarmonyParser.mMessage).count
    let info = parseChannelHeader(headerText, inheritedToolName: pendingRecipient)
    pendingRecipient = nil
    switch info.channel {
      case .analysis:
        if let name = info.toolName {
          state = .inToolCallArgs(name: name)
          return openFunctionCallItem(name: name)
        }
        state = .inReasoning
        return []
      case .commentary:
        if let name = info.toolName {
          state = .inToolCallArgs(name: name)
          return openFunctionCallItem(name: name)
        }
        state = .inMessage(phase: .commentary)
        return []
      case .final:
        state = .inMessage(phase: .finalAnswer)
        return []
      case .unknown:
        // Unknown channel – fall back to no-phase message so the content
        // is preserved rather than dropped.
        state = .inMessage(phase: nil)
        return []
    }
  }

  // MARK: Reasoning content

  private mutating func scanReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    var events: [ResponseStreamingEvent] = []
    // `<|return|>` is included for lenient handling of malformed
    // streams, mirroring `scanMessage`. A well-formed reasoning
    // block closes with `<|end|>`, but treating `<|return|>` as a
    // logical close prevents leaking literal token bytes into the
    // emitted reasoning text when the model misuses it.
    let endMarkers: [(MarkerKind, String)] = [
      (.end, HarmonyParser.mEnd),
      (.ret, HarmonyParser.mReturn),
      (.call, HarmonyParser.mCall),
      (.start, HarmonyParser.mStart),
      (.channel, HarmonyParser.mChannel),
    ]
    let earliest = findEarliestMarker(in: chars, from: parsedIdx, candidates: endMarkers)
    let textEnd: Int = if let e = earliest {
      e.startIdx
    } else if isEnd {
      chars.count
    } else {
      chars.count - maxOverlap(suffixOf: chars, with: endMarkers.map { $0.1 })
    }
    if textEnd > parsedIdx {
      let chunk = String(chars[parsedIdx ..< textEnd])
      if !chunk.isEmpty {
        if openReasoning == nil {
          events.append(contentsOf: openReasoningItem())
        }
        if var r = openReasoning {
          r.emittedText += chunk
          openReasoning = r
          events.append(.reasoningDelta(.init(
            itemId: r.id,
            outputIndex: r.outputIndex,
            contentIndex: 0,
            delta: chunk,
            sequenceNumber: takeSequence(),
          )))
        }
      }
      parsedIdx = textEnd
    }
    guard let e = earliest, e.startIdx == parsedIdx else { return events }
    switch e.kind {
      case .end, .ret:
        events.append(contentsOf: closeReasoning(status: .completed))
        parsedIdx = e.endIdx
        state = .idle
      case .call:
        // analysis with <|call|> = built-in tool call header; we close
        // the reasoning we accumulated and arm the commentary filler
        // filter so any trailing "commentary" word is discarded.
        events.append(contentsOf: closeReasoning(status: .completed))
        filterCommentaryFiller = true
        parsedIdx = e.endIdx
        state = .idle
      case .start:
        events.append(contentsOf: closeReasoning(status: .completed))
        parsedIdx = e.endIdx
        state = .afterStart
      case .channel:
        events.append(contentsOf: closeReasoning(status: .completed))
        parsedIdx = e.endIdx
        state = .inChannelHeader
      default:
        parsedIdx = e.endIdx
    }
    return events
  }

  // MARK: Message content

  private mutating func scanMessage(
    phase: ResponseOutputMessage.Phase?,
    isEnd: Bool,
  ) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    var events: [ResponseStreamingEvent] = []
    // Any structural marker counts as a logical close. sglang's
    // `_parse_block` is stricter – final channels accept only
    // `<|return|>`, commentary only `<|end|>`/`<|call|>` – but in
    // practice models occasionally emit the wrong close for a
    // channel and the lenient handling lets us extract useful
    // content from those streams instead of including raw marker
    // text in the message body. The "SGLang adversarial" test suite
    // pins this behavior. `<|start|>` and `<|channel|>` always
    // count: they begin a new block, so the current one is over.
    let endMarkers: [(MarkerKind, String)] = [
      (.end, HarmonyParser.mEnd),
      (.ret, HarmonyParser.mReturn),
      (.call, HarmonyParser.mCall),
      (.start, HarmonyParser.mStart),
      (.channel, HarmonyParser.mChannel),
    ]
    let earliest = findEarliestMarker(in: chars, from: parsedIdx, candidates: endMarkers)
    let textEnd: Int = if let e = earliest {
      e.startIdx
    } else if isEnd {
      chars.count
    } else {
      chars.count - maxOverlap(suffixOf: chars, with: endMarkers.map { $0.1 })
    }
    if textEnd > parsedIdx {
      let chunk = String(chars[parsedIdx ..< textEnd])
      if !chunk.isEmpty {
        if openMessage == nil {
          events.append(contentsOf: openMessageItem(phase: phase))
        }
        if var msg = openMessage {
          msg.emittedText += chunk
          openMessage = msg
          events.append(.outputTextDelta(.init(
            itemId: msg.id,
            outputIndex: msg.outputIndex,
            contentIndex: 0,
            delta: chunk,
            sequenceNumber: takeSequence(),
          )))
        }
      }
      parsedIdx = textEnd
    }
    guard let e = earliest, e.startIdx == parsedIdx else { return events }
    switch e.kind {
      case .end, .ret:
        // Diverges from sglang's `_parse_block`, which absorbs trailing
        // TEXT after `<|return|>` into the same final message. We close
        // on `<|return|>` and surface subsequent text as a fresh
        // no-phase message via `scanIdle`. With auto-injected halt the
        // case shouldn't arise in streaming; offline-parse callers get
        // cleaner item separation when it does.
        events.append(contentsOf: closeMessage(status: .completed))
        parsedIdx = e.endIdx
        state = .idle
      case .call:
        events.append(contentsOf: closeMessage(status: .completed))
        filterCommentaryFiller = true
        parsedIdx = e.endIdx
        state = .idle
      case .start:
        events.append(contentsOf: closeMessage(status: .completed))
        parsedIdx = e.endIdx
        state = .afterStart
      case .channel:
        events.append(contentsOf: closeMessage(status: .completed))
        parsedIdx = e.endIdx
        state = .inChannelHeader
      default:
        parsedIdx = e.endIdx
    }
    return events
  }

  // MARK: Tool call arguments

  private mutating func scanToolCall(name _: String, isEnd: Bool) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    var events: [ResponseStreamingEvent] = []
    let endMarkers: [(MarkerKind, String)] = [
      (.call, HarmonyParser.mCall),
      (.end, HarmonyParser.mEnd),
      (.start, HarmonyParser.mStart),
      (.channel, HarmonyParser.mChannel),
    ]
    let earliest = findEarliestMarker(in: chars, from: parsedIdx, candidates: endMarkers)
    let textEnd: Int = if let e = earliest {
      e.startIdx
    } else if isEnd {
      chars.count
    } else {
      chars.count - maxOverlap(suffixOf: chars, with: endMarkers.map { $0.1 })
    }

    // sglang's harmony_parser strips both ends of the tool-call body
    // before producing a `tool_call` event. We approximate this in
    // the streaming path by skipping leading whitespace before the
    // first emitted byte and always trimming trailing whitespace
    // from the emit. Whitespace at the right edge is HELD when no
    // close marker is in view yet (parsedIdx stays before it) – the
    // bytes will either become interior the moment non-whitespace
    // follows (next scan emits the held ws + new content) or get
    // dropped entirely when the close marker arrives. That keeps
    // the cumulative emit equal to sglang's stripped one-shot
    // output, including across arbitrary chunk boundaries.
    var emitStart = parsedIdx
    if let fc = openFunctionCall, fc.argsEmitted.isEmpty {
      while emitStart < textEnd, chars[emitStart].isWhitespace {
        emitStart += 1
      }
    }
    var emitEnd = textEnd
    while emitEnd > emitStart, chars[emitEnd - 1].isWhitespace {
      emitEnd -= 1
    }
    if emitEnd > emitStart {
      let chunk = String(chars[emitStart ..< emitEnd])
      if var fc = openFunctionCall {
        fc.argsEmitted += chunk
        openFunctionCall = fc
        events.append(.functionCallArgumentsDelta(.init(
          itemId: fc.id,
          outputIndex: fc.outputIndex,
          delta: chunk,
          sequenceNumber: takeSequence(),
        )))
      }
    }
    // Advance the cursor:
    //   - close marker visible OR EOS: skip past the trimmed
    //     trailing whitespace (it's discarded, matching strip).
    //   - mid-stream, no close: hold the trailing whitespace by
    //     leaving parsedIdx at `emitEnd`, so later non-whitespace
    //     content emits the held bytes as interior characters.
    if earliest != nil || isEnd {
      if textEnd > parsedIdx { parsedIdx = textEnd }
    } else {
      if emitEnd > parsedIdx { parsedIdx = emitEnd }
    }
    guard let e = earliest, e.startIdx == parsedIdx else { return events }
    switch e.kind {
      case .call:
        events.append(contentsOf: closeFunctionCall(status: .completed))
        filterCommentaryFiller = true
        parsedIdx = e.endIdx
        state = .idle
      case .end:
        events.append(contentsOf: closeFunctionCall(status: .completed))
        parsedIdx = e.endIdx
        state = .idle
      case .start:
        events.append(contentsOf: closeFunctionCall(status: .completed))
        parsedIdx = e.endIdx
        state = .afterStart
      case .channel:
        events.append(contentsOf: closeFunctionCall(status: .completed))
        parsedIdx = e.endIdx
        state = .inChannelHeader
      default:
        parsedIdx = e.endIdx
    }
    return events
  }

  // MARK: Text-mode scanning (for `skip_special_tokens=True` output)

  private mutating func scanTextIdle(isEnd: Bool) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    // Skip leading whitespace before a label.
    var i = parsedIdx
    while i < chars.count, chars[i].isWhitespace {
      i += 1
    }
    parsedIdx = i
    guard i < chars.count else { return [] }

    let slice = String(chars[i...]).lowercased()
    // `assistantfinal` must be checked before the `assistant` prefix
    // path – it is the most specific label and the prefix path would
    // otherwise consume `assistant` greedily and fail to find a
    // following channel.
    if slice.hasPrefix("assistantfinal") {
      parsedIdx = i + "assistantfinal".count
      state = .inMessage(phase: .finalAnswer)
      return []
    }
    if slice.hasPrefix("analysis") {
      parsedIdx = i + "analysis".count
      state = .inReasoning
      return []
    }
    if slice.hasPrefix("commentary") {
      parsedIdx = i + "commentary".count
      state = .inMessage(phase: .commentary)
      return []
    }
    // Optional `assistant` prefix with optional whitespace, mirroring
    // sglang's `(?:^|\s)(?:assistant)?\s*(analysis|commentary|...)`.
    // Reached when the slice did not start with a bare label.
    if slice.hasPrefix(HarmonyParser.assistantPrefix) {
      var j = i + HarmonyParser.assistantPrefix.count
      while j < chars.count, chars[j].isWhitespace {
        j += 1
      }
      let postSlice = j < chars.count ? String(chars[j...]).lowercased() : ""
      if postSlice.hasPrefix("analysis") {
        parsedIdx = j + "analysis".count
        state = .inReasoning
        return []
      }
      if postSlice.hasPrefix("commentary") {
        parsedIdx = j + "commentary".count
        state = .inMessage(phase: .commentary)
        return []
      }
    }
    // Slice didn't match any label fully. If it could still complete
    // into one, hold; otherwise forward the leading bytes as content.
    if isEnd {
      let rest = String(chars[parsedIdx ..< chars.count])
      parsedIdx = chars.count
      return emitMessageContent(rest)
    }
    let sliceIsLabelPrefix = HarmonyParser.textLabels.contains { label in
      label.lowercased().hasPrefix(slice)
    }
    if sliceIsLabelPrefix { return [] }
    // `assistant<ws*><partial-label>` is also a hold case: the slice
    // already contains `assistant`, and the post-prefix slice is a
    // prefix of `analysis` or `commentary`.
    if slice.hasPrefix(HarmonyParser.assistantPrefix) {
      var j = i + HarmonyParser.assistantPrefix.count
      while j < chars.count, chars[j].isWhitespace {
        j += 1
      }
      let postSlice = j < chars.count ? String(chars[j...]).lowercased() : ""
      for label in HarmonyParser.postAssistantLabels {
        if label.hasPrefix(postSlice) { return [] }
      }
    }
    let chunk = String(chars[parsedIdx ..< chars.count])
    parsedIdx = chars.count
    return emitMessageContent(chunk)
  }

  private mutating func scanTextReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    var events: [ResponseStreamingEvent] = []
    let nextLabelIdx = nextTextLabelIndex(in: chars, from: parsedIdx)
    let textEnd: Int = if let n = nextLabelIdx {
      n
    } else if isEnd {
      chars.count
    } else {
      chars.count - maxOverlap(suffixOf: chars, with: HarmonyParser.textLabels)
    }
    if textEnd > parsedIdx {
      var chunk = String(chars[parsedIdx ..< textEnd])
      // Trim a single leading whitespace to match sglang's behavior:
      // "analysis I need to think" → reasoning content "I need to think".
      if openReasoning == nil, chunk.first?.isWhitespace == true {
        chunk.removeFirst()
      }
      if !chunk.isEmpty {
        if openReasoning == nil {
          events.append(contentsOf: openReasoningItem())
        }
        if var r = openReasoning {
          r.emittedText += chunk
          openReasoning = r
          events.append(.reasoningDelta(.init(
            itemId: r.id,
            outputIndex: r.outputIndex,
            contentIndex: 0,
            delta: chunk,
            sequenceNumber: takeSequence(),
          )))
        }
      }
      parsedIdx = textEnd
    }
    if nextLabelIdx != nil {
      events.append(contentsOf: closeReasoning(status: .completed))
      state = .idle
    }
    return events
  }

  private mutating func scanTextMessage(
    phase: ResponseOutputMessage.Phase?,
    isEnd: Bool,
  ) -> [ResponseStreamingEvent] {
    let chars = Array(buffer)
    var events: [ResponseStreamingEvent] = []
    // The final-answer phase is the absorbing state – once entered, no
    // further label transitions occur. Commentary can transition to
    // final via `assistantfinal`.
    let watchedLabels: [String] = if phase == .finalAnswer {
      []
    } else {
      ["assistantfinal"]
    }
    let nextLabelIdx: Int? = {
      if watchedLabels.isEmpty { return nil }
      return nextIndex(in: chars, from: parsedIdx, of: watchedLabels)
    }()
    let textEnd: Int = if let n = nextLabelIdx {
      n
    } else if isEnd {
      chars.count
    } else if !watchedLabels.isEmpty {
      chars.count - maxOverlap(suffixOf: chars, with: watchedLabels)
    } else {
      chars.count
    }
    if textEnd > parsedIdx {
      var chunk = String(chars[parsedIdx ..< textEnd])
      if openMessage == nil, chunk.first?.isWhitespace == true {
        chunk.removeFirst()
      }
      if !chunk.isEmpty {
        if openMessage == nil {
          events.append(contentsOf: openMessageItem(phase: phase))
        }
        if var msg = openMessage {
          msg.emittedText += chunk
          openMessage = msg
          events.append(.outputTextDelta(.init(
            itemId: msg.id,
            outputIndex: msg.outputIndex,
            contentIndex: 0,
            delta: chunk,
            sequenceNumber: takeSequence(),
          )))
        }
      }
      parsedIdx = textEnd
    }
    if nextLabelIdx != nil {
      events.append(contentsOf: closeMessage(status: .completed))
      state = .idle
    }
    return events
  }

  private func nextTextLabelIndex(in chars: [Character], from start: Int) -> Int? {
    nextIndex(in: chars, from: start, of: HarmonyParser.textLabels)
  }

  private func nextIndex(in chars: [Character], from start: Int, of needles: [String]) -> Int? {
    // ASCII-only case folding done in-place per comparison rather than
    // materializing a lowered copy of `chars` on every call – long
    // analysis blocks can hit this hot path once per scan, and a
    // per-call O(n) allocation otherwise compounds to O(n^2) over the
    // life of the stream. Label characters are ASCII; using
    // `Character(_.lowercased())` would collapse `ß` -> `ss` to a
    // single grapheme and shift later positions.
    var best: Int? = nil
    for needle in needles {
      let n = Array(needle)
      if let idx = caseInsensitiveFirstIndex(in: chars, from: start, of: n) {
        if best == nil || idx < best! { best = idx }
      }
    }
    return best
  }

  private func caseInsensitiveFirstIndex(
    in chars: [Character], from start: Int, of needle: [Character],
  ) -> Int? {
    if needle.isEmpty || chars.count - start < needle.count { return nil }
    let last = chars.count - needle.count
    var i = start
    while i <= last {
      var match = true
      for j in 0 ..< needle.count {
        if asciiLowercased(chars[i + j]) != asciiLowercased(needle[j]) {
          match = false
          break
        }
      }
      if match { return i }
      i += 1
    }
    return nil
  }

  private func asciiLowercased(_ ch: Character) -> Character {
    if let asc = ch.asciiValue, asc >= 0x41, asc <= 0x5A {
      return Character(Unicode.Scalar(asc + 0x20))
    }
    return ch
  }

  // MARK: Header parsing

  private struct ChannelInfo {
    let channel: ChannelType
    let toolName: String?
  }

  private enum ChannelType {
    case analysis, commentary, final, unknown
  }

  private func parseChannelHeader(_ raw: String, inheritedToolName: String? = nil) -> ChannelInfo {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    let lower = trimmed.lowercased()
    let channel: ChannelType = if matchesChannelName(lower, name: "analysis") {
      .analysis
    } else if matchesChannelName(lower, name: "commentary") {
      .commentary
    } else if matchesChannelName(lower, name: "final") {
      .final
    } else {
      .unknown
    }

    let toolName = extractToolName(from: raw) ?? inheritedToolName
    return ChannelInfo(channel: channel, toolName: toolName)
  }

  private func extractToolName(from raw: String) -> String? {
    guard let toRange = raw.range(of: "to=") else { return nil }
    let after = raw[toRange.upperBound...]
    var name = ""
    for c in after {
      if c.isWhitespace || c == "<" { break }
      name.append(c)
    }
    if name.hasPrefix("functions.") {
      // Strip the `functions.` namespace prefix so the emitted
      // function name matches the schema-side identifier the consumer
      // registered (`get_weather`, not `functions.get_weather`).
      // Reject `to=functions.` with no suffix – emitting a function
      // call with an empty name is strictly worse than routing the
      // block as plain commentary, since downstream tool dispatch
      // can't act on it. Matches the spirit of sglang's behavior,
      // where the regex accepts an empty post-prefix name but the
      // resulting empty key silently fails to dispatch.
      let stripped = String(name.dropFirst("functions.".count))
      return stripped.isEmpty ? nil : stripped
    }
    // Built-in tool targets (`to=browser.search`, `to=python`, etc.)
    // keep their full prefixed name. The `functions.` namespace is
    // reserved for caller-supplied tools; everything else is a
    // model-side built-in whose identity is the prefixed string.
    return name.isEmpty ? nil : name
  }

  /// Match a channel name as a whole token, not as a prefix substring.
  /// Accepts `name` followed by end-of-header, whitespace, or `<`
  /// (for `<|constrain|>` etc.) – but not e.g. `analysis-foo`.
  private func matchesChannelName(_ header: String, name: String) -> Bool {
    guard header.hasPrefix(name) else { return false }
    let after = header.index(header.startIndex, offsetBy: name.count)
    if after == header.endIndex { return true }
    let next = header[after]
    return next.isWhitespace || next == "<"
  }

  // MARK: Commentary-filler filter

  private enum FillerResult {
    case holdAll
    case consumed(Int)
    case noMatch
  }

  private func consumeCommentaryFiller(
    in slice: String,
    isEnd: Bool,
    hasMarkerAfter: Bool,
  ) -> FillerResult {
    let target = "commentary"
    var i = slice.startIndex
    while i < slice.endIndex, slice[i].isWhitespace {
      i = slice.index(after: i)
    }
    let wsCount = slice.distance(from: slice.startIndex, to: i)
    let remainder = String(slice[i...]).lowercased()

    if remainder.hasPrefix(target) {
      // Consume "commentary" plus any trailing whitespace.
      let afterTargetIdx = slice.index(slice.startIndex, offsetBy: wsCount + target.count)
      var j = afterTargetIdx
      while j < slice.endIndex, slice[j].isWhitespace {
        j = slice.index(after: j)
      }
      return .consumed(slice.distance(from: slice.startIndex, to: j))
    }
    if !remainder.isEmpty, target.hasPrefix(remainder) {
      // Partial match.
      if isEnd || hasMarkerAfter {
        // Final answer for this slice; treat it as filler and drop.
        return .consumed(slice.count)
      }
      return .holdAll
    }
    return .noMatch
  }

  // MARK: Marker scanning

  private func findNextMarker(in chars: [Character], from: Int) -> MarkerHit? {
    var i = from
    let n = chars.count
    while i < n {
      if i + 1 < n, chars[i] == "<", chars[i + 1] == "|" {
        if let hit = matchKnownMarker(in: chars, at: i) {
          return hit
        }
      }
      i += 1
    }
    return nil
  }

  private func findEarliestMarker(
    in chars: [Character],
    from: Int,
    candidates: [(MarkerKind, String)],
  ) -> MarkerHit? {
    var i = from
    let n = chars.count
    while i < n {
      if i + 1 < n, chars[i] == "<", chars[i + 1] == "|" {
        for (kind, str) in candidates {
          let mc = Array(str)
          if i + mc.count <= n, chars[i ..< (i + mc.count)].elementsEqual(mc) {
            return MarkerHit(kind: kind, startIdx: i, endIdx: i + mc.count)
          }
        }
      }
      i += 1
    }
    return nil
  }

  private func matchKnownMarker(in chars: [Character], at i: Int) -> MarkerHit? {
    for (kind, str) in HarmonyParser.markerTable {
      let mc = Array(str)
      if i + mc.count <= chars.count, chars[i ..< (i + mc.count)].elementsEqual(mc) {
        return MarkerHit(kind: kind, startIdx: i, endIdx: i + mc.count)
      }
    }
    return nil
  }

  private func maxLeadingMarkerOverlap(suffixOf chars: [Character]) -> Int {
    maxOverlap(suffixOf: chars, with: HarmonyParser.allMarkers)
  }

  private func maxOverlap(suffixOf chars: [Character], with markers: [String]) -> Int {
    var best = 0
    for marker in markers {
      let m = Array(marker)
      let overlap = partialOverlap(suffixOf: chars, with: m)
      if overlap > best { best = overlap }
    }
    return best
  }

  // MARK: Item open/close

  private mutating func openMessageItem(phase: ResponseOutputMessage.Phase?) -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.message)
    let outputIndex = takeOutputIndex()
    openMessage = OpenMessage(id: id, outputIndex: outputIndex, phase: phase)
    return [
      .outputItemAdded(.init(
        item: .message(.init(id: id, content: [], status: .inProgress, phase: phase)),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id,
        outputIndex: outputIndex,
        contentIndex: 0,
        part: .outputText(.init(text: "")),
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeMessage(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let msg = openMessage else { return [] }
    openMessage = nil
    let part = ResponseOutputText(text: msg.emittedText)
    return [
      .outputTextDone(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        text: msg.emittedText,
        sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        part: .outputText(part),
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .message(.init(
          id: msg.id,
          content: [.outputText(part)],
          status: status,
          phase: msg.phase,
        )),
        outputIndex: msg.outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func openReasoningItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.reasoning)
    let outputIndex = takeOutputIndex()
    openReasoning = OpenReasoning(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .reasoning(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id,
        outputIndex: outputIndex,
        contentIndex: 0,
        part: .reasoningText(.init(text: "")),
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeReasoning(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let r = openReasoning else { return [] }
    openReasoning = nil
    let part = ReasoningTextContent(text: r.emittedText)
    return [
      .reasoningDone(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        text: r.emittedText,
        sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        part: .reasoningText(part),
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .reasoning(.init(
          id: r.id,
          content: [.reasoningText(part)],
          status: status,
        )),
        outputIndex: r.outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func openFunctionCallItem(name: String) -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.functionCall)
    let callId = IDFactory.make(.callId)
    let outputIndex = takeOutputIndex()
    openFunctionCall = OpenFunctionCall(
      id: id, callId: callId, outputIndex: outputIndex, name: name,
    )
    let openItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: name, arguments: "", status: .inProgress,
    )
    return [
      .outputItemAdded(.init(
        item: .functionCall(openItem),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeFunctionCall(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let fc = openFunctionCall else { return [] }
    openFunctionCall = nil
    let doneItem = ResponseFunctionToolCall(
      id: fc.id,
      callId: fc.callId,
      name: fc.name,
      arguments: fc.argsEmitted,
      status: status,
    )
    return [
      .functionCallArgumentsDone(.init(
        itemId: fc.id,
        outputIndex: fc.outputIndex,
        arguments: fc.argsEmitted,
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .functionCall(doneItem),
        outputIndex: fc.outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
