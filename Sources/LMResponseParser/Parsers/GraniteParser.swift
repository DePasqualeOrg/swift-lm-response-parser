// Copyright © Anthony DePasquale

import Foundation

/// Parser for IBM Granite 3.x's tool-call and reasoning format.
///
/// **Tool-call wire shape.** Tool calls appear inside a single JSON array
/// preceded by either of two markers:
///
/// - `<|tool_call|>` — Granite 3.0 (registered as a special token in the
///   tokenizer; surfaces as literal text after detokenization).
/// - `<tool_call>` — Granite 3.1 (plain string).
///
/// ```text
/// <|tool_call|> [{"name": "fn", "arguments": {"x": 1}}, ...]
/// ```
///
/// Each array element is a JSON object with a `name` field and either
/// `arguments` (canonical) or `parameters` (alias). Multiple calls share a
/// single envelope. Plain text may precede or follow the envelope and is
/// forwarded as message content – a deliberate departure from vLLM's
/// reference behavior, which strips surrounding content when tool calls
/// are present, but consistent with how the other parsers in this package
/// handle preamble.
///
/// **Reasoning wire shape.** Granite 3.2+ reasoning checkpoints emit
/// chain-of-thought between two prose delimiters at the start of the
/// response:
///
/// ```text
/// Here is my thought process: ... Here is my response: ... <|tool_call|> [...]
/// ```
///
/// Both `Here is my` and `Here's my` variants are accepted (mirroring
/// vLLM's `granite_reasoning_parser`). Reasoning detection is opt-in by
/// the model output: when the response doesn't start with one of the
/// think-start prefixes, the parser bypasses reasoning extraction and
/// flows directly to the tool-call phase. Granite 3.0 / 3.1 outputs
/// (which never emit the prose delimiters) pay only a small leading
/// hold-back overhead.
///
/// **Streaming.** The parser holds the array bytes until a complete
/// bracket-balanced array arrives, then emits all calls at once. Per-call
/// argument-delta streaming via partial-JSON is not implemented, matching
/// the trade-off in `Phi4MiniParser`.
struct GraniteParser: ResponseFormatParser {
  /// Initial reasoning phase. ``reasoning`` is used by continuation
  /// requests whose `priorOutput` ended inside an unclosed
  /// `Here is my thought process:` block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  // Granite 3.0 emits the special-token marker; Granite 3.1 emits the
  // plain-string marker. Both must be accepted.
  private static let toolMarkers: [String] = ["<|tool_call|>", "<tool_call>"]
  // Granite 3.2 reasoning prose delimiters. Both `Here is my` and
  // `Here's my` variants appear in observed outputs (the latter
  // primarily in quantized checkpoints).
  private static let thinkStarts: [String] = [
    "Here is my thought process:",
    "Here's my thought process:",
  ]
  private static let responseStarts: [String] = [
    "Here is my response:",
    "Here's my response:",
  ]

  private var buffer: String = ""
  private var openMessage: OpenMessage?
  private var openReasoning: OpenReasoning?
  private var phase: Phase
  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  /// The parser walks one of five phases per response. Granite emits
  /// at most one tool-call envelope per response, mirroring vLLM's
  /// reference parser, so once the array has been consumed everything
  /// else flows through as plain content.
  private enum Phase {
    /// At start of response: deciding whether to enter the reasoning
    /// phase based on a prefix check against the think-start markers.
    /// Holds back bytes that might still grow into a marker; once the
    /// hold is broken (or a complete marker arrives) transitions to
    /// either ``reasoning`` or ``lookingForMarker``.
    case preReasoning
    /// Inside a `Here is my thought process:` … `Here is my response:`
    /// block. Emits reasoning text and looks for the response-start
    /// marker.
    case reasoning
    /// Pre-marker: scanning the buffer for `<|tool_call|>` or
    /// `<tool_call>`. Bytes before a marker are emitted as message
    /// content; bytes that could be a partial marker prefix are held
    /// back.
    case lookingForMarker
    /// Marker consumed; waiting for the JSON array's opening `[` and
    /// then for the array to balance.
    case awaitingArray
    /// Array consumed (or surfaced as content on parse failure). All
    /// further bytes are plain content.
    case arrayConsumed
  }

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  init(initialState: InitialState = .normal) {
    switch initialState {
      case .normal:
        phase = .preReasoning
      case .reasoning:
        phase = .reasoning
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
      // Reasoning ran to EOS without a `Here is my response:` marker.
      // Surface as truncated rather than dropping the partial block.
      events.append(contentsOf: closeReasoning(status: .incomplete))
    }
    if !buffer.isEmpty {
      events.append(contentsOf: emitMessageDelta(text: buffer))
      buffer = ""
    }
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    return events
  }

  // MARK: Scan

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    var didProgress = true
    while didProgress {
      didProgress = false

      switch phase {
        case .arrayConsumed:
          if !buffer.isEmpty {
            events.append(contentsOf: emitMessageDelta(text: buffer))
            buffer = ""
          }
          return events

        case .preReasoning:
          // Empty buffer — nothing to decide yet.
          if buffer.isEmpty { return events }

          // Buffer starts with a complete think-start marker → enter
          // reasoning. The marker itself is consumed.
          var matchedMarker: String? = nil
          for marker in Self.thinkStarts where buffer.hasPrefix(marker) {
            matchedMarker = marker
            break
          }
          if let marker = matchedMarker {
            buffer.removeFirst(marker.count)
            phase = .reasoning
            didProgress = true
            continue
          }

          // Buffer is itself a (proper) prefix of a think-start marker
          // → hold; might still grow into the full marker.
          let isThinkPrefix = Self.thinkStarts.contains { marker in
            marker.hasPrefix(buffer)
          }
          if isThinkPrefix, !isEnd { return events }

          // Confirmed not a reasoning preamble (or end of stream).
          // Bypass reasoning and let the tool-call phase handle the
          // buffer. No bytes are consumed here – the next iteration
          // re-runs `lookingForMarker` on the same buffer.
          phase = .lookingForMarker
          didProgress = true

        case .reasoning:
          // Find the earliest response-start marker in the buffer.
          var earliestRespIdx: String.Index? = nil
          var earliestRespLen = 0
          for marker in Self.responseStarts {
            if let r = buffer.range(of: marker) {
              if earliestRespIdx == nil || r.lowerBound < earliestRespIdx! {
                earliestRespIdx = r.lowerBound
                earliestRespLen = marker.count
              }
            }
          }

          let safeEndIdx: String.Index
          if let idx = earliestRespIdx {
            safeEndIdx = idx
          } else if isEnd {
            safeEndIdx = buffer.endIndex
          } else {
            // Hold back any partial-marker overlap on either
            // response-start variant.
            let bufChars = Array(buffer)
            var maxOverlap = 0
            for marker in Self.responseStarts {
              let o = partialOverlap(suffixOf: bufChars, with: Array(marker))
              maxOverlap = Swift.max(maxOverlap, o)
            }
            safeEndIdx = buffer.index(buffer.endIndex, offsetBy: -maxOverlap)
          }

          if safeEndIdx > buffer.startIndex {
            let chunk = String(buffer[buffer.startIndex ..< safeEndIdx])
            events.append(contentsOf: emitReasoningDelta(text: chunk))
            buffer.removeFirst(buffer.distance(from: buffer.startIndex, to: safeEndIdx))
            didProgress = true
          }

          // If a response-start marker is now at the head of the
          // buffer, close reasoning and transition to tool-call phase.
          if earliestRespIdx != nil {
            buffer.removeFirst(earliestRespLen)
            events.append(contentsOf: closeReasoning(status: .completed))
            phase = .lookingForMarker
            didProgress = true
          }

        case .lookingForMarker:
          let earliest = Self.toolMarkers.compactMap { marker -> (range: Range<String.Index>, marker: String)? in
            guard let r = buffer.range(of: marker) else { return nil }
            return (r, marker)
          }.min(by: { $0.range.lowerBound < $1.range.lowerBound })

          guard let earliest else {
            // No marker visible. Emit safe content; hold back any
            // partial-marker suffix until the next chunk completes
            // or doesn't.
            let partial = trailingPartialMarkerOverlap(of: buffer)
            let safeEnd = isEnd ? buffer.count : buffer.count - partial
            if safeEnd > 0 {
              let safeText = String(buffer.prefix(safeEnd))
              events.append(contentsOf: emitMessageDelta(text: safeText))
              buffer.removeFirst(safeEnd)
              didProgress = safeEnd > 0
            }
            return events
          }

          let preText = String(buffer[buffer.startIndex ..< earliest.range.lowerBound])
          if !preText.isEmpty {
            events.append(contentsOf: emitMessageDelta(text: preText))
            buffer.removeFirst(preText.count)
            didProgress = true
            continue
          }

          // Buffer starts with the marker. Consume it and transition.
          buffer.removeFirst(earliest.marker.count)
          phase = .awaitingArray
          didProgress = true

        case .awaitingArray:
          // Skip whitespace, look for `[`. If only whitespace is
          // present so far, wait for more bytes (unless end-of-stream).
          let trimmed = buffer.drop(while: { $0.isWhitespace })
          guard let first = trimmed.first else {
            if isEnd {
              // Marker with no array body — silently drop; nothing
              // to surface.
              buffer = ""
              phase = .arrayConsumed
            }
            return events
          }
          if first != "[" {
            // Marker followed by something other than `[`. vLLM
            // treats this as a parse failure and returns the raw
            // text as content; we surface what's left in the buffer.
            let stray = String(buffer)
            events.append(contentsOf: emitMessageDelta(text: stray))
            buffer = ""
            phase = .arrayConsumed
            return events
          }

          let bracketStart = trimmed.startIndex
          guard let bracketEnd = matchingCloseBracket(in: buffer, openAt: bracketStart) else {
            if isEnd {
              // Truncated mid-array — surface what we have as
              // content.
              let stray = String(buffer)
              events.append(contentsOf: emitMessageDelta(text: stray))
              buffer = ""
              phase = .arrayConsumed
            }
            return events
          }

          let arrayText = String(buffer[bracketStart ... bracketEnd])
          let consumed = buffer.distance(from: buffer.startIndex, to: bracketEnd) + 1

          if let calls = parseToolCallArray(arrayText) {
            if openMessage != nil {
              events.append(contentsOf: closeMessage(status: .completed))
            }
            for call in calls {
              events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
            }
          } else {
            events.append(contentsOf: emitMessageDelta(text: arrayText))
          }
          buffer.removeFirst(consumed)
          phase = .arrayConsumed
          didProgress = true
      }
    }
    return events
  }

  // MARK: Bracket helpers

  /// Find the `]` that balances the `[` at `openAt`. Skips brackets
  /// inside JSON string literals so a string value containing `[` or
  /// `]` does not close the array prematurely.
  private func matchingCloseBracket(in slice: String, openAt: String.Index) -> String.Index? {
    var depth = 0
    var inString = false
    var escape = false
    var i = openAt
    while i < slice.endIndex {
      let ch = slice[i]
      if escape { escape = false; i = slice.index(after: i); continue }
      if ch == "\\" { escape = true; i = slice.index(after: i); continue }
      if inString {
        if ch == "\"" { inString = false }
        i = slice.index(after: i)
        continue
      }
      if ch == "\"" {
        inString = true
      } else if ch == "[" {
        depth += 1
      } else if ch == "]" {
        depth -= 1
        if depth == 0 { return i }
      }
      i = slice.index(after: i)
    }
    return nil
  }

  private func trailingPartialMarkerOverlap(of text: String) -> Int {
    let chars = Array(text)
    var maxOverlap = 0
    for marker in Self.toolMarkers {
      let overlap = partialOverlap(suffixOf: chars, with: Array(marker))
      maxOverlap = Swift.max(maxOverlap, overlap)
    }
    return maxOverlap
  }

  // MARK: JSON parsing

  private struct ParsedCall {
    let name: String
    let arguments: String
  }

  private func parseToolCallArray(_ arrayText: String) -> [ParsedCall]? {
    guard let data = arrayText.data(using: .utf8),
          let array = try? JSONSerialization.jsonObject(with: data) as? [Any]
    else {
      return nil
    }
    if array.isEmpty { return [] }
    var calls: [ParsedCall] = []
    for entry in array {
      guard let dict = entry as? [String: Any],
            let name = dict["name"] as? String
      else {
        return nil
      }
      // vLLM accepts only `arguments`; sglang's shared `parse_base_json`
      // accepts both. We follow the more lenient path, matching the
      // dual-key handling already in `Llama3Parser`, `JSONFallbackParser`,
      // `MistralParser`, and `Phi4MiniParser`.
      let argsValue = dict["arguments"] ?? dict["parameters"] ?? [String: Any]()
      guard let argsJSON = serializeJSONArgument(argsValue) else {
        return nil
      }
      calls.append(ParsedCall(name: name, arguments: argsJSON))
    }
    return calls
  }

  /// vLLM applies `json.dumps(function_call["arguments"])` to the
  /// raw value. That is usually an object, but JSON fragments like
  /// arrays, strings, numbers, booleans, and `null` should serialize
  /// as themselves rather than being rewritten to `{}`.
  private func serializeJSONArgument(_ value: Any) -> String? {
    guard JSONSerialization.isValidJSONObject([value]),
          let data = try? JSONSerialization.data(
            withJSONObject: value,
            options: [.sortedKeys, .withoutEscapingSlashes, .fragmentsAllowed],
          )
    else {
      return nil
    }
    return String(data: data, encoding: .utf8)
  }

  // MARK: Event emission

  private mutating func emitReasoningDelta(text: String) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var events: [ResponseStreamingEvent] = []
    if openReasoning == nil {
      events.append(contentsOf: openReasoningItem())
    }
    if var r = openReasoning {
      r.emittedText += text
      openReasoning = r
      events.append(.reasoningTextDelta(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        delta: text,
        sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  private mutating func openReasoningItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.reasoning)
    let outputIndex = takeOutputIndex()
    openReasoning = OpenReasoning(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .reasoning(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex, sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id, outputIndex: outputIndex, contentIndex: 0,
        part: .reasoningText(.init(text: "")), sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeReasoning(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let r = openReasoning else { return [] }
    openReasoning = nil
    let part = ReasoningTextContent(text: r.emittedText)
    return [
      .reasoningTextDone(.init(
        itemId: r.id, outputIndex: r.outputIndex, contentIndex: 0,
        text: r.emittedText, sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: r.id, outputIndex: r.outputIndex, contentIndex: 0,
        part: .reasoningText(part), sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .reasoning(.init(id: r.id, content: [.reasoningText(part)], status: status)),
        outputIndex: r.outputIndex, sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func emitMessageDelta(text: String) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem())
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

  private mutating func openMessageItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.message)
    let outputIndex = takeOutputIndex()
    openMessage = OpenMessage(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .message(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex, sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id, outputIndex: outputIndex, contentIndex: 0,
        part: .outputText(.init(text: "")), sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeMessage(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let msg = openMessage else { return [] }
    openMessage = nil
    let part = ResponseOutputText(text: msg.emittedText)
    return [
      .outputTextDone(.init(
        itemId: msg.id, outputIndex: msg.outputIndex, contentIndex: 0,
        text: msg.emittedText, sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: msg.id, outputIndex: msg.outputIndex, contentIndex: 0,
        part: .outputText(part), sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .message(.init(id: msg.id, content: [.outputText(part)], status: status)),
        outputIndex: msg.outputIndex, sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func emitToolCall(name: String, arguments: String) -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.functionCall)
    let callId = IDFactory.make(.callId)
    let outputIndex = takeOutputIndex()
    let openItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: name, arguments: "", status: .inProgress,
    )
    let doneItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: name, arguments: arguments, status: .completed,
    )
    var events: [ResponseStreamingEvent] = []
    events.append(.outputItemAdded(.init(
      item: .functionCall(openItem), outputIndex: outputIndex, sequenceNumber: takeSequence(),
    )))
    if !arguments.isEmpty {
      events.append(.functionCallArgumentsDelta(.init(
        itemId: id, outputIndex: outputIndex, delta: arguments, sequenceNumber: takeSequence(),
      )))
    }
    events.append(.functionCallArgumentsDone(.init(
      itemId: id, outputIndex: outputIndex, name: name, arguments: arguments, sequenceNumber: takeSequence(),
    )))
    events.append(.outputItemDone(.init(
      item: .functionCall(doneItem), outputIndex: outputIndex, sequenceNumber: takeSequence(),
    )))
    return events
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
