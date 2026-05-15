// Copyright © Anthony DePasquale

import Foundation

/// Parser for MiniMax-Text-01 / M1 (the non-M2 line) tool-call format.
///
/// **Wire shape.** Tool calls share a single envelope `<tool_calls>` …
/// `</tool_calls>` containing **NDJSON** – one `{"name", "arguments"}`
/// JSON object per line, separated by newlines (not a JSON array):
///
/// ```text
/// <tool_calls>
/// {"name": "get_weather", "arguments": {"city": "Paris"}}
/// {"name": "get_time", "arguments": {"timezone": "UTC"}}
/// </tool_calls>
/// ```
///
/// **Reasoning interaction.** MiniMax-Text-01 / M1 may emit a
/// `<think>...</think>` reasoning preamble. vLLM's `minimax` tool parser
/// is tool-only: it strips any `<tool_calls>` envelope appearing inside
/// `<think>` from the input before extraction so reasoning-block
/// "examples" don't surface as real tool calls. This parser mirrors that
/// behavior – an envelope wholly inside `<think>...</think>` is treated
/// as message content, not as a tool call. Reasoning extraction itself
/// is out of scope here (see §4 of the coverage doc).
///
/// **Streaming.** The parser holds the envelope bytes until
/// `</tool_calls>` arrives, then emits all NDJSON entries at once. Per-
/// call argument-delta streaming is not implemented (mirrors `Jamba`,
/// `Phi4Mini`).
///
/// **Naming caveat.** Distinct from `.miniMaxM2` (which uses
/// `<minimax:tool_call>` XML invoke/parameter pairs – a completely
/// different shape). The `<tool_calls>` envelope is shared with `.jamba`
/// and `.hunyuanA13B`; route by model name / type.
struct MiniMaxParser: ResponseFormatParser {
  private static let startMarker = "<tool_calls>"
  private static let endMarker = "</tool_calls>"
  private static let thinkStart = "<think>"
  private static let thinkEnd = "</think>"

  private var buffer: String = ""
  private var openMessage: OpenMessage?
  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  init() {}

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return scan(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = scan(isEnd: true)
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

      // Find the first `<tool_calls>` that is not inside a closed
      // `<think>...</think>` block. Closed think blocks are inert –
      // their contents (including any inner `<tool_calls>`) flush as
      // message content. An open `<think>` (no closer yet) holds the
      // buffer so we don't prematurely classify an inner envelope as
      // a real tool call.
      let inOpenThink = startsInsideOpenThink(buffer)

      // If we're inside an unclosed `<think>` block, hold all bytes
      // until `</think>` arrives (or EOS).
      if inOpenThink {
        if isEnd {
          events.append(contentsOf: emitMessageDelta(text: buffer))
          buffer = ""
          didProgress = true
        }
        return events
      }

      guard let startRange = nextRealEnvelopeStart() else {
        // No envelope visible. Emit safe content, holding back any
        // partial-marker overlap (envelope or `<think>`).
        let partial = trailingPartialMarkerOverlap(of: buffer)
        let safeEnd = isEnd ? buffer.count : buffer.count - partial
        if safeEnd > 0 {
          var safeText = String(buffer.prefix(safeEnd))
          if safeText.contains(Self.endMarker) {
            safeText = safeText.replacingOccurrences(of: Self.endMarker, with: "")
          }
          if !safeText.isEmpty {
            events.append(contentsOf: emitMessageDelta(text: safeText))
          }
          buffer.removeFirst(safeEnd)
          didProgress = safeEnd > 0
        }
        return events
      }

      // Emit pre-envelope text as content.
      let preText = String(buffer[buffer.startIndex ..< startRange.lowerBound])
      if !preText.isEmpty {
        events.append(contentsOf: emitMessageDelta(text: preText))
        buffer.removeFirst(preText.count)
        didProgress = true
        continue
      }

      // Buffer starts with the envelope start. Look for the close.
      let bodyStart = buffer.index(buffer.startIndex, offsetBy: Self.startMarker.count)
      guard let endRange = buffer.range(of: Self.endMarker, range: bodyStart ..< buffer.endIndex) else {
        if isEnd {
          // vLLM's regex also accepts a final `<tool_calls>...`
          // region without `</tool_calls>`. Parse complete NDJSON
          // lines when possible, otherwise surface the partial
          // envelope as content.
          let body = String(buffer[bodyStart...])
          let calls = parseNDJSONCalls(body)
          if !calls.isEmpty {
            if openMessage != nil {
              events.append(contentsOf: closeMessage(status: .completed))
            }
            for call in calls {
              events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
            }
          } else {
            let stray = String(buffer)
            events.append(contentsOf: emitMessageDelta(text: stray))
          }
          buffer = ""
          didProgress = true
        }
        return events
      }

      let body = String(buffer[bodyStart ..< endRange.lowerBound])
      let consumed = buffer.distance(from: buffer.startIndex, to: endRange.upperBound)

      let calls = parseNDJSONCalls(body)
      if !calls.isEmpty {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        for call in calls {
          events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
        }
      } else {
        // No parseable NDJSON entries – surface the entire envelope as
        // content.
        let envelope = String(buffer.prefix(consumed))
        events.append(contentsOf: emitMessageDelta(text: envelope))
      }
      buffer.removeFirst(consumed)
      didProgress = true
    }
    return events
  }

  /// True iff the buffer contains a `<think>` opener with no matching
  /// `</think>` after it (and no `<tool_calls>` already past the
  /// pending think). Used to hold the buffer until the think block
  /// closes, so an envelope inside reasoning isn't treated as a real
  /// tool call.
  private func startsInsideOpenThink(_ s: String) -> Bool {
    guard let lastOpen = s.range(of: Self.thinkStart, options: .backwards) else {
      return false
    }
    let after = s[lastOpen.upperBound...]
    return after.range(of: Self.thinkEnd) == nil
  }

  /// Find the first `<tool_calls>` opener that does NOT lie inside a
  /// closed `<think>...</think>` region. Returns nil when no such
  /// opener exists in the current buffer.
  private func nextRealEnvelopeStart() -> Range<String.Index>? {
    var searchFrom = buffer.startIndex
    while let candidate = buffer.range(of: Self.startMarker, range: searchFrom ..< buffer.endIndex) {
      if !isPositionInsideClosedThink(candidate.lowerBound) {
        return candidate
      }
      searchFrom = candidate.upperBound
    }
    return nil
  }

  /// True when `position` lies inside a closed `<think>...</think>`
  /// region in the buffer.
  private func isPositionInsideClosedThink(_ position: String.Index) -> Bool {
    var cursor = buffer.startIndex
    while let openRange = buffer.range(of: Self.thinkStart, range: cursor ..< buffer.endIndex) {
      guard let closeRange = buffer.range(of: Self.thinkEnd, range: openRange.upperBound ..< buffer.endIndex) else {
        return false
      }
      if openRange.lowerBound <= position, position < closeRange.upperBound {
        return true
      }
      cursor = closeRange.upperBound
    }
    return false
  }

  private func trailingPartialMarkerOverlap(of text: String) -> Int {
    let chars = Array(text)
    let startOverlap = partialOverlap(suffixOf: chars, with: Array(Self.startMarker))
    let endOverlap = partialOverlap(suffixOf: chars, with: Array(Self.endMarker))
    let thinkOpenOverlap = partialOverlap(suffixOf: chars, with: Array(Self.thinkStart))
    let thinkCloseOverlap = partialOverlap(suffixOf: chars, with: Array(Self.thinkEnd))
    return Swift.max(
      Swift.max(startOverlap, endOverlap),
      Swift.max(thinkOpenOverlap, thinkCloseOverlap),
    )
  }

  // MARK: NDJSON parsing

  private struct ParsedCall {
    let name: String
    let arguments: String
  }

  /// Parse the envelope body as NDJSON: one JSON object per non-empty
  /// line. Lines that don't decode (or that lack `name`) are skipped,
  /// matching vLLM's per-line `try / except` loop.
  private func parseNDJSONCalls(_ body: String) -> [ParsedCall] {
    var calls: [ParsedCall] = []
    let lines = body.split(omittingEmptySubsequences: false, whereSeparator: { $0 == "\n" })
    for raw in lines {
      let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      if line.isEmpty { continue }
      guard line.hasPrefix("{"), line.hasSuffix("}") else { continue }
      guard let data = line.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let name = obj["name"] as? String,
            !name.isEmpty
      else { continue }
      guard let argsValue = obj["arguments"] ?? obj["parameters"] else {
        continue
      }
      guard let argsJSON = serializeJSONArgument(argsValue) else {
        continue
      }
      calls.append(ParsedCall(name: name, arguments: argsJSON))
    }
    return calls
  }

  /// vLLM applies `json.dumps(function_call["arguments"])` to whatever
  /// JSON value the line supplied. That is usually an object, but arrays,
  /// strings, numbers, booleans, and null are all serialized as JSON
  /// fragments rather than rewritten to `{}`.
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
      itemId: id, outputIndex: outputIndex, arguments: arguments, sequenceNumber: takeSequence(),
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
