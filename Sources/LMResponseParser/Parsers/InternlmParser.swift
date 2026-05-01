// Copyright © Anthony DePasquale

import Foundation

/// Parser for Shanghai AI Lab's InternLM 2.x and Intern-S1 tool-call format.
///
/// **Wire shape.** Tool calls are delimited by reserved tokens:
///
/// ```text
/// <|action_start|><|plugin|>{"name": "...", "parameters": {...}}<|action_end|>
/// ```
///
/// The body is a JSON object with a `name` field and either `parameters`
/// (canonical for InternLM) or `arguments`. Multiple calls may appear in
/// sequence, each with its own `<|action_start|><|plugin|>` …
/// `<|action_end|>` envelope. Plain text outside the envelopes is normal
/// message content.
///
/// **Marker variants.** Both `<|action_start|><|plugin|>` (vLLM's reference
/// fixture) and `<|action_start|> <|plugin|>` (sglang's `InternlmDetector`)
/// are accepted; the upstreams differ by a single space, and observed
/// outputs vary.
///
/// **Streaming.** The parser holds the inner JSON until the matching
/// `<|action_end|>` arrives, then emits the call. Per-call argument-delta
/// streaming via partial-JSON is not implemented (mirroring
/// `Phi4MiniParser`'s trade-off).
struct InternlmParser: ResponseFormatParser {
  private static let startMarkers: [String] = [
    "<|action_start|><|plugin|>",
    "<|action_start|> <|plugin|>",
  ]
  private static let endMarker = "<|action_end|>"

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

      // Find the earliest start marker (across both variants).
      var earliestStart: (range: Range<String.Index>, marker: String)? = nil
      for marker in Self.startMarkers {
        if let r = buffer.range(of: marker) {
          if earliestStart == nil || r.lowerBound < earliestStart!.range.lowerBound {
            earliestStart = (r, marker)
          }
        }
      }

      guard let start = earliestStart else {
        // No envelope visible. Emit safe content; hold back any
        // partial-marker suffix.
        let partial = trailingPartialStartOverlap(of: buffer)
        let safeEnd = isEnd ? buffer.count : buffer.count - partial
        if safeEnd > 0 {
          let safeText = String(buffer.prefix(safeEnd))
          // Strip any stray `<|action_end|>` literal so a bare close
          // tag doesn't leak into message content. Mirrors sglang's
          // `_clean_normal_text`.
          var cleaned = safeText
          if cleaned.contains(Self.endMarker) {
            cleaned = cleaned.replacingOccurrences(of: Self.endMarker, with: "")
          }
          if !cleaned.isEmpty {
            events.append(contentsOf: emitMessageDelta(text: cleaned))
          }
          buffer.removeFirst(safeEnd)
          didProgress = safeEnd > 0
        }
        return events
      }

      // Emit pre-envelope text as content.
      let preText = String(buffer[buffer.startIndex ..< start.range.lowerBound])
      if !preText.isEmpty {
        events.append(contentsOf: emitMessageDelta(text: preText))
        buffer.removeFirst(preText.count)
        didProgress = true
        continue
      }

      // Buffer starts with the start marker. Look for the end marker.
      let bodyStart = buffer.index(buffer.startIndex, offsetBy: start.marker.count)
      guard let endRange = buffer.range(of: Self.endMarker, range: bodyStart ..< buffer.endIndex) else {
        // End marker not yet present. At end-of-stream, surface the
        // truncated envelope as content.
        if isEnd {
          let stray = String(buffer)
          events.append(contentsOf: emitMessageDelta(text: stray))
          buffer = ""
          didProgress = true
        }
        return events
      }

      let body = String(buffer[bodyStart ..< endRange.lowerBound])
        .trimmingCharacters(in: .whitespacesAndNewlines)
      let consumed = buffer.distance(from: buffer.startIndex, to: endRange.upperBound)

      if let call = parseToolCallObject(body) {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
      } else {
        // Body wasn't parseable JSON — surface the entire envelope as
        // content. This matches sglang's `detect_and_parse` parse-error
        // path (returns the raw text as `normal_text`).
        let envelope = String(buffer.prefix(consumed))
        events.append(contentsOf: emitMessageDelta(text: envelope))
      }
      buffer.removeFirst(consumed)
      didProgress = true
    }
    return events
  }

  private func trailingPartialStartOverlap(of text: String) -> Int {
    let chars = Array(text)
    var maxOverlap = 0
    for marker in Self.startMarkers {
      let o = partialOverlap(suffixOf: chars, with: Array(marker))
      maxOverlap = Swift.max(maxOverlap, o)
    }
    // Also hold back partial end markers so a stray `<|action_e…`
    // suffix doesn't leak as content while the close is being written.
    let endOverlap = partialOverlap(suffixOf: chars, with: Array(Self.endMarker))
    return Swift.max(maxOverlap, endOverlap)
  }

  // MARK: JSON parsing

  private struct ParsedCall {
    let name: String
    let arguments: String
  }

  private func parseToolCallObject(_ jsonText: String) -> ParsedCall? {
    // The body might have leading garbage before `{` (vLLM's reference
    // does `action[action.find("{"):]`). Find the first `{` and parse
    // from there.
    guard let braceIdx = jsonText.firstIndex(of: "{") else { return nil }
    let trimmed = String(jsonText[braceIdx...])
    guard let data = trimmed.data(using: .utf8),
          let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let name = dict["name"] as? String
    else {
      return nil
    }
    let argsValue = dict["parameters"] ?? dict["arguments"] ?? [String: Any]()
    guard let argsJSON = serializeArguments(argsValue) else {
      return nil
    }
    return ParsedCall(name: name, arguments: argsJSON)
  }

  private func serializeArguments(_ value: Any) -> String? {
    guard let data = try? JSONSerialization.data(
      withJSONObject: value,
      // `.sortedKeys` enforces deterministic key order; Foundation
      // dictionaries don't preserve insertion order. Diverges from
      // sglang/vLLM, which emit in declaration order via Python
      // dicts' insertion-order guarantee. `.fragmentsAllowed` mirrors
      // Python's `json.dumps`, which also serializes strings/numbers
      // when malformed output uses a non-object `parameters` value.
      options: [.sortedKeys, .fragmentsAllowed],
    ),
      let s = String(data: data, encoding: .utf8)
    else {
      return nil
    }
    return s
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
