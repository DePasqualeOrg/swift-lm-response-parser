// Copyright © Anthony DePasquale

import Foundation

/// Parser for AI21's Jamba 1.5 / 1.7 tool-call format.
///
/// **Wire shape.** Tool calls share a single envelope `<tool_calls>` …
/// `</tool_calls>` containing a JSON array of `{name, arguments}` objects:
///
/// ```text
/// <tool_calls>[{"name": "fn", "arguments": {"x": 1}}, ...]</tool_calls>
/// ```
///
/// The envelope tokens are registered as special tokens in Jamba's
/// tokenizer vocabulary; after detokenization they surface as literal
/// text and the parser matches on the strings.
///
/// **Streaming.** The parser holds the array bytes until the closing
/// `</tool_calls>` arrives, then emits all calls at once. Per-call
/// argument-delta streaming via partial-JSON is not implemented (mirrors
/// `Phi4MiniParser`'s trade-off).
///
/// **Naming caveat.** The `<tool_calls>` envelope is also used by vLLM's
/// `hunyuan_a13b` (with a JSON array body, similar to Jamba) and by
/// `minimax` (with NDJSON body — distinct shape). When porting those,
/// the envelope itself is not enough to disambiguate; route by model
/// name / type instead.
struct JambaParser: ResponseFormatParser {
  private static let startMarker = "<tool_calls>"
  private static let endMarker = "</tool_calls>"

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

      guard let startRange = buffer.range(of: Self.startMarker) else {
        // No envelope visible. Emit safe content; hold back any
        // partial-marker suffix.
        let partial = trailingPartialMarkerOverlap(of: buffer)
        let safeEnd = isEnd ? buffer.count : buffer.count - partial
        if safeEnd > 0 {
          var safeText = String(buffer.prefix(safeEnd))
          // Strip a stray bare `</tool_calls>` literal so it doesn't
          // leak into message content. Mirrors sglang's
          // `_clean_normal_text`.
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
          // Truncated envelope at EOS — surface as content.
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

      if let calls = parseToolCallArray(body) {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        for call in calls {
          events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
        }
      } else {
        // Body wasn't parseable JSON — surface the entire envelope
        // as content. Matches vLLM's `extract_tool_calls` failure
        // path (returns the raw model_output as `content`).
        let envelope = String(buffer.prefix(consumed))
        events.append(contentsOf: emitMessageDelta(text: envelope))
      }
      buffer.removeFirst(consumed)
      didProgress = true
    }
    return events
  }

  private func trailingPartialMarkerOverlap(of text: String) -> Int {
    let chars = Array(text)
    let startOverlap = partialOverlap(suffixOf: chars, with: Array(Self.startMarker))
    let endOverlap = partialOverlap(suffixOf: chars, with: Array(Self.endMarker))
    return Swift.max(startOverlap, endOverlap)
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
      // vLLM accepts only `arguments`; we also accept `parameters`
      // as an alias for symmetry with other parsers in this package.
      let argsValue = dict["arguments"] ?? dict["parameters"] ?? [String: Any]()
      guard let argsJSON = serializeJSONArgument(argsValue) else {
        return nil
      }
      calls.append(ParsedCall(name: name, arguments: argsJSON))
    }
    return calls
  }

  /// vLLM applies `json.dumps(function_call["arguments"])` to the raw
  /// `arguments` value. Objects are the normal shape, but malformed or
  /// unusual outputs with arrays, strings, numbers, booleans, or `null`
  /// should remain visible as JSON fragments.
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
