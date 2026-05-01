// Copyright © Anthony DePasquale

import Foundation

/// Parser for the legacy Gemma function-call format (Gemma 1 / Gemma 2)
/// and Google's `functiongemma` model family. Both share the same wire
/// shape and routing differs only in inference (`model_type == "gemma"`
/// for Gemma 1/2, HF name prefix `functiongemma` for the standalone
/// model).
///
/// **Wire shape.** A tool call is wrapped in literal markers and uses
/// `call:NAME` to introduce the function name, then a brace-delimited
/// list of `key:value` pairs:
///
/// ```text
/// <start_function_call>call:get_weather{location:<escape>Paris<escape>,unit:<escape>celsius<escape>}<end_function_call>
/// ```
///
/// String values are wrapped in literal `<escape>` markers (treated as
/// text, not tokens). Non-string values appear bare and are interpreted
/// as JSON literals where possible (numbers, booleans, `null`) – the
/// parser falls back to a string when JSON-decoding fails. Multiple
/// tool calls are concatenated without a separator.
///
/// **No reasoning channel.** This format does not interleave reasoning;
/// any text outside the markers is plain message content.
struct GemmaFunctionCallParser: ResponseFormatParser {
  private static let callStart = "<start_function_call>"
  private static let callEnd = "<end_function_call>"
  private static let escapeMarker = "<escape>"
  private static let callPrefix = "call:"

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

      guard let startRange = buffer.range(of: Self.callStart) else {
        // No envelope opener. Emit safe content; hold back any
        // partial-opener suffix.
        let partial = trailingPartialOpenerOverlap(of: buffer)
        let safeEnd = isEnd ? buffer.count : buffer.count - partial
        if safeEnd > 0 {
          let safeText = String(buffer.prefix(safeEnd))
          events.append(contentsOf: emitMessageDelta(text: safeText))
          buffer.removeFirst(safeEnd)
          didProgress = safeEnd > 0
        }
        return events
      }

      // Pre-envelope text emits as message content.
      let preText = String(buffer[buffer.startIndex ..< startRange.lowerBound])
      if !preText.isEmpty {
        events.append(contentsOf: emitMessageDelta(text: preText))
        buffer.removeFirst(preText.count)
        didProgress = true
        continue
      }

      // Buffer starts with `<start_function_call>`. Look for the
      // matching `<end_function_call>`.
      guard let endRange = buffer.range(of: Self.callEnd, range: startRange.upperBound ..< buffer.endIndex) else {
        // Envelope not closed yet. At end-of-stream, recover a
        // parseable partial call as incomplete; vLLM's non-streaming
        // FunctionGemma parser also extracts this shape via its
        // unclosed-envelope regex alternative.
        if isEnd {
          let bodyText = String(buffer[startRange.upperBound ..< buffer.endIndex])
          if let call = parseCallBody(bodyText, allowUnclosedBrace: true) {
            if openMessage != nil {
              events.append(contentsOf: closeMessage(status: .completed))
            }
            events.append(contentsOf: emitToolCall(
              name: call.name,
              arguments: call.arguments,
              status: .incomplete,
            ))
          } else {
            events.append(contentsOf: emitMessageDelta(text: buffer))
          }
          buffer = ""
          didProgress = true
        }
        return events
      }

      // Extract the body between `<start_function_call>` and
      // `<end_function_call>`. Body shape: `call:NAME{...}`.
      let bodyText = String(buffer[startRange.upperBound ..< endRange.lowerBound])
      let envelopeUpper = endRange.upperBound
      let consumed = buffer.distance(from: buffer.startIndex, to: envelopeUpper)

      if let call = parseCallBody(bodyText) {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
      } else {
        // Body didn't parse – surface the entire envelope as
        // content rather than dropping it silently.
        let envelopeText = String(buffer.prefix(consumed))
        events.append(contentsOf: emitMessageDelta(text: envelopeText))
      }
      buffer.removeFirst(consumed)
      didProgress = true
    }
    return events
  }

  // MARK: Body parsing

  private struct ParsedCall {
    let name: String
    let arguments: String
  }

  /// Parse a `call:NAME{key:value,key:<escape>str<escape>}` body.
  /// Returns nil if the shape is malformed (no `call:` prefix, missing
  /// braces, unparseable name).
  private func parseCallBody(_ body: String, allowUnclosedBrace: Bool = false) -> ParsedCall? {
    let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix(Self.callPrefix) else { return nil }
    let afterPrefix = trimmed.dropFirst(Self.callPrefix.count)

    guard let braceStart = afterPrefix.firstIndex(of: "{") else {
      return nil
    }

    let name = String(afterPrefix[afterPrefix.startIndex ..< braceStart])
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard isValidIdentifier(name) else { return nil }

    let argsStart = afterPrefix.index(after: braceStart)
    let argsEnd: Substring.Index
    if let braceEnd = afterPrefix.lastIndex(of: "}"), braceStart < braceEnd {
      argsEnd = braceEnd
    } else if allowUnclosedBrace {
      argsEnd = afterPrefix.endIndex
    } else {
      return nil
    }

    let argsRegion = String(afterPrefix[argsStart ..< argsEnd])
    let args = parseArgumentList(argsRegion)
    guard let argsJSON = encodeArgsAsJSON(args) else { return nil }
    return ParsedCall(name: name, arguments: argsJSON)
  }

  private func isValidIdentifier(_ name: String) -> Bool {
    guard let first = name.first, first.isLetter || first == "_" else { return false }
    return name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
  }

  /// Walk the `key:value,key:value` list. Returns key-value pairs as
  /// `(key, JSON-encoded value)` tuples, preserving key order.
  private func parseArgumentList(_ region: String) -> [(key: String, valueJSON: String)] {
    var pairs: [(String, String)] = []
    var i = region.startIndex
    while i < region.endIndex {
      // Skip leading whitespace and commas between pairs.
      while i < region.endIndex, region[i].isWhitespace || region[i] == "," {
        i = region.index(after: i)
      }
      if i >= region.endIndex { break }

      // Read the key (identifier characters until `:`).
      let keyStart = i
      while i < region.endIndex,
            region[i].isLetter || region[i].isNumber || region[i] == "_"
      {
        i = region.index(after: i)
      }
      let key = String(region[keyStart ..< i])
      guard !key.isEmpty else { return pairs }

      while i < region.endIndex, region[i].isWhitespace {
        i = region.index(after: i)
      }
      guard i < region.endIndex, region[i] == ":" else { return pairs }
      i = region.index(after: i)
      while i < region.endIndex, region[i].isWhitespace {
        i = region.index(after: i)
      }

      // Read the value. `<escape>...<escape>` for strings, bare
      // tokens otherwise.
      let valueJSON: String
      let escape = Self.escapeMarker
      if region[i...].hasPrefix(escape) {
        let afterOpen = region.index(i, offsetBy: escape.count)
        if let closeRange = region.range(of: escape, range: afterOpen ..< region.endIndex) {
          let body = String(region[afterOpen ..< closeRange.lowerBound])
          valueJSON = encodeEscapedValue(body)
          i = closeRange.upperBound
        } else {
          return pairs // unmatched escape opener – bail out
        }
      } else {
        let bareStart = i
        while i < region.endIndex, region[i] != "," {
          i = region.index(after: i)
        }
        let bare = String(region[bareStart ..< i]).trimmingCharacters(in: .whitespacesAndNewlines)
        valueJSON = encodeBareValue(bare)
      }

      pairs.append((key, valueJSON))
    }
    return pairs
  }

  /// Try to parse a bare token as a JSON literal (number, boolean,
  /// `null`). Falls back to a JSON-encoded string on failure. Mirrors
  /// vLLM's `FunctionGemmaToolParser`, which JSON-decodes bare values
  /// permissively and keeps them as raw strings on failure.
  private func encodeBareValue(_ token: String) -> String {
    encodeJSONLiteralOrString(token)
  }

  /// vLLM JSON-decodes the contents of `<escape>...</escape>` as well
  /// as bare values. That means `<escape>42<escape>` is the number 42,
  /// while `<escape>hello<escape>` remains the string `"hello"`.
  private func encodeEscapedValue(_ token: String) -> String {
    encodeJSONLiteralOrString(token)
  }

  private func encodeJSONLiteralOrString(_ token: String) -> String {
    if token.isEmpty { return jsonString("") }
    if let data = token.data(using: .utf8),
       let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    {
      // Re-encode to ensure canonical JSON form (e.g., `True` → fail,
      // `true` → `true`, `42` → `42`).
      if let encoded = try? JSONSerialization.data(
        withJSONObject: parsed,
        options: [.fragmentsAllowed],
      ),
        let s = String(data: encoded, encoding: .utf8)
      {
        return s
      }
    }
    return jsonString(token)
  }

  private func encodeArgsAsJSON(_ pairs: [(key: String, valueJSON: String)]) -> String? {
    var out = "{"
    for (idx, pair) in pairs.enumerated() {
      if idx > 0 { out += ", " }
      out += jsonString(pair.key) + ": " + pair.valueJSON
    }
    out += "}"
    return out
  }

  private func jsonString(_ s: String) -> String {
    if let data = try? JSONSerialization.data(withJSONObject: [s], options: []),
       let encoded = String(data: data, encoding: .utf8),
       encoded.count >= 2
    {
      return String(encoded.dropFirst().dropLast())
    }
    var out = "\""
    for ch in s {
      switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out.append(ch)
      }
    }
    out += "\""
    return out
  }

  // MARK: Helpers

  private func trailingPartialOpenerOverlap(of text: String) -> Int {
    let chars = Array(text)
    return partialOverlap(suffixOf: chars, with: Array(Self.callStart))
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

  private mutating func emitToolCall(
    name: String,
    arguments: String,
    status: ItemStatus = .completed,
  ) -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.functionCall)
    let callId = IDFactory.make(.callId)
    let outputIndex = takeOutputIndex()
    let openItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: name, arguments: "", status: .inProgress,
    )
    let doneItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: name, arguments: arguments, status: status,
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
