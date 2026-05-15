// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Llama 3 / 3.1 / 3.2 inline-JSON tool-call format.
///
/// **Wire shape.** Tool calls are JSON objects, optionally prefixed by the
/// reserved `<|python_tag|>` marker:
///
/// ```text
/// <|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}
/// ```
///
/// Some templates produce parallel calls separated by `;`:
///
/// ```text
/// <|python_tag|>{"name": "f1", "arguments": {}}; {"name": "f2", "arguments": {"x": 1}}
/// ```
///
/// When the prompt opts out of the python tag, the model may emit the JSON
/// directly without the marker; the parser handles both. Plain text
/// before the marker (or that doesn't parse as JSON) is forwarded as
/// message content.
///
/// **Llama 2 vs Llama 3.** Llama 2 shares `model_type=llama` but has a
/// 32k vocab and uses a different format. ``ResponseFormat/resolveByType``
/// gates `.llama3` on `vocab_size >= 128000` so Llama 2 falls through to
/// the JSON fallback.
struct Llama3Parser: ResponseFormatParser {
  private static let pythonTag = "<|python_tag|>"
  private static let pythonTagChars = Array(pythonTag)

  /// Active suffix that has not yet been proven safe to discard.
  private var buffer: String = ""
  private var parsedIdx: Int = 0
  /// Once we've consumed the python-tag marker (or decided the buffer
  /// starts with a bare JSON object), the parser commits to "JSON
  /// streaming mode" and never re-emits content. nil means we're still
  /// in plain-text mode.
  private var jsonModeStartIdx: Int?

  private var openMessage: OpenMessage?
  private var emittedCallCount: Int = 0

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
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    return events
  }

  // MARK: Scan

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    defer { pruneConsumedPrefix() }

    // Detect entry into JSON mode if not already there.
    if jsonModeStartIdx == nil {
      // Mode 1: the buffer contains the python_tag marker.
      if let tagIdx = buffer.range(of: Llama3Parser.pythonTag) {
        let preTagEnd = buffer.distance(from: buffer.startIndex, to: tagIdx.lowerBound)
        events.append(contentsOf: emitContent(upTo: preTagEnd))
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        let jsonStart = buffer.distance(from: buffer.startIndex, to: tagIdx.upperBound)
        jsonModeStartIdx = jsonStart
        parsedIdx = jsonStart
      }
      // Mode 2: no marker, but the first non-whitespace char is `{`
      // – treat the whole stream as a tool call. We can only commit
      // to this when we're confident no python_tag marker will arrive
      // (i.e., at end of stream, or after a non-`{` non-`<` char
      // settles the question).
      else if let firstNonWS = firstNonWhitespaceCharacter() {
        if firstNonWS.character == "{" {
          if isEnd || canRuleOutPythonTagPrefix() {
            jsonModeStartIdx = firstNonWS.index
            parsedIdx = firstNonWS.index
          }
        } else if firstNonWS.character != "<" {
          // Definitely not a tool call. Stay in content mode.
          // Fall through to emit-content below.
        }
      }
    }

    if let _ = jsonModeStartIdx {
      events.append(contentsOf: scanJSONCalls(isEnd: isEnd))
    } else {
      // Still in plain-text mode. Emit safely (hold back partial
      // python_tag prefix so we don't accidentally emit the marker
      // text as content).
      let safeEnd = safeContentEnd(isEnd: isEnd)
      if safeEnd > parsedIdx {
        events.append(contentsOf: emitContent(upTo: safeEnd))
      }
    }

    return events
  }

  private mutating func pruneConsumedPrefix() {
    guard parsedIdx > 0 else { return }

    let dropCount = min(parsedIdx, buffer.count)
    buffer.removeFirst(dropCount)
    parsedIdx = 0

    if let start = jsonModeStartIdx {
      jsonModeStartIdx = max(0, start - dropCount)
    }
  }

  private func firstNonWhitespaceCharacter() -> (character: Character, index: Int)? {
    for (offset, ch) in buffer.enumerated() {
      if !ch.isWhitespace {
        return (ch, offset)
      }
    }
    return nil
  }

  /// True when the current buffer can be ruled out as a leading
  /// `<|python_tag|>` marker – i.e., its leading non-whitespace bytes
  /// don't form a prefix of the marker.
  private func canRuleOutPythonTagPrefix() -> Bool {
    // If buffer doesn't start with `<` after whitespace, it can't be
    // the marker. If it does, hold to wait for more.
    let trimmed = buffer.drop(while: { $0.isWhitespace })
    return !trimmed.hasPrefix("<") || !Llama3Parser.pythonTag.hasPrefix(String(trimmed.prefix(Llama3Parser.pythonTag.count)))
  }

  private func safeContentEnd(isEnd: Bool) -> Int {
    let bufChars = Array(buffer)
    if isEnd { return bufChars.count }
    let overlap = partialOverlap(suffixOf: bufChars, with: Array(Llama3Parser.pythonTag))
    return bufChars.count - overlap
  }

  private mutating func emitContent(upTo end: Int) -> [ResponseStreamingEvent] {
    guard end > parsedIdx else { return [] }
    let bufChars = Array(buffer)
    let chunk = String(bufChars[parsedIdx ..< end])
    parsedIdx = end
    if chunk.isEmpty { return [] }

    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem())
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
    return events
  }

  // MARK: JSON streaming

  /// Walk the buffer for `{...}` tool-call objects, decoding each one
  /// fully before emitting events. The wire format embeds args inside a
  /// JSON object with no explicit `arguments` delimiter, so emitting
  /// arg deltas mid-object would require partial-JSON parsing to know
  /// which bytes are part of the args value. Buffering until the call
  /// balances trades per-token latency on large arg payloads for
  /// implementation simplicity. The cumulative event sequence and final
  /// content are identical either way. Diverges from sglang and vLLM,
  /// both of which stream args incrementally via `partial_json_parser`.
  private mutating func scanJSONCalls(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)

    while parsedIdx < bufChars.count {
      // Skip leading whitespace, `;` separators, and repeated
      // `<|python_tag|>` markers between JSON objects. SGLang accepts
      // both `;` and repeated tag separators for Llama 3.2 parallel
      // calls.
      var skippedSeparator = true
      while skippedSeparator, parsedIdx < bufChars.count {
        skippedSeparator = false
        while parsedIdx < bufChars.count, bufChars[parsedIdx].isWhitespace || bufChars[parsedIdx] == ";" {
          parsedIdx += 1
          skippedSeparator = true
        }
        if matchesPythonTag(in: bufChars, at: parsedIdx) {
          parsedIdx += Self.pythonTagChars.count
          skippedSeparator = true
        }
      }
      if parsedIdx >= bufChars.count { break }
      guard bufChars[parsedIdx] == "{" else {
        if isEnd {
          events.append(contentsOf: emitContent(upTo: bufChars.count))
        }
        break
      }

      guard let (jsonText, endIdx, complete) = extractJSONObject(in: bufChars, from: parsedIdx) else {
        break
      }
      if !complete, !isEnd {
        break
      }
      if let object = decodeCallObject(jsonText) {
        events.append(contentsOf: emitParsedCall(object, status: complete ? .completed : .incomplete))
        parsedIdx = endIdx
        if !complete { break }
        continue
      }
      // Strict JSON failed. Try the Python-dict fallback (single
      // quotes, True/False/None) before giving up on this object.
      if let dictJSON = PythonLiteral.parseTopLevelDict(jsonText),
         let object = decodeCallObject(dictJSON)
      {
        events.append(contentsOf: emitParsedCall(object, status: complete ? .completed : .incomplete))
        parsedIdx = endIdx
        if !complete { break }
        continue
      }
      // Both decoders failed. Recover by skipping to the next
      // `{"name":` candidate so that one malformed call doesn't
      // suppress the rest.
      if let nextStart = findNextNameObjectStart(in: bufChars, after: parsedIdx) {
        parsedIdx = nextStart
        continue
      }
      // No further candidates – surface the non-tool JSON/text rather
      // than silently dropping model output.
      if isEnd || complete {
        events.append(contentsOf: emitContent(upTo: bufChars.count))
      }
      break
    }

    return events
  }

  private func matchesPythonTag(in chars: [Character], at index: Int) -> Bool {
    guard index >= 0, index + Self.pythonTagChars.count <= chars.count else {
      return false
    }
    for offset in 0 ..< Self.pythonTagChars.count where chars[index + offset] != Self.pythonTagChars[offset] {
      return false
    }
    return true
  }

  private func decodeCallObject(_ jsonText: String) -> [String: Any]? {
    guard let data = jsonText.data(using: .utf8),
          let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          object["name"] is String else { return nil }
    return object
  }

  private mutating func emitParsedCall(_ object: [String: Any], status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let name = object["name"] as? String else { return [] }
    let argsValue = object["arguments"] ?? object["parameters"] ?? [:]
    let argsText: String = if let s = argsValue as? String {
      s
    } else if let data = try? JSONSerialization.data(
      // `.sortedKeys` for deterministic key order; Foundation
      // dictionaries don't preserve insertion order. Diverges from
      // sglang/vLLM, which emit in declaration order via Python
      // dicts' insertion-order guarantee.
      withJSONObject: argsValue, options: [.sortedKeys, .withoutEscapingSlashes],
    ), let s = String(data: data, encoding: .utf8) {
      s
    } else {
      "{}"
    }
    emittedCallCount += 1
    return emitToolCall(name: name, arguments: argsText, status: status)
  }

  /// Scan forward from `start + 1` for the next plausible call object,
  /// matching `{` followed by an optional whitespace, then `"name"`. This
  /// lets us skip a malformed object without dropping subsequent valid
  /// calls in the same buffer.
  private func findNextNameObjectStart(in chars: [Character], after start: Int) -> Int? {
    let needle: [Character] = ["\"", "n", "a", "m", "e", "\""]
    var i = start + 1
    while i < chars.count {
      guard chars[i] == "{" else { i += 1; continue }
      var j = i + 1
      while j < chars.count, chars[j].isWhitespace {
        j += 1
      }
      if j + needle.count <= chars.count {
        var matches = true
        for k in 0 ..< needle.count where chars[j + k] != needle[k] {
          matches = false; break
        }
        if matches { return i }
      }
      i += 1
    }
    return nil
  }

  private func extractJSONObject(in chars: [Character], from start: Int) -> (text: String, end: Int, complete: Bool)? {
    guard start < chars.count, chars[start] == "{" else { return nil }
    var depth = 0
    var inString = false
    var escape = false
    for i in start ..< chars.count {
      let c = chars[i]
      if inString {
        if escape { escape = false; continue }
        if c == "\\" { escape = true; continue }
        if c == "\"" { inString = false }
        continue
      }
      if c == "\"" { inString = true; continue }
      if c == "{" { depth += 1 }
      else if c == "}" {
        depth -= 1
        if depth == 0 {
          return (String(chars[start ..< i + 1]), i + 1, true)
        }
      }
    }
    return (String(chars[start ..< chars.count]), chars.count, false)
  }

  // MARK: Tool call emission

  private mutating func emitToolCall(name: String, arguments: String, status: ItemStatus) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    let id = IDFactory.make(.functionCall)
    let callId = IDFactory.make(.callId)
    let outputIndex = takeOutputIndex()
    let openItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: name, arguments: "", status: .inProgress,
    )
    let doneItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: name, arguments: arguments, status: status,
    )
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

  // MARK: Open/close

  private mutating func openMessageItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.message)
    let outputIndex = takeOutputIndex()
    openMessage = OpenMessage(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .message(.init(id: id, content: [], status: .inProgress)),
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

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
