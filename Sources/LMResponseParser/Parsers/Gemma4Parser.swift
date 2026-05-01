// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Gemma 4 reasoning + tool-call format.
///
/// **Reasoning markers.** The thinking marker is a multi-token pattern: a
/// special token `<|channel>` followed by the regular-text run `thought`,
/// terminated by a single special token `<channel|>`. Both pipes are inside
/// the *opening* token (`<|channel>`) and outside the *closing* token
/// (`<channel|>`); confusing this with Harmony's `<|channel|>` (single
/// reserved token, both pipes inside) routes to the wrong parser.
///
/// **Tool-call wire shape.** Tool calls use a non-JSON syntax:
///
/// ```text
/// <|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>celsius<|"|>,limit:5}<tool_call|>
/// ```
///
/// Keys are unquoted; string values are delimited by `<|"|>`; numbers,
/// booleans (`true` / `false`), and `null` (also `none` / `nil`) are bare;
/// nested objects and arrays follow the same syntax recursively. The parser
/// converts each tool call's arguments into the canonical JSON the spec
/// expects on the wire.
///
/// **Streaming behavior.** Reasoning text streams as deltas during the
/// `<|channel>thought…<channel|>` block. The function name is emitted as
/// soon as `call:NAME{` is parsed; the converted JSON arguments are emitted
/// as a single delta when the matching `}` (and trailing `<tool_call|>`)
/// arrives. Truncation closes any open item with `incomplete` status.
///
/// **Marker matching: text-based.** Both pieces of the start marker
/// (`<|channel>` followed by `thought`) and the end marker
/// (`<channel|>`) decode to canonical literal strings, so the parser
/// scans the detokenized text for these markers. SGLang's
/// `Gemma4Detector` in `reasoning_parser.py` takes the same approach.
/// vLLM's `gemma4_reasoning_parser.py` keys off token IDs instead
/// (looking up `<|channel>`, `<channel|>`, `<|tool_call>`, etc. via
/// `self.vocab[...]` and walking `input_ids`), but the motivations
/// there are speculative-decoding handling and `skip_special_tokens`
/// robustness – neither applies at the parser-library layer.
///
/// The `ParserTokenizer` parameter and `ParserInput.tokenIds` field
/// are preserved on the protocol surface as forward-looking
/// infrastructure; this parser doesn't read either.
struct Gemma4Parser: ResponseFormatParser {
  /// Initial reasoning phase. Set to ``reasoning`` when the parser should
  /// start already inside a `<|channel>thought` block (typically because
  /// the chat template placed the opener in the prompt).
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let thinkStart = "<|channel>thought"
  private static let thinkEnd = "<channel|>"
  private static let toolCallStart = "<|tool_call>"
  private static let toolCallEnd = "<tool_call|>"
  private static let stringDelim = #"<|"|>"#

  private var buffer: String = ""

  private enum Phase { case reasoning, normal }
  private var phase: Phase

  private var sentReasoningIdx: Int = 0
  private var sentContentIdx: Int = 0

  private var openReasoning: OpenReasoning?
  private var openMessage: OpenMessage?
  private var toolCalls: [OpenToolCall] = []
  private var activeToolCallIndex: Int?

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenToolCall {
    var id: String
    var callId: String
    /// Allocated lazily, once the name has been parsed and we're
    /// about to emit `output_item.added`. A malformed `<|tool_call>`
    /// region whose body never matches `call:NAME{` leaves this nil so
    /// no slot is consumed and the next item's index stays
    /// consecutive. Mirrors DeepSeekV3Parser's pattern.
    var outputIndex: Int?
    var name: String?
    var argsEmitted: String = ""
    var closed: Bool = false
  }

  init(initialState: InitialState = .normal) {
    switch initialState {
      case .normal: phase = .normal
      case .reasoning: phase = .reasoning
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
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    for index in toolCalls.indices where !toolCalls[index].closed {
      events.append(contentsOf: closeToolCall(at: index, status: .incomplete))
    }
    return events
  }

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var lastPhase: Phase? = nil
    while lastPhase != phase {
      lastPhase = phase
      switch phase {
        case .reasoning:
          events.append(contentsOf: scanReasoning(isEnd: isEnd))
        case .normal:
          events.append(contentsOf: scanNormal(isEnd: isEnd))
      }
    }
    return events
  }

  // MARK: Reasoning phase

  private mutating func scanReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)
    let startChars = Array(Gemma4Parser.thinkStart)
    let endChars = Array(Gemma4Parser.thinkEnd)
    let toolStartChars = Array(Gemma4Parser.toolCallStart)

    if sentReasoningIdx == 0,
       bufChars.count >= startChars.count,
       Array(bufChars[0 ..< startChars.count]) == startChars
    {
      // Want to also consume an optional trailing newline. If the buffer
      // ends exactly at the marker, more data may still arrive – hold so
      // we can swallow that newline atomically with the marker rather
      // than emit it as the first byte of reasoning text.
      let afterMarker = startChars.count
      if afterMarker >= bufChars.count, !isEnd {
        return events
      }
      var skip = afterMarker
      while skip < bufChars.count, bufChars[skip] == " " || bufChars[skip] == "\t" {
        skip += 1
      }
      if skip < bufChars.count, bufChars[skip] == "\n" { skip += 1 }
      sentReasoningIdx = skip
    } else if sentReasoningIdx == 0, !isEnd {
      let leadingOverlap = leadingPartialOverlap(of: bufChars, with: startChars)
      if leadingOverlap > 0, leadingOverlap == bufChars.count {
        return events
      }
    }

    let endIdx = bufChars.firstIndexOf(substring: Gemma4Parser.thinkEnd, after: sentReasoningIdx)
    let toolIdx = bufChars.firstIndexOf(substring: Gemma4Parser.toolCallStart, after: sentReasoningIdx)

    let exitIdx: Int?
    let exitMarker: ExitMarker?
    switch (endIdx, toolIdx) {
      case let (.some(e), .some(t)) where e <= t:
        exitIdx = e; exitMarker = .thinkEnd
      case (.some, .some):
        exitIdx = toolIdx; exitMarker = .toolCall
      case (.some, .none):
        exitIdx = endIdx; exitMarker = .thinkEnd
      case (.none, .some):
        exitIdx = toolIdx; exitMarker = .toolCall
      case (.none, .none):
        exitIdx = nil; exitMarker = nil
    }

    let safeEnd: Int
    if let exitIdx {
      safeEnd = exitIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let endOverlap = partialOverlap(suffixOf: bufChars, with: endChars)
      let toolOverlap = partialOverlap(suffixOf: bufChars, with: toolStartChars)
      safeEnd = bufChars.count - Swift.max(endOverlap, toolOverlap)
    }

    if safeEnd > sentReasoningIdx {
      let chunk = String(bufChars[sentReasoningIdx ..< safeEnd])
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
      sentReasoningIdx = safeEnd
    }

    guard let exitIdx, exitIdx == safeEnd, let exitMarker else { return events }

    switch exitMarker {
      case .thinkEnd:
        sentContentIdx = exitIdx + endChars.count
      case .toolCall:
        sentContentIdx = exitIdx
    }
    events.append(contentsOf: closeReasoning(status: .completed))
    phase = .normal
    return events
  }

  private enum ExitMarker { case thinkEnd, toolCall }

  private func leadingPartialOverlap(of chars: [Character], with tag: [Character]) -> Int {
    let limit = Swift.min(chars.count, tag.count - 1)
    if limit <= 0 { return 0 }
    var k = limit
    while k > 0 {
      if chars[..<k].elementsEqual(tag[..<k]) {
        return k
      }
      k -= 1
    }
    return 0
  }

  // MARK: Normal phase

  private mutating func scanNormal(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    if openMessage == nil, openReasoning == nil, toolCalls.isEmpty {
      if let transitionEvents = transitionToReasoningIfMarkerPresent(isEnd: isEnd) {
        return transitionEvents
      }
    }

    while true {
      events.append(contentsOf: flushContent(isEnd: isEnd))

      guard let cursor = buffer.index(
        buffer.startIndex,
        offsetBy: sentContentIdx,
        limitedBy: buffer.endIndex,
      ) else {
        return events
      }

      guard buffer[cursor...].hasPrefix(Gemma4Parser.toolCallStart) else {
        return events
      }

      if openMessage != nil {
        events.append(contentsOf: closeMessage(status: .completed))
      }

      let innerStart = buffer.index(cursor, offsetBy: Gemma4Parser.toolCallStart.count)
      let region: ToolCallRegion
      let regionEndOffset: Int?
      if let endRange = buffer.range(of: Gemma4Parser.toolCallEnd, range: innerStart ..< buffer.endIndex) {
        region = ToolCallRegion(rawInner: String(buffer[innerStart ..< endRange.lowerBound]), isComplete: true)
        regionEndOffset = buffer.distance(from: buffer.startIndex, to: endRange.upperBound)
      } else {
        region = ToolCallRegion(rawInner: String(buffer[innerStart ..< buffer.endIndex]), isComplete: false)
        regionEndOffset = nil
      }

      let index: Int
      if let existingIndex = activeToolCallIndex {
        index = existingIndex
      } else {
        // outputIndex is allocated lazily in `processRegion` once
        // the name is known, so a region whose body never becomes
        // `call:NAME{` doesn't burn an output slot.
        index = toolCalls.count
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
        ))
        activeToolCallIndex = index
      }

      events.append(contentsOf: processRegion(at: index, region: region, isEnd: isEnd))

      if region.isComplete, let regionEndOffset {
        sentContentIdx = regionEndOffset
        activeToolCallIndex = nil
        continue
      }

      return events
    }
  }

  private mutating func transitionToReasoningIfMarkerPresent(isEnd: Bool) -> [ResponseStreamingEvent]? {
    let bufChars = Array(buffer)
    let startChars = Array(Gemma4Parser.thinkStart)
    let cursor = sentContentIdx
    let available = bufChars.count - cursor

    if available >= startChars.count {
      if Array(bufChars[cursor ..< cursor + startChars.count]) == startChars {
        let afterMarker = cursor + startChars.count
        // Hold one cycle if the buffer ends exactly at the marker so a
        // trailing newline (if it shows up next) is consumed alongside
        // the marker rather than emitted as the first reasoning byte.
        if afterMarker >= bufChars.count, !isEnd {
          return []
        }
        var skip = afterMarker
        while skip < bufChars.count, bufChars[skip] == " " || bufChars[skip] == "\t" {
          skip += 1
        }
        if skip < bufChars.count, bufChars[skip] == "\n" { skip += 1 }
        sentReasoningIdx = skip
        phase = .reasoning
        return []
      }
      return nil
    }

    if available > 0 {
      let slice = Array(bufChars[cursor ..< bufChars.count])
      if Array(startChars[0 ..< slice.count]) == slice, !isEnd {
        return []
      }
    }
    return nil
  }

  private mutating func flushContent(isEnd: Bool) -> [ResponseStreamingEvent] {
    let bufChars = Array(buffer)
    let toolStartChars = Array(Gemma4Parser.toolCallStart)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: Gemma4Parser.toolCallStart, after: sentContentIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: toolStartChars)
      sendableEnd = bufChars.count - overlap
    }
    guard sendableEnd > sentContentIdx else { return [] }

    let chunk = String(bufChars[sentContentIdx ..< sendableEnd])
    sentContentIdx = sendableEnd
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

  private struct ToolCallRegion {
    var rawInner: String
    var isComplete: Bool
  }

  private mutating func processRegion(
    at index: Int,
    region: ToolCallRegion,
    isEnd: Bool,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var call = toolCalls[index]

    if call.name == nil {
      if let name = extractName(from: region.rawInner) {
        call.name = name
        let outputIndex = takeOutputIndex()
        call.outputIndex = outputIndex
        // Persist immediately so subsequent streaming passes see
        // the assigned name + outputIndex and skip re-emitting
        // `output_item.added`. Without this, the local-only
        // mutation would be re-derived on every char (since
        // `extractName` is deterministic) and burn a fresh
        // outputIndex slot per pass.
        toolCalls[index] = call
        let openItem = ResponseFunctionToolCall(
          id: call.id,
          callId: call.callId,
          name: name,
          arguments: "",
          status: .inProgress,
        )
        events.append(.outputItemAdded(.init(
          item: .functionCall(openItem),
          outputIndex: outputIndex,
          sequenceNumber: takeSequence(),
        )))
      }
    }

    if call.name != nil, let outputIndex = call.outputIndex, !call.closed {
      if region.isComplete || isEnd {
        let argsBody = extractArgsBody(from: region.rawInner) ?? ""
        let parsed = parseGemmaArgs(argsBody)
        let json = toJSONString(parsed)
        // Always emit the delta – even for `{}` – so that the
        // sum of `function_call_arguments.delta` events equals
        // the final `arguments` value on the closed item.
        // Suppressing the `{}` delta on truncation broke that
        // invariant for consumers that reconstruct args from
        // streamed deltas alone.
        events.append(.functionCallArgumentsDelta(.init(
          itemId: call.id,
          outputIndex: outputIndex,
          delta: json,
          sequenceNumber: takeSequence(),
        )))
        call.argsEmitted = json
        toolCalls[index] = call
        if region.isComplete {
          events.append(contentsOf: closeToolCall(at: index, status: .completed))
        }
      }
    }

    return events
  }

  private func extractName(from inner: String) -> String? {
    let trimmed = inner.drop(while: { $0.isWhitespace })
    guard trimmed.hasPrefix("call:") else { return nil }
    let afterCall = trimmed.dropFirst("call:".count)
    guard let braceIdx = afterCall.firstIndex(of: "{") else { return nil }
    let name = String(afterCall[..<braceIdx])
    return name.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  private func extractArgsBody(from inner: String) -> String? {
    let trimmed = inner.drop(while: { $0.isWhitespace })
    guard trimmed.hasPrefix("call:") else { return nil }
    let afterCall = trimmed.dropFirst("call:".count)
    guard let braceIdx = afterCall.firstIndex(of: "{") else { return nil }
    let argsContent = afterCall[afterCall.index(after: braceIdx)...]
    let argsChars = Array(argsContent)
    if let endIdx = findMatchingBrace(in: argsChars) {
      return String(argsChars[0 ..< endIdx])
    }
    return String(argsContent)
  }

  // MARK: Args parsing

  private func startsWith(_ chars: [Character], at idx: Int, prefix: [Character]) -> Bool {
    if idx + prefix.count > chars.count { return false }
    return chars[idx ..< idx + prefix.count].elementsEqual(prefix)
  }

  private func findMatchingBrace(in chars: [Character]) -> Int? {
    let delim = Array(Gemma4Parser.stringDelim)
    var depth = 1
    var i = 0
    let n = chars.count
    while i < n, depth > 0 {
      if startsWith(chars, at: i, prefix: delim) {
        i += delim.count
        var foundDelimEnd = false
        while i < n {
          if startsWith(chars, at: i, prefix: delim) {
            i += delim.count
            foundDelimEnd = true
            break
          }
          i += 1
        }
        if !foundDelimEnd { return nil }
        continue
      }
      if chars[i] == "{" {
        depth += 1
      } else if chars[i] == "}" {
        depth -= 1
        if depth == 0 { return i }
      }
      i += 1
    }
    return nil
  }

  private func findMatchingBracket(in chars: [Character]) -> Int? {
    let delim = Array(Gemma4Parser.stringDelim)
    var depth = 1
    var i = 0
    let n = chars.count
    while i < n, depth > 0 {
      if startsWith(chars, at: i, prefix: delim) {
        i += delim.count
        var foundDelimEnd = false
        while i < n {
          if startsWith(chars, at: i, prefix: delim) {
            i += delim.count
            foundDelimEnd = true
            break
          }
          i += 1
        }
        if !foundDelimEnd { return nil }
        continue
      }
      if chars[i] == "[" {
        depth += 1
      } else if chars[i] == "]" {
        depth -= 1
        if depth == 0 { return i }
      }
      i += 1
    }
    return nil
  }

  private func parseGemmaArgs(_ args: String) -> [String: Any] {
    var result: [String: Any] = [:]
    let chars = Array(args)
    var i = 0
    let n = chars.count
    while i < n {
      while i < n, chars[i].isWhitespace || chars[i] == "," {
        i += 1
      }
      if i >= n { break }

      let keyStart = i
      while i < n, chars[i] != ":" {
        i += 1
      }
      if i >= n { break }
      let key = String(chars[keyStart ..< i]).trimmingCharacters(in: .whitespacesAndNewlines)
      i += 1

      // `key:` with no value (or only trailing whitespace after the
      // colon) records `result[key] = ""`. Both vLLM
      // (`_parse_gemma4_args` non-partial path) and sglang
      // (`_parse_gemma4_args`, only path) explicitly handle this
      // case the same way; vLLM's test fixture pins
      // `_parse_gemma4_args("key:") == {"key": ""}`.
      if i >= n {
        if !key.isEmpty { result[key] = "" }
        break
      }
      while i < n, chars[i] == " " || chars[i] == "\t" {
        i += 1
      }
      if i >= n {
        if !key.isEmpty { result[key] = "" }
        break
      }

      let (value, newI) = parseGemmaValue(chars: chars, from: i)
      if !key.isEmpty {
        result[key] = value
      }
      i = newI
    }
    return result
  }

  private func parseGemmaValue(chars: [Character], from start: Int) -> (Any, Int) {
    let n = chars.count
    if start >= n { return ("", start) }
    let delim = Array(Gemma4Parser.stringDelim)

    if startsWith(chars, at: start, prefix: delim) {
      let i = start + delim.count
      var j = i
      while j < n {
        if startsWith(chars, at: j, prefix: delim) {
          return (String(chars[i ..< j]), j + delim.count)
        }
        j += 1
      }
      return (String(chars[i ..< n]), n)
    }

    if chars[start] == "{" {
      let inner = Array(chars[(start + 1) ..< n])
      if let endIdx = findMatchingBrace(in: inner) {
        let body = String(inner[0 ..< endIdx])
        return (parseGemmaArgs(body), start + endIdx + 2)
      }
      return (parseGemmaArgs(String(inner)), n)
    }

    if chars[start] == "[" {
      let inner = Array(chars[(start + 1) ..< n])
      if let endIdx = findMatchingBracket(in: inner) {
        let body = String(inner[0 ..< endIdx])
        return (parseGemmaArray(body), start + endIdx + 2)
      }
      return (parseGemmaArray(String(inner)), n)
    }

    var i = start
    while i < n, chars[i] != ",", chars[i] != "}", chars[i] != "]" {
      i += 1
    }
    let raw = String(chars[start ..< i]).trimmingCharacters(in: .whitespacesAndNewlines)
    return (parseGemmaScalar(raw), i)
  }

  private func parseGemmaArray(_ arr: String) -> [Any] {
    var items: [Any] = []
    let chars = Array(arr)
    var i = 0
    let n = chars.count
    while i < n {
      while i < n, chars[i].isWhitespace || chars[i] == "," {
        i += 1
      }
      if i >= n { break }
      let (value, newI) = parseGemmaValue(chars: chars, from: i)
      items.append(value)
      i = newI
    }
    return items
  }

  private func parseGemmaScalar(_ raw: String) -> Any {
    if raw.isEmpty { return raw }
    if raw == "true" { return true }
    if raw == "false" { return false }
    let lower = raw.lowercased()
    if lower == "null" || lower == "none" || lower == "nil" {
      return NSNull()
    }
    if raw.contains(".") {
      if let d = Double(raw) { return d }
    }
    if let i = Int(raw) { return i }
    // Python's `int(...)` is arbitrary precision, so the reference
    // emits a JSON number even for values that overflow Int64.
    // ``NSDecimalNumber`` can represent up to 38 significant digits and
    // round-trips through ``JSONSerialization`` as a JSON number, which
    // is enough for any realistic big-int a model emits.
    if isIntegerShaped(raw), let dec = Decimal(string: raw) {
      return NSDecimalNumber(decimal: dec)
    }
    return raw
  }

  private func isIntegerShaped(_ raw: String) -> Bool {
    var chars = raw[...]
    if chars.first == "-" || chars.first == "+" {
      chars = chars.dropFirst()
    }
    guard !chars.isEmpty else { return false }
    return chars.allSatisfy { $0.isASCII && $0.isNumber }
  }

  private func toJSONString(_ obj: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(
      withJSONObject: obj,
      // `.sortedKeys` for deterministic key order; Foundation
      // dictionaries don't preserve insertion order. Diverges from
      // sglang/vLLM, which emit in declaration order via Python
      // dicts' insertion-order guarantee.
      options: [.sortedKeys, .withoutEscapingSlashes],
    ) else { return "{}" }
    return String(data: data, encoding: .utf8) ?? "{}"
  }

  // MARK: Item open/close

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
        )),
        outputIndex: msg.outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeToolCall(at index: Int, status: ItemStatus) -> [ResponseStreamingEvent] {
    var call = toolCalls[index]
    guard !call.closed, let name = call.name, let outputIndex = call.outputIndex else {
      return []
    }
    call.closed = true
    toolCalls[index] = call
    let doneItem = ResponseFunctionToolCall(
      id: call.id,
      callId: call.callId,
      name: name,
      arguments: call.argsEmitted,
      status: status,
    )
    return [
      .functionCallArgumentsDone(.init(
        itemId: call.id,
        outputIndex: outputIndex,
        arguments: call.argsEmitted,
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .functionCall(doneItem),
        outputIndex: outputIndex,
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
