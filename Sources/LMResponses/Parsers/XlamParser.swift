// Copyright © Anthony DePasquale

import Foundation

/// Parser for Salesforce xLAM-style tool calls.
///
/// **Wire shape.** xLAM models emit a JSON **array** of `{name,
/// arguments}` objects, optionally wrapped in one of four envelopes:
///
/// - `<tool_call>...</tool_call>` containing a JSON array (distinct from
///   Hermes / Qwen, which wrap a single object).
/// - `[TOOL_CALLS]...` prefix followed by a JSON array (overlaps with
///   `.mistral`).
/// - Triple-backtick fenced JSON: ```` ```json ... ``` ```` or
///   ```` ``` ... ``` ````.
/// - Bare JSON array (no wrapper).
///
/// A `<think>...</think>` reasoning block at the start is recognized
/// only as a position cue: text up to and including `</think>` is
/// preserved as message content, and the tool-call detector runs on
/// what follows. Standalone reasoning extraction is out of scope here.
///
/// **Streaming.** The parser holds the buffer until a complete envelope
/// is detected, then emits all calls in one batch. Partial JSON
/// streaming is not implemented (mirrors `Phi4Mini`, `Jamba`).
///
/// **Reference**: `vllm/tool_parsers/xlam_tool_parser.py`. Reference
/// models include `Salesforce/Llama-xLAM-2-8B-fc-r`,
/// `Salesforce/xLAM-1B-fc-r`, `Salesforce/Qwen-xLAM-32B-fc-r`.
struct XlamParser: ResponseFormatParser {
  private static let toolCallStart = "<tool_call>"
  private static let toolCallEnd = "</tool_call>"
  private static let toolCallsPrefix = "[TOOL_CALLS]"
  private static let codeFence = "```"
  private static let thinkEnd = "</think>"

  private var buffer: String = ""
  private var openMessage: OpenMessage?
  private var emitted: Bool = false
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
    return tryEmit(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = tryEmit(isEnd: true)
    if !emitted, !buffer.isEmpty {
      events.append(contentsOf: emitMessageDelta(text: buffer))
      buffer = ""
    }
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    return events
  }

  /// Attempt to detect and emit a complete tool-call envelope. Returns
  /// the events emitted (possibly empty if no envelope detected yet).
  /// Once emitted, sets `emitted = true` so subsequent buffer growth
  /// flushes only as plain content.
  private mutating func tryEmit(isEnd: Bool) -> [ResponseStreamingEvent] {
    if emitted {
      // Already emitted; further bytes are ordinary content.
      let text = buffer
      buffer = ""
      return text.isEmpty ? [] : emitMessageDelta(text: text)
    }
    guard let detected = detectEnvelope() else {
      // No envelope yet. At end-of-stream, fall through to emit-as-
      // content in finalize.
      if isEnd { return [] }
      return []
    }

    var events: [ResponseStreamingEvent] = []
    if !detected.preText.isEmpty {
      events.append(contentsOf: emitMessageDelta(text: detected.preText))
    }
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    for call in detected.calls {
      events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
    }
    buffer = String(buffer[detected.consumedEnd...])
    emitted = true
    return events
  }

  // MARK: Envelope detection

  private struct Detected {
    var preText: String
    var calls: [ParsedCall]
    var consumedEnd: String.Index
  }

  private struct ParsedCall {
    let name: String
    let arguments: String
  }

  /// Inspect the current buffer for a complete envelope. Tries the four
  /// wrappers in priority order; falls back to a bare JSON array.
  private func detectEnvelope() -> Detected? {
    // Strip a leading `<think>...</think>` reasoning block as
    // pre-content. Anything before `</think>` (inclusive) is preserved
    // as message text; tool-call detection runs on what follows.
    var scanFrom = buffer.startIndex
    var preText = ""
    if let thinkEndRange = buffer.range(of: Self.thinkEnd) {
      preText = String(buffer[buffer.startIndex ..< thinkEndRange.upperBound])
      scanFrom = thinkEndRange.upperBound
    }

    // 1. `<tool_call>...</tool_call>` — explicit envelope.
    var toolCallSearchStart = scanFrom
    while let openRange = buffer.range(of: Self.toolCallStart, range: toolCallSearchStart ..< buffer.endIndex),
          let closeRange = buffer.range(of: Self.toolCallEnd, range: openRange.upperBound ..< buffer.endIndex)
    {
      let inner = String(buffer[openRange.upperBound ..< closeRange.lowerBound])
        .trimmingCharacters(in: .whitespacesAndNewlines)
      if let calls = parseJSONArray(inner) {
        let between = String(buffer[scanFrom ..< openRange.lowerBound])
        let combinedPre = preText + between
        return Detected(preText: combinedPre, calls: calls, consumedEnd: closeRange.upperBound)
      }
      toolCallSearchStart = closeRange.upperBound
    }

    // 2. `[TOOL_CALLS]` prefix — Mistral-style.
    var toolCallsSearchStart = scanFrom
    while let prefixRange = buffer.range(of: Self.toolCallsPrefix, range: toolCallsSearchStart ..< buffer.endIndex) {
      let bodyStart = prefixRange.upperBound
      // Skip whitespace.
      var afterPrefix = bodyStart
      while afterPrefix < buffer.endIndex, buffer[afterPrefix].isWhitespace {
        afterPrefix = buffer.index(after: afterPrefix)
      }
      if afterPrefix < buffer.endIndex, buffer[afterPrefix] == "[" {
        if let closeIdx = matchingBracket(in: buffer, openAt: afterPrefix) {
          let inner = String(buffer[afterPrefix ... closeIdx])
            .trimmingCharacters(in: .whitespacesAndNewlines)
          if let calls = parseJSONArray(inner) {
            let between = String(buffer[scanFrom ..< prefixRange.lowerBound])
            let combinedPre = preText + between
            return Detected(
              preText: combinedPre,
              calls: calls,
              consumedEnd: buffer.index(after: closeIdx),
            )
          }
        }
      }
      toolCallsSearchStart = prefixRange.upperBound
    }

    // 3. Triple-backtick fenced code block.
    var fenceSearchStart = scanFrom
    while let openFence = buffer.range(of: Self.codeFence, range: fenceSearchStart ..< buffer.endIndex) {
      var bodyStart = openFence.upperBound
      // Skip optional `json` language tag.
      if buffer[bodyStart...].hasPrefix("json") {
        bodyStart = buffer.index(bodyStart, offsetBy: "json".count)
      }
      while bodyStart < buffer.endIndex, buffer[bodyStart].isWhitespace {
        bodyStart = buffer.index(after: bodyStart)
      }
      if let closeFence = buffer.range(of: Self.codeFence, range: bodyStart ..< buffer.endIndex) {
        let inner = String(buffer[bodyStart ..< closeFence.lowerBound])
          .trimmingCharacters(in: .whitespacesAndNewlines)
        if let calls = parseJSONArray(inner) {
          let between = String(buffer[scanFrom ..< openFence.lowerBound])
          let combinedPre = preText + between
          return Detected(
            preText: combinedPre,
            calls: calls,
            consumedEnd: closeFence.upperBound,
          )
        }
        fenceSearchStart = closeFence.upperBound
      } else {
        break
      }
    }

    // 4. Bare JSON array.
    var arrayStart = scanFrom
    while arrayStart < buffer.endIndex, buffer[arrayStart].isWhitespace {
      arrayStart = buffer.index(after: arrayStart)
    }
    if arrayStart < buffer.endIndex, buffer[arrayStart] == "[" {
      if let closeIdx = matchingBracket(in: buffer, openAt: arrayStart) {
        let inner = String(buffer[arrayStart ... closeIdx])
        if let calls = parseJSONArray(inner) {
          let between = String(buffer[scanFrom ..< arrayStart])
          let combinedPre = preText + between
          return Detected(
            preText: combinedPre,
            calls: calls,
            consumedEnd: buffer.index(after: closeIdx),
          )
        }
      }
    }

    return nil
  }

  /// Find the index of the `]` that closes the `[` at `openAt`, with
  /// JSON string/escape awareness. Returns nil when the array isn't
  /// balanced yet.
  private func matchingBracket(in slice: String, openAt: String.Index) -> String.Index? {
    var depth = 0
    var inString = false
    var escape = false
    var i = openAt
    while i < slice.endIndex {
      let c = slice[i]
      if escape { escape = false; i = slice.index(after: i); continue }
      if c == "\\" { escape = true; i = slice.index(after: i); continue }
      if inString {
        if c == "\"" { inString = false }
        i = slice.index(after: i)
        continue
      }
      if c == "\"" {
        inString = true
        i = slice.index(after: i)
        continue
      }
      if c == "[" { depth += 1 }
      else if c == "]" {
        depth -= 1
        if depth == 0 { return i }
      }
      i = slice.index(after: i)
    }
    return nil
  }

  /// Parse the array text as `[{name, arguments}, ...]`. Returns nil
  /// when the JSON doesn't decode or the shape isn't an array of
  /// `{name, arguments}` objects. Returns an empty array when the JSON
  /// parses but contains no entries.
  private func parseJSONArray(_ text: String) -> [ParsedCall]? {
    guard let data = text.data(using: .utf8),
          let array = try? JSONSerialization.jsonObject(with: data) as? [Any]
    else { return nil }
    if array.isEmpty { return [] }
    var calls: [ParsedCall] = []
    for entry in array {
      guard let dict = entry as? [String: Any],
            let name = dict["name"] as? String,
            !name.isEmpty
      else {
        continue
      }
      let argsValue = dict["arguments"] ?? dict["parameters"] ?? [String: Any]()
      let argsJSON: String
      if let argsDict = argsValue as? [String: Any] {
        guard let argsData = try? JSONSerialization.data(
          withJSONObject: argsDict,
          options: [.sortedKeys, .withoutEscapingSlashes],
        ),
          let s = String(data: argsData, encoding: .utf8)
        else {
          return nil
        }
        argsJSON = s
      } else if let argsString = argsValue as? String {
        argsJSON = argsString
      } else {
        argsJSON = "{}"
      }
      calls.append(ParsedCall(name: name, arguments: argsJSON))
    }
    return calls
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
