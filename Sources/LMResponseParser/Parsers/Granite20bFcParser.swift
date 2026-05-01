// Copyright © Anthony DePasquale

import Foundation

/// Parser for IBM Granite-20B-FunctionCalling's tool-call format.
///
/// **Wire shape.** Each tool call is introduced by the literal string
/// `<function_call>` (a plain string, not a special token) followed by a
/// single JSON object `{"name": ..., "arguments": {...}}`. There is no
/// closing tag – calls are delimited by the next `<function_call>` or by
/// end of stream:
///
/// ```text
/// Let me look that up.
/// <function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}
/// <function_call> {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}
/// ```
///
/// **Reasoning.** Tool-call only. The 20B-FC checkpoint predates the
/// Granite reasoning lines.
///
/// **Streaming.** The parser holds the JSON bytes until a complete
/// brace-balanced object arrives (or the next `<function_call>` marker
/// closes it). Argument-delta streaming is per-call all-at-once, matching
/// the trade-off in `Phi4MiniParser` and `GraniteParser`. Mirrors vLLM's
/// `Granite20bFCToolParser`, which uses `JSONDecoder.raw_decode` on the
/// text after each marker.
struct Granite20bFcParser: ResponseFormatParser {
  private static let marker = "<function_call>"

  private var buffer: String = ""
  private var sentContentIdx: Int = 0
  /// Set once a `<function_call>` marker has been observed. After
  /// that, no more content is emitted – mirrors vLLM, which only
  /// surfaces text before the first marker as content and ignores
  /// inter-call whitespace and any trailing text.
  private var inToolCallPhase: Bool = false
  private var openMessage: OpenMessage?
  private var toolCalls: [OpenToolCall] = []

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenToolCall {
    var id: String
    var callId: String
    var outputIndex: Int?
    var name: String?
    var argsEmitted: String = ""
    var closed: Bool = false
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
    for index in toolCalls.indices where !toolCalls[index].closed {
      events.append(contentsOf: closeToolCall(at: index, status: .incomplete))
    }
    return events
  }

  // MARK: Scan

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let regions = extractRegions()
    for (index, region) in regions.enumerated() {
      events.append(contentsOf: flushContent(isEnd: isEnd))
      if index >= toolCalls.count {
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
        ))
      }
      events.append(contentsOf: processRegion(at: index, region: region, isEnd: isEnd))
    }
    events.append(contentsOf: flushContent(isEnd: isEnd))
    return events
  }

  private struct Region {
    var rawText: String
    var jsonText: String
    var isComplete: Bool
    /// Buffer offset right after the JSON object closed (when
    /// `isComplete`). Drives the content-cursor advance so trailing
    /// text after a complete call surfaces as a fresh message.
    var endOffset: Int?
  }

  /// Scan the buffer for each `<function_call>` marker and the JSON
  /// object that follows. The end of the object is determined by
  /// brace-balance, scoped by the next marker (which delimits the
  /// previous call) or end of buffer.
  private func extractRegions() -> [Region] {
    var results: [Region] = []
    let chars = Array(buffer)
    let needle = Array(Granite20bFcParser.marker)
    var pos = 0
    while let startIdx = chars.firstIndexOf(substring: Granite20bFcParser.marker, after: pos) {
      let jsonStart = skipWhitespace(in: chars, from: startIdx + needle.count)
      let nextStart = chars.firstIndexOf(substring: Granite20bFcParser.marker, after: jsonStart)
      let scope = nextStart ?? chars.count
      // Try to find a brace-balanced JSON object inside [jsonStart, scope).
      if jsonStart < scope, chars[jsonStart] == "{" {
        let end = endOfJSONValue(in: Array(chars[0 ..< scope]), from: jsonStart)
        let bracesClosed = end < scope || (end == scope && depthAtEnd(chars: chars, start: jsonStart, end: end) == 0)
        if bracesClosed, end <= scope {
          let inner = String(chars[jsonStart ..< end])
          if isValidJSON(inner) {
            results.append(Region(
              rawText: String(chars[startIdx ..< end]),
              jsonText: inner,
              isComplete: true,
              endOffset: end,
            ))
            pos = end
            continue
          }
        }
      }
      // Object isn't complete (or doesn't start with `{`). Take what's
      // available between this marker and the next.
      let raw = String(chars[jsonStart ..< scope])
      let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      let isComplete = !trimmed.isEmpty && isValidJSON(trimmed)
      results.append(Region(
        rawText: String(chars[startIdx ..< scope]),
        jsonText: trimmed,
        isComplete: isComplete,
        endOffset: isComplete ? scope : nil,
      ))
      if let nextStart {
        pos = nextStart
      } else {
        break
      }
    }
    return results
  }

  /// Compute the brace depth at `end` for chars[start..<end]. Used only
  /// for the marginal case where `endOfJSONValue` returns the buffer
  /// length – when that happens, the value is complete iff depth is 0.
  private func depthAtEnd(chars: [Character], start: Int, end: Int) -> Int {
    var depth = 0
    var inString = false
    var escape = false
    var i = start
    while i < end {
      let c = chars[i]
      if inString {
        if escape { escape = false; i += 1; continue }
        if c == "\\" { escape = true; i += 1; continue }
        if c == "\"" { inString = false }
        i += 1
        continue
      }
      if c == "\"" { inString = true; i += 1; continue }
      if c == "{" || c == "[" { depth += 1 }
      else if c == "}" || c == "]" { depth -= 1 }
      i += 1
    }
    return depth
  }

  private func skipWhitespace(in chars: [Character], from idx: Int) -> Int {
    var i = idx
    while i < chars.count, chars[i].isWhitespace {
      i += 1
    }
    return i
  }

  // MARK: Content flush

  private mutating func flushContent(isEnd: Bool) -> [ResponseStreamingEvent] {
    if inToolCallPhase { return [] }

    let chars = Array(buffer)
    let needle = Array(Granite20bFcParser.marker)

    let sendableEnd: Int
    if let startIdx = chars.firstIndexOf(substring: Granite20bFcParser.marker, after: sentContentIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = chars.count
    } else {
      let overlap = partialOverlap(suffixOf: chars, with: needle)
      sendableEnd = chars.count - overlap
    }
    guard sendableEnd > sentContentIdx else { return [] }

    let chunk = String(chars[sentContentIdx ..< sendableEnd])
    sentContentIdx = sendableEnd
    return emitMessageDelta(text: chunk)
  }

  private mutating func emitMessageDelta(text chunk: String) -> [ResponseStreamingEvent] {
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

  // MARK: Region processing

  private mutating func processRegion(
    at index: Int,
    region: Region,
    isEnd: Bool,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var call = toolCalls[index]

    if call.closed { return events }

    if call.name == nil {
      if let name = extractToolName(from: region.jsonText) {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        inToolCallPhase = true
        call.name = name
        let outputIndex = takeOutputIndex()
        call.outputIndex = outputIndex
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
      } else if region.isComplete || isEnd {
        // Complete but invalid `<function_call>` regions should not
        // disappear. vLLM's final extractor falls back to raw content on
        // malformed objects; preserving the bad region keeps that output
        // visible while still allowing earlier/later valid calls to parse.
        events.append(contentsOf: emitMessageDelta(text: region.rawText))
        if let endOffset = region.endOffset, endOffset > sentContentIdx {
          sentContentIdx = endOffset
        }
        call.closed = true
      }
    }

    if let outputIndex = call.outputIndex, call.name != nil, !call.closed {
      if region.isComplete || isEnd {
        let argsText = extractArgumentsText(from: region.jsonText, isComplete: region.isComplete) ?? ""
        let canonical = canonicalizeArgs(argsText)
        if !canonical.isEmpty, canonical.count > call.argsEmitted.count {
          let diffStart = canonical.index(canonical.startIndex, offsetBy: call.argsEmitted.count)
          let diff = String(canonical[diffStart...])
          call.argsEmitted = canonical
          events.append(.functionCallArgumentsDelta(.init(
            itemId: call.id,
            outputIndex: outputIndex,
            delta: diff,
            sequenceNumber: takeSequence(),
          )))
        }
      }
    }

    toolCalls[index] = call

    if !call.closed, region.isComplete {
      events.append(contentsOf: closeToolCall(at: index, status: .completed))
      if let endOffset = region.endOffset, endOffset > sentContentIdx {
        // Advance past any text (typically whitespace) between this
        // region and the next marker so it doesn't leak as a message.
        // Mirrors vLLM's `Granite20bFCToolParser`, which only takes
        // text *before the first marker* as content – everything after
        // is structured (JSON via `raw_decode`).
        let chars = Array(buffer)
        if let nextMarker = chars.firstIndexOf(substring: Granite20bFcParser.marker, after: endOffset) {
          sentContentIdx = nextMarker
        } else {
          sentContentIdx = chars.count
        }
      }
    }

    return events
  }

  /// Re-emit `arguments` in canonical form so the streamed delta and
  /// the final `done` payload match exactly. Mirrors how vLLM
  /// canonicalizes via `json.dumps(arguments)` after `raw_decode`.
  private func canonicalizeArgs(_ raw: String) -> String {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
          let data = trimmed.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    else { return trimmed }
    guard let serialized = try? JSONSerialization.data(
      withJSONObject: obj,
      options: [.sortedKeys, .withoutEscapingSlashes],
    ),
      let s = String(data: serialized, encoding: .utf8)
    else { return trimmed }
    return s
  }

  // MARK: Item open/close

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
        name: name,
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
