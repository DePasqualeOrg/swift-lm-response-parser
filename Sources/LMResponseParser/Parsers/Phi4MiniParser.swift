// Copyright © Anthony DePasquale

import Foundation

/// Parser for `microsoft/Phi-4-mini-instruct`'s `functools[...]` tool-call
/// format.
///
/// **Wire shape.** Tool calls appear inside a JSON array prefixed with the
/// literal opener `functools[`:
///
/// ```text
/// functools[{"name": "fn", "arguments": {"x": 1}}, {"name": "fn2", "arguments": {}}]
/// ```
///
/// The opener is plain text (not a special token). Each array element is a
/// JSON object with a `"name"` field and either an `"arguments"` or
/// `"parameters"` field whose value is the call's argument object. Multiple
/// calls share a single `functools[...]` envelope. Plain text may precede
/// the envelope and is forwarded as message content.
///
/// **Streaming.** The parser holds the buffer until a complete bracket-
/// balanced array arrives, then emits all calls in that envelope at once.
/// Per-call argument-delta streaming is not implemented because Phi-4-mini
/// outputs are typically short and vLLM's reference parser is also
/// non-streaming. The leading message content streams chunk by chunk; only
/// the tool-call payload is held back.
struct Phi4MiniParser: ResponseFormatParser {
  private static let envelopeOpener = "functools["

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

      guard let envelopeRange = buffer.range(of: Self.envelopeOpener) else {
        // No `functools[` in buffer. Emit safe content; hold back
        // any partial-opener suffix until the next chunk completes
        // or doesn't.
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

      // Emit pre-envelope text (if any) as message content.
      let preText = String(buffer[buffer.startIndex ..< envelopeRange.lowerBound])
      if !preText.isEmpty {
        events.append(contentsOf: emitMessageDelta(text: preText))
        buffer.removeFirst(preText.count)
        didProgress = true
        continue
      }

      // Buffer starts with `functools[`. The `[` of the opener is
      // also the start of the JSON array we need to balance.
      let openerLength = Self.envelopeOpener.count
      let bracketStart = buffer.index(buffer.startIndex, offsetBy: openerLength - 1)
      guard let bracketEnd = matchingCloseBracket(in: buffer, openAt: bracketStart) else {
        // Array not yet balanced. At end-of-stream, gracefully
        // surface the unbalanced envelope as plain content rather
        // than dropping it.
        if isEnd {
          events.append(contentsOf: emitMessageDelta(text: buffer))
          buffer = ""
          didProgress = true
        }
        return events
      }

      let arrayText = String(buffer[bracketStart ... bracketEnd])
      let consumed = buffer.distance(from: buffer.startIndex, to: bracketEnd) + 1

      if let calls = parseFunctoolsArray(arrayText) {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        for call in calls {
          events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
        }
      } else {
        // JSON parse failed – treat the whole envelope as content.
        let envelopeText = String(buffer.prefix(consumed))
        events.append(contentsOf: emitMessageDelta(text: envelopeText))
      }
      buffer.removeFirst(consumed)
      didProgress = true
    }
    return events
  }

  // MARK: Bracket and overlap helpers

  /// Find the `]` that balances the `[` at `openAt`. Skips brackets
  /// inside JSON string literals (which use `"` quotes) so a string
  /// value containing `[` or `]` does not close the array prematurely.
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

  private func trailingPartialOpenerOverlap(of text: String) -> Int {
    let chars = Array(text)
    return partialOverlap(suffixOf: chars, with: Array(Self.envelopeOpener))
  }

  // MARK: JSON parsing

  private struct ParsedCall {
    let name: String
    let arguments: String
  }

  private func parseFunctoolsArray(_ arrayText: String) -> [ParsedCall]? {
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
      // Phi-4-mini's reference accepts either `"arguments"` or the
      // older `"parameters"` field. If both are missing, default to
      // an empty object – vLLM treats this as a malformed call but
      // there's no harm in surfacing the call with empty args.
      let argsValue = dict["arguments"] ?? dict["parameters"] ?? [String: Any]()
      let argsJSON: String
      if let argsDict = argsValue as? [String: Any] {
        guard let argsData = try? JSONSerialization.data(
          withJSONObject: argsDict,
          // `.sortedKeys` for deterministic key order; Foundation
          // dictionaries don't preserve insertion order. Diverges
          // from sglang/vLLM, which emit in declaration order via
          // Python dicts' insertion-order guarantee.
          options: [.sortedKeys],
        ),
          let s = String(data: argsData, encoding: .utf8)
        else {
          return nil
        }
        argsJSON = s
      } else if let argsString = argsValue as? String {
        // Some chat templates double-encode args as a JSON string.
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
