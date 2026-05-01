// Copyright © Anthony DePasquale

import Foundation

/// Best-effort fallback parser used when neither the name nor model_type
/// inference resolves to a known format.
///
/// Per the standalone-package plan, this is a deliberately minimal parser:
/// it streams plain-text output as message deltas, and at end-of-stream
/// retroactively recognizes a top-level JSON tool-call object or array if
/// the entire output happens to take that shape. Models that consistently
/// emit tool calls in some other format should be wired up to a per-format
/// parser; the JSON fallback exists so unknown models don't crash the
/// stream.
///
/// **State machine.**
///
/// - `undecided`: no non-whitespace text has arrived yet. The first
///   non-whitespace character decides the mode: `{` or `[` enters
///   `jsonBuffering`; anything else opens a message and enters `textOpen`.
/// - `textOpen`: a message item is open. Every chunk's text is emitted as a
///   `output_text.delta` against that message. `finalize()` closes the
///   message cleanly.
/// - `jsonBuffering`: text is accumulated until `finalize()`. At finalize,
///   the buffer is parsed: if it matches the tool-call shape (object with
///   `name` and `arguments`, or array of such objects), function_call items
///   are emitted; otherwise the buffer is replayed as a plain message.
struct JSONFallbackParser: ResponseFormatParser {
  private enum Mode {
    case undecided(buffer: String)
    case textOpen(itemId: String, accumulated: String)
    case jsonBuffering(buffer: String)
    case finished
  }

  private var mode: Mode = .undecided(buffer: "")
  private var nextSequence: Int = 0

  init() {}

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }

    switch mode {
      case var .undecided(buffer):
        buffer += chunk.text
        // Find the first non-whitespace character; if there is one,
        // pick the mode and process the buffer accordingly. Otherwise
        // keep accumulating whitespace.
        if let firstNonWS = buffer.firstIndex(where: { !$0.isWhitespace }) {
          let firstChar = buffer[firstNonWS]
          if firstChar == "{" || firstChar == "[" {
            mode = .jsonBuffering(buffer: buffer)
            return []
          } else {
            return openMessageAndEmit(buffer: buffer)
          }
        } else {
          mode = .undecided(buffer: buffer)
          return []
        }

      case .textOpen(let itemId, var accumulated):
        accumulated += chunk.text
        mode = .textOpen(itemId: itemId, accumulated: accumulated)
        return [
          .outputTextDelta(.init(
            itemId: itemId,
            outputIndex: 0,
            contentIndex: 0,
            delta: chunk.text,
            sequenceNumber: takeSequence(),
          )),
        ]

      case var .jsonBuffering(buffer):
        buffer += chunk.text
        mode = .jsonBuffering(buffer: buffer)
        return []

      case .finished:
        return []
    }
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    defer { mode = .finished }

    switch mode {
      case .undecided:
        // No content arrived. Nothing to close.
        return []

      case let .textOpen(itemId, accumulated):
        return closeMessage(itemId: itemId, text: accumulated, status: .completed)

      case let .jsonBuffering(buffer):
        return finalizeJsonBuffer(buffer)

      case .finished:
        return []
    }
  }

  // MARK: Mode transitions

  private mutating func openMessageAndEmit(buffer: String) -> [ResponseStreamingEvent] {
    let itemId = IDFactory.make(.message)
    mode = .textOpen(itemId: itemId, accumulated: buffer)

    let openMessage = ResponseOutputMessage(id: itemId, content: [], status: .inProgress)

    return [
      .outputItemAdded(.init(
        item: .message(openMessage),
        outputIndex: 0,
        sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: itemId,
        outputIndex: 0,
        contentIndex: 0,
        part: .outputText(.init(text: "")),
        sequenceNumber: takeSequence(),
      )),
      .outputTextDelta(.init(
        itemId: itemId,
        outputIndex: 0,
        contentIndex: 0,
        delta: buffer,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeMessage(
    itemId: String,
    text: String,
    status: ItemStatus,
  ) -> [ResponseStreamingEvent] {
    let finalPart = ResponseOutputText(text: text)
    let finalMessage = ResponseOutputMessage(
      id: itemId,
      content: [.outputText(finalPart)],
      status: status,
    )
    return [
      .outputTextDone(.init(
        itemId: itemId,
        outputIndex: 0,
        contentIndex: 0,
        text: text,
        sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: itemId,
        outputIndex: 0,
        contentIndex: 0,
        part: .outputText(finalPart),
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .message(finalMessage),
        outputIndex: 0,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func finalizeJsonBuffer(_ buffer: String) -> [ResponseStreamingEvent] {
    let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let data = trimmed.data(using: .utf8),
          let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    else {
      // Not parseable as JSON – replay as plain text message.
      return replayAsMessage(buffer: buffer, status: .incomplete)
    }

    let calls = extractToolCalls(from: parsed)
    if calls.isEmpty {
      // Valid JSON but not a tool-call shape – replay as a message
      // so downstream consumers see something useful.
      return replayAsMessage(buffer: buffer, status: .completed)
    }

    var events: [ResponseStreamingEvent] = []
    for (index, call) in calls.enumerated() {
      events.append(contentsOf: emitFunctionCall(call, outputIndex: index))
    }
    return events
  }

  private mutating func replayAsMessage(
    buffer: String,
    status: ItemStatus,
  ) -> [ResponseStreamingEvent] {
    let itemId = IDFactory.make(.message)
    var events: [ResponseStreamingEvent] = []

    events.append(.outputItemAdded(.init(
      item: .message(.init(id: itemId, content: [], status: .inProgress)),
      outputIndex: 0,
      sequenceNumber: takeSequence(),
    )))
    events.append(.contentPartAdded(.init(
      itemId: itemId,
      outputIndex: 0,
      contentIndex: 0,
      part: .outputText(.init(text: "")),
      sequenceNumber: takeSequence(),
    )))
    events.append(.outputTextDelta(.init(
      itemId: itemId,
      outputIndex: 0,
      contentIndex: 0,
      delta: buffer,
      sequenceNumber: takeSequence(),
    )))
    events.append(contentsOf: closeMessage(itemId: itemId, text: buffer, status: status))
    return events
  }

  private mutating func emitFunctionCall(
    _ call: ParsedToolCall,
    outputIndex: Int,
  ) -> [ResponseStreamingEvent] {
    let itemId = IDFactory.make(.functionCall)
    let callId = IDFactory.make(.callId)
    let openItem = ResponseFunctionToolCall(
      id: itemId,
      callId: callId,
      name: call.name,
      arguments: "",
      status: .inProgress,
    )
    let doneItem = ResponseFunctionToolCall(
      id: itemId,
      callId: callId,
      name: call.name,
      arguments: call.arguments,
      status: .completed,
    )

    return [
      .outputItemAdded(.init(
        item: .functionCall(openItem),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
      .functionCallArgumentsDelta(.init(
        itemId: itemId,
        outputIndex: outputIndex,
        delta: call.arguments,
        sequenceNumber: takeSequence(),
      )),
      .functionCallArgumentsDone(.init(
        itemId: itemId,
        outputIndex: outputIndex,
        name: call.name,
        arguments: call.arguments,
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .functionCall(doneItem),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
    ]
  }

  // MARK: JSON shape detection

  private struct ParsedToolCall {
    var name: String
    /// JSON-encoded arguments string (the spec wire shape).
    var arguments: String
  }

  private func extractToolCalls(from value: Any) -> [ParsedToolCall] {
    if let array = value as? [Any] {
      return array.compactMap(toolCall(from:))
    }
    if let single = toolCall(from: value) {
      return [single]
    }
    return []
  }

  private func toolCall(from value: Any) -> ParsedToolCall? {
    guard let dict = value as? [String: Any], let name = dict["name"] as? String else {
      return nil
    }
    let argsValue = dict["arguments"] ?? dict["parameters"]
    let arguments: String
    if let argsString = argsValue as? String {
      // Some models JSON-encode arguments as a string. Echo it back
      // verbatim – that's the spec's wire shape for `arguments`.
      arguments = argsString
    } else if let argsValue {
      guard let argsData = try? JSONSerialization.data(
        withJSONObject: argsValue,
        // `.sortedKeys` for deterministic key order; Foundation
        // dictionaries don't preserve insertion order. Diverges
        // from sglang/vLLM, which emit in declaration order via
        // Python dicts' insertion-order guarantee.
        options: [.sortedKeys, .fragmentsAllowed],
      ) else {
        return nil
      }
      arguments = String(data: argsData, encoding: .utf8) ?? ""
    } else {
      arguments = "{}"
    }
    return ParsedToolCall(name: name, arguments: arguments)
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
