// Copyright © Anthony DePasquale

import Foundation

/// Parser for Tencent Hunyuan A13B's tool-call and reasoning format.
///
/// **Reasoning wire shape.** When reasoning is enabled, the canonical
/// output template is:
///
/// ```text
/// <think>\n REASONING \n</think>\n<answer>\n RESPONSE \n</answer>
/// ```
///
/// `RESPONSE` is the user-visible content, optionally containing tool
/// calls. The `<answer>` envelope is stripped from the surfaced
/// content; the leading and trailing newlines around the markers are
/// part of the markers themselves.
///
/// **Tool-call wire shape.** Tool calls share a single envelope
/// `<tool_calls>` … `</tool_calls>` containing a JSON array of
/// `{name, arguments}` objects, structurally identical to Jamba:
///
/// ```text
/// <tool_calls>[{"name": "fn", "arguments": {"x": 1}}, ...]</tool_calls>
/// ```
///
/// vLLM's reference parser filters out tool calls that appear inside a
/// `<think>` block (treating them as content); this parser inherits
/// that behavior because the reasoning phase consumes those bytes as
/// reasoning text rather than handing them to the tool-call extractor.
///
/// **Streaming.** The parser holds the array bytes until the closing
/// `</tool_calls>` arrives, then emits all calls at once. Per-call
/// argument-delta streaming via partial-JSON is not implemented.
struct HunyuanA13BParser: ResponseFormatParser {
  /// Initial reasoning phase. ``reasoning`` is used by continuation
  /// requests whose `priorOutput` ended inside an unclosed `<think>`
  /// block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  // Reasoning markers. The opening token is `<think>\n` and the
  // closing is `\n</think>\n` per Hunyuan A13B's canonical template
  // (the surrounding newlines are part of the marker IDs upstream).
  private static let thinkStart = "<think>\n"
  private static let thinkEnd = "\n</think>\n"
  // Optional answer envelope that wraps the response after `</think>`.
  private static let answerStart = "<answer>\n"
  private static let answerEnd = "\n</answer>"
  // Tool-call envelope (shared shape with Jamba).
  private static let toolCallsStart = "<tool_calls>"
  private static let toolCallsEnd = "</tool_calls>"
  // Hunyuan's chat template emits a literal `助手：` ("Assistant:")
  // prefix in some cases that vLLM strips from content. Mirror that
  // here so the consumer doesn't see the chat-template artifact.
  private static let assistantPrefix = "助手："

  private var buffer: String = ""
  private var openMessage: OpenMessage?
  private var openReasoning: OpenReasoning?
  private var phase: Phase
  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  /// State machine driven by the canonical
  /// `<think>...</think><answer>...</answer>` template plus optional
  /// tool-call envelopes inside the answer block. ``preReasoning`` is
  /// the entry state: a leading `<think>\n` enters reasoning, anything
  /// else jumps directly to ``content`` (for outputs without the
  /// reasoning preamble).
  private enum Phase {
    case preReasoning
    case reasoning
    case content
    case ended
  }

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  init(initialState: InitialState = .normal) {
    switch initialState {
      case .normal:
        phase = .preReasoning
      case .reasoning:
        phase = .reasoning
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

      switch phase {
        case .ended:
          // Drop everything past the answer-end marker.
          buffer = ""
          return events

        case .preReasoning:
          if buffer.isEmpty { return events }
          if buffer.hasPrefix(Self.thinkStart) {
            buffer.removeFirst(Self.thinkStart.count)
            phase = .reasoning
            didProgress = true
            continue
          }
          // Buffer is itself a (proper) prefix of the think-start
          // marker → hold; might still grow into the full marker.
          if Self.thinkStart.hasPrefix(buffer), !isEnd {
            return events
          }
          // No reasoning preamble. Skip directly to content phase.
          phase = .content
          didProgress = true

        case .reasoning:
          // Find the close marker. When found, emit reasoning text up
          // to it and transition. When not found, emit safe text and
          // hold back partial-marker overlap.
          if let endRange = buffer.range(of: Self.thinkEnd) {
            let chunk = String(buffer[buffer.startIndex ..< endRange.lowerBound])
            if !chunk.isEmpty {
              events.append(contentsOf: emitReasoningDelta(text: chunk))
            }
            let consumed = buffer.distance(from: buffer.startIndex, to: endRange.upperBound)
            buffer.removeFirst(consumed)
            events.append(contentsOf: closeReasoning(status: .completed))
            // After `\n</think>\n`, optionally consume `<answer>\n`.
            if buffer.hasPrefix(Self.answerStart) {
              buffer.removeFirst(Self.answerStart.count)
            }
            phase = .content
            didProgress = true
            continue
          }
          // No close marker. Emit safe text minus partial-marker
          // overlap.
          let bufChars = Array(buffer)
          let endChars = Array(Self.thinkEnd)
          let overlap = isEnd ? 0 : partialOverlap(suffixOf: bufChars, with: endChars)
          let safeEnd = bufChars.count - overlap
          if safeEnd > 0 {
            let chunk = String(bufChars[0 ..< safeEnd])
            events.append(contentsOf: emitReasoningDelta(text: chunk))
            buffer.removeFirst(safeEnd)
            didProgress = safeEnd > 0
          }
          return events

        case .content:
          // The content phase emits message text and parses
          // `<tool_calls>` envelopes. The optional `<answer>` start
          // (if reasoning was absent) is also stripped here as the
          // first thing so we don't surface it as content.
          if buffer.hasPrefix(Self.answerStart) {
            buffer.removeFirst(Self.answerStart.count)
            didProgress = true
            continue
          }

          // Strip leading chat-template prefix at most once per call.
          // Mirrors vLLM's `content.replace("助手：", "", 1)`.
          if buffer.hasPrefix(Self.assistantPrefix) {
            buffer.removeFirst(Self.assistantPrefix.count)
            didProgress = true
            continue
          }

          // Look for the answer-end marker; once found, drop it and
          // any trailing content.
          if let answerEndRange = buffer.range(of: Self.answerEnd) {
            // Emit safe content up to the answer-end (passing through
            // tool-call processing first if applicable).
            let safeText = String(buffer[buffer.startIndex ..< answerEndRange.lowerBound])
            buffer = safeText + String(buffer[answerEndRange.upperBound...])
            // Mark we should drop everything past the next iteration.
            // Simpler: just process the safeText here as content +
            // tool calls, then transition to .ended.
            // For simplicity, fall through: the buffer now is
            // safeText + (possible trailing). Process as normal.
            // Then on next iteration, we'd find another answerEnd
            // (no, we removed it). Need a cleaner approach.
            // Reset: replace buffer with just safeText, drop the
            // rest, mark phase=.ended at the start of the next loop.
            buffer = safeText
            // Process the remaining safeText through the
            // tool-call/content branches below, then fall out into
            // .ended on the next iteration.
            phase = .ended
            // Re-enter the content branch one more time to process
            // the safeText. But we've changed phase to .ended which
            // would just clear buffer. Workaround: process
            // content-events here, then return.
            // Simplest: forward to the normal content/tool-call
            // emission, then break.
            events.append(contentsOf: processContentChunk(isEnd: true))
            buffer = ""
            return events
          }

          // No answer-end yet. Process content/tool-calls as usual.
          let processed = processContentChunk(isEnd: isEnd)
          if !processed.isEmpty || buffer.isEmpty {
            events.append(contentsOf: processed)
            didProgress = !processed.isEmpty
          } else {
            return events
          }
      }
    }
    return events
  }

  /// Emit content / tool calls from the current buffer up to the
  /// safe boundary. Mirrors `JambaParser.scan`'s inner logic.
  private mutating func processContentChunk(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var didProgress = true

    while didProgress {
      didProgress = false

      guard let startRange = buffer.range(of: Self.toolCallsStart) else {
        // No envelope visible. Emit safe content; hold back any
        // partial-marker suffix. Also hold a partial `<answer>\n`
        // prefix when no content has been emitted yet — it might be
        // the optional answer-envelope opener that the outer
        // `.content` branch consumes.
        let bufChars = Array(buffer)
        let startOverlap = partialOverlap(suffixOf: bufChars, with: Array(Self.toolCallsStart))
        let endOverlap = partialOverlap(suffixOf: bufChars, with: Array(Self.toolCallsEnd))
        let answerEndOverlap = partialOverlap(suffixOf: bufChars, with: Array(Self.answerEnd))
        let answerStartOverlap = openMessage == nil
          ? partialOverlap(suffixOf: bufChars, with: Array(Self.answerStart))
          : 0
        let maxOverlap = Swift.max(startOverlap, Swift.max(endOverlap, Swift.max(answerEndOverlap, answerStartOverlap)))
        let safeEnd = isEnd ? buffer.count : buffer.count - maxOverlap
        if safeEnd > 0 {
          var safeText = String(buffer.prefix(safeEnd))
          if safeText.contains(Self.toolCallsEnd) {
            safeText = safeText.replacingOccurrences(of: Self.toolCallsEnd, with: "")
          }
          if !safeText.isEmpty {
            events.append(contentsOf: emitMessageDelta(text: safeText))
          }
          buffer.removeFirst(safeEnd)
          didProgress = safeEnd > 0
        }
        return events
      }

      let preText = String(buffer[buffer.startIndex ..< startRange.lowerBound])
      if !preText.isEmpty {
        events.append(contentsOf: emitMessageDelta(text: preText))
        buffer.removeFirst(preText.count)
        didProgress = true
        continue
      }

      let bodyStart = buffer.index(buffer.startIndex, offsetBy: Self.toolCallsStart.count)
      guard let endRange = buffer.range(of: Self.toolCallsEnd, range: bodyStart ..< buffer.endIndex) else {
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

      if let calls = parseToolCallArray(body) {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        for call in calls {
          events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
        }
      } else {
        let envelope = String(buffer.prefix(consumed))
        events.append(contentsOf: emitMessageDelta(text: envelope))
      }
      buffer.removeFirst(consumed)
      didProgress = true
    }
    return events
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
      let argsValue = dict["arguments"] ?? dict["parameters"] ?? [String: Any]()
      let argsJSON: String
      if let argsDict = argsValue as? [String: Any] {
        guard let argsData = try? JSONSerialization.data(
          withJSONObject: argsDict,
          // `.sortedKeys` enforces deterministic key order; Foundation
          // dictionaries don't preserve insertion order. Diverges from
          // sglang/vLLM, which emit in declaration order via Python
          // dicts' insertion-order guarantee.
          options: [.sortedKeys],
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

  private mutating func emitReasoningDelta(text: String) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var events: [ResponseStreamingEvent] = []
    if openReasoning == nil {
      events.append(contentsOf: openReasoningItem())
    }
    if var r = openReasoning {
      r.emittedText += text
      openReasoning = r
      events.append(.reasoningDelta(.init(
        itemId: r.id,
        outputIndex: r.outputIndex,
        contentIndex: 0,
        delta: text,
        sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  private mutating func openReasoningItem() -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.reasoning)
    let outputIndex = takeOutputIndex()
    openReasoning = OpenReasoning(id: id, outputIndex: outputIndex)
    return [
      .outputItemAdded(.init(
        item: .reasoning(.init(id: id, content: [], status: .inProgress)),
        outputIndex: outputIndex, sequenceNumber: takeSequence(),
      )),
      .contentPartAdded(.init(
        itemId: id, outputIndex: outputIndex, contentIndex: 0,
        part: .reasoningText(.init(text: "")), sequenceNumber: takeSequence(),
      )),
    ]
  }

  private mutating func closeReasoning(status: ItemStatus) -> [ResponseStreamingEvent] {
    guard let r = openReasoning else { return [] }
    openReasoning = nil
    let part = ReasoningTextContent(text: r.emittedText)
    return [
      .reasoningDone(.init(
        itemId: r.id, outputIndex: r.outputIndex, contentIndex: 0,
        text: r.emittedText, sequenceNumber: takeSequence(),
      )),
      .contentPartDone(.init(
        itemId: r.id, outputIndex: r.outputIndex, contentIndex: 0,
        part: .reasoningText(part), sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .reasoning(.init(id: r.id, content: [.reasoningText(part)], status: status)),
        outputIndex: r.outputIndex, sequenceNumber: takeSequence(),
      )),
    ]
  }

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
