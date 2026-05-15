// Copyright © Anthony DePasquale

import Foundation

/// Parser for the DeepSeek V3.1 / V3.2-Exp tool-call format.
///
/// **Wire shape.** V3.1 keeps the same CJK-bracket envelope as V3 but
/// drops V3's `function\n` literal and the `` ```json `` code fence
/// around the arguments. The function name appears directly after
/// `<｜tool▁call▁begin｜>` and the arguments are a bare JSON object
/// between `<｜tool▁sep｜>` and `<｜tool▁call▁end｜>`:
///
/// ```text
/// <｜tool▁calls▁begin｜>
/// <｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city":"Paris"}<｜tool▁call▁end｜>
/// <｜tool▁calls▁end｜>
/// ```
///
/// Plain text outside the envelope is normal message content.
///
/// **Optional reasoning preamble.** When constructed with
/// ``acceptThink`` true, a leading `<think>...</think>` block is
/// extracted as a reasoning item before the tool-call scan. Mirrors
/// vLLM's `DeepSeekV3ReasoningParser` with `chat_template_kwargs.thinking=True`,
/// which delegates to the R1 reasoning shape.
struct DeepSeekV31Parser: ResponseFormatParser {
  /// Initial reasoning phase. Used by continuation requests on
  /// thinking-enabled checkpoints whose `priorOutput` ended either
  /// inside or after the `<think>...</think>` block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let toolCallsBegin = "<｜tool▁calls▁begin｜>"
  private static let toolCallsEnd = "<｜tool▁calls▁end｜>"
  private static let toolCallBegin = "<｜tool▁call▁begin｜>"
  private static let toolCallEnd = "<｜tool▁call▁end｜>"
  private static let toolSep = "<｜tool▁sep｜>"

  /// Active suffix that has not yet been proven safe to discard.
  private var buffer: String = ""
  private var parsedIdx: Int = 0

  private var openMessage: OpenMessage?
  private var toolCalls: [OpenToolCall] = []

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private var insideToolCallsEnvelope: Bool = false
  private var insideSingleToolCall: Bool = false

  private let acceptThink: Bool
  private var thinkPreamble: ThinkPreambleExtractor

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  private struct OpenToolCall {
    var id: String
    var callId: String
    /// Allocated lazily, once the name has been parsed and we're
    /// about to emit `output_item.added`. Truncation between
    /// `<｜tool▁call▁begin｜>` and the name leaves this nil so no slot
    /// is consumed and the next item's index stays consecutive.
    /// Mirrors DeepSeekV3Parser's pattern.
    var outputIndex: Int?
    var name: String?
    var argsEmitted: String = ""
    var closed: Bool = false
  }

  /// - Parameters:
  ///   - acceptThink: When true, scans for a leading `<think>...</think>`
  ///     reasoning preamble before the tool-call body. Mirrors vLLM's
  ///     `DeepSeekV3ReasoningParser` with `chat_template_kwargs.thinking=True`.
  ///   - initialState: Used by continuation requests. ``InitialState/reasoning``
  ///     resumes mid-`<think>`; ``InitialState/normal`` skips the
  ///     preamble. Ignored when ``acceptThink`` is false.
  init(acceptThink: Bool = false, initialState: InitialState = .reasoning) {
    self.acceptThink = acceptThink
    if acceptThink {
      let preambleState: ThinkPreambleExtractor.InitialState = switch initialState {
        case .normal: .normal
        case .reasoning: .reasoning
      }
      thinkPreamble = ThinkPreambleExtractor(initialState: preambleState)
    } else {
      thinkPreamble = ThinkPreambleExtractor(initialState: .normal)
    }
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    var events: [ResponseStreamingEvent] = []
    if acceptThink, thinkPreamble.phase != .done {
      events.append(contentsOf: thinkPreamble.drain(
        buffer: &buffer,
        isEnd: false,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
    }
    if thinkPreamble.phase == .done {
      events.append(contentsOf: scan(isEnd: false))
    }
    return events
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    if acceptThink, thinkPreamble.phase != .done {
      events.append(contentsOf: thinkPreamble.drain(
        buffer: &buffer,
        isEnd: true,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
    }
    events.append(contentsOf: scan(isEnd: true))
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    for index in toolCalls.indices where !toolCalls[index].closed {
      events.append(contentsOf: closeToolCall(at: index, status: .incomplete))
    }
    if acceptThink {
      events.append(contentsOf: thinkPreamble.finalizeIfOpen(nextSequence: &nextSequence))
    }
    return events
  }

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    defer { pruneConsumedPrefix() }

    events.append(contentsOf: emitNormalText(isEnd: isEnd))

    while parsedIdx < buffer.count {
      let slice = buffer.dropFirst(parsedIdx)

      if slice.hasPrefix(DeepSeekV31Parser.toolCallsBegin) {
        parsedIdx += DeepSeekV31Parser.toolCallsBegin.count
        insideToolCallsEnvelope = true
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        continue
      }
      if slice.hasPrefix(DeepSeekV31Parser.toolCallsEnd) {
        parsedIdx += DeepSeekV31Parser.toolCallsEnd.count
        insideToolCallsEnvelope = false
        continue
      }
      if slice.hasPrefix(DeepSeekV31Parser.toolCallBegin) {
        parsedIdx += DeepSeekV31Parser.toolCallBegin.count
        insideSingleToolCall = true
        // outputIndex is allocated lazily in `parseFunctionHeader`
        // once the name is known, so a truncated header doesn't
        // burn an output slot.
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
        ))
        continue
      }
      if slice.hasPrefix(DeepSeekV31Parser.toolCallEnd) {
        parsedIdx += DeepSeekV31Parser.toolCallEnd.count
        insideSingleToolCall = false
        if let index = toolCalls.indices.last,
           !toolCalls[index].closed,
           toolCalls[index].name != nil
        {
          events.append(contentsOf: closeToolCall(at: index, status: .completed))
        }
        continue
      }

      if insideSingleToolCall, let last = toolCalls.indices.last, !toolCalls[last].closed {
        if toolCalls[last].name == nil {
          let advanced = parseFunctionHeader(at: parsedIdx, callIndex: last, events: &events)
          if advanced { continue } else { return events }
        } else {
          let advanced = parseArguments(at: parsedIdx, isEnd: isEnd, callIndex: last, events: &events)
          if advanced { continue } else { return events }
        }
      }

      if insideToolCallsEnvelope {
        if slice.first == "<", couldStillBecomeATag(slice: String(slice)), !isEnd {
          return events
        }
        if slice.first == "<" {
          // Stray `<` that doesn't begin a recognized tag (e.g.,
          // a malformed envelope where `<｜tool▁sep｜>` appears
          // without a preceding `<｜tool▁call▁begin｜>`). Consume
          // one character so the scan loop makes progress.
          parsedIdx += 1
          if parsedIdx == buffer.count { return events }
          continue
        }
        if let nextLt = slice.firstIndex(of: "<") {
          parsedIdx += slice.distance(from: slice.startIndex, to: nextLt)
          if parsedIdx == buffer.count { return events }
          continue
        } else {
          parsedIdx += slice.count
          continue
        }
      }
      break
    }
    return events
  }

  private mutating func pruneConsumedPrefix() {
    guard parsedIdx > 0 else { return }

    buffer.removeFirst(parsedIdx)
    parsedIdx = 0

    while let first = toolCalls.first, first.closed {
      toolCalls.removeFirst()
    }
  }

  private mutating func parseFunctionHeader(
    at start: Int,
    callIndex: Int,
    events: inout [ResponseStreamingEvent],
  ) -> Bool {
    let bufChars = Array(buffer)
    let sepChars = Array(DeepSeekV31Parser.toolSep)
    guard let sepIdx = bufChars.firstIndexOf(substring: DeepSeekV31Parser.toolSep, after: start) else {
      return false
    }
    let name = String(bufChars[start ..< sepIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
    let argsStart = sepIdx + sepChars.count

    if name.isEmpty {
      // Malformed envelope (`<｜tool_call_begin｜><｜tool_sep｜>…`).
      // Discard the slot. With lazy outputIndex allocation, no
      // index has been consumed yet, so simply removing the slot
      // keeps the output_index sequence consecutive. sglang's V3.1
      // detector emits `name=""` here; we go further and drop
      // the call rather than surface a nameless function call to
      // consumers.
      if callIndex < toolCalls.count {
        toolCalls.remove(at: callIndex)
      }
      insideSingleToolCall = false
      parsedIdx = argsStart
      return true
    }

    var call = toolCalls[callIndex]
    call.name = name
    let outputIndex = takeOutputIndex()
    call.outputIndex = outputIndex
    toolCalls[callIndex] = call

    events.append(.outputItemAdded(.init(
      item: .functionCall(.init(
        id: call.id, callId: call.callId, name: name, arguments: "", status: .inProgress,
      )),
      outputIndex: outputIndex,
      sequenceNumber: takeSequence(),
    )))

    parsedIdx = argsStart
    return true
  }

  private mutating func parseArguments(
    at start: Int,
    isEnd: Bool,
    callIndex: Int,
    events: inout [ResponseStreamingEvent],
  ) -> Bool {
    let bufChars = Array(buffer)
    let endTokenChars = Array(DeepSeekV31Parser.toolCallEnd)

    let endIdx = bufChars.firstIndexOf(substring: DeepSeekV31Parser.toolCallEnd, after: start)

    let safeEnd: Int
    if let endIdx {
      safeEnd = endIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: endTokenChars)
      safeEnd = bufChars.count - overlap
    }

    let newArgs = String(bufChars[start ..< safeEnd])
    if !newArgs.isEmpty {
      var call = toolCalls[callIndex]
      call.argsEmitted += newArgs
      toolCalls[callIndex] = call
      if let outputIndex = call.outputIndex {
        events.append(.functionCallArgumentsDelta(.init(
          itemId: call.id,
          outputIndex: outputIndex,
          delta: newArgs,
          sequenceNumber: takeSequence(),
        )))
      }
    }

    guard let endIdx, endIdx == safeEnd else {
      parsedIdx = safeEnd
      return false
    }
    parsedIdx = endIdx
    return true
  }

  private mutating func emitNormalText(isEnd: Bool) -> [ResponseStreamingEvent] {
    if insideToolCallsEnvelope { return [] }

    let bufChars = Array(buffer)
    let beginChars = Array(DeepSeekV31Parser.toolCallsBegin)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: DeepSeekV31Parser.toolCallsBegin, after: parsedIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: beginChars)
      sendableEnd = bufChars.count - overlap
    }
    guard sendableEnd > parsedIdx else { return [] }

    let chunk = String(bufChars[parsedIdx ..< sendableEnd])
    parsedIdx = sendableEnd
    if chunk.isEmpty { return [] }

    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem())
    }
    if var msg = openMessage {
      msg.emittedText += chunk
      openMessage = msg
      events.append(.outputTextDelta(.init(
        itemId: msg.id, outputIndex: msg.outputIndex, contentIndex: 0,
        delta: chunk, sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  private func couldStillBecomeATag(slice: String) -> Bool {
    for tag in [
      DeepSeekV31Parser.toolCallsBegin,
      DeepSeekV31Parser.toolCallsEnd,
      DeepSeekV31Parser.toolCallBegin,
      DeepSeekV31Parser.toolCallEnd,
      DeepSeekV31Parser.toolSep,
    ] {
      if tag.hasPrefix(slice) { return true }
    }
    return false
  }

  private mutating func closeToolCall(at index: Int, status: ItemStatus) -> [ResponseStreamingEvent] {
    var call = toolCalls[index]
    guard !call.closed, let name = call.name, let outputIndex = call.outputIndex else {
      return []
    }
    call.closed = true
    toolCalls[index] = call

    let doneItem = ResponseFunctionToolCall(
      id: call.id, callId: call.callId, name: name, arguments: call.argsEmitted, status: status,
    )
    return [
      .functionCallArgumentsDone(.init(
        itemId: call.id, outputIndex: outputIndex,
        arguments: call.argsEmitted, sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .functionCall(doneItem), outputIndex: outputIndex, sequenceNumber: takeSequence(),
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

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
