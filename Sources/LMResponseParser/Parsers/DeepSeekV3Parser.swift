// Copyright © Anthony DePasquale

import Foundation

/// Parser for the DeepSeek V3 base tool-call format.
///
/// **Wire shape.** Same CJK-bracket protocol as DeepSeek R1, but without
/// the `<think>` reasoning split:
///
/// ```text
/// <｜tool▁calls▁begin｜>
/// <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
/// ```json
/// {"arg": "value"}
/// ```<｜tool▁call▁end｜>
/// <｜tool▁calls▁end｜>
/// ```
///
/// Plain text outside the envelope is normal message content.
///
/// **Optional reasoning preamble.** When constructed with
/// ``acceptThink`` true, a leading `<think>...</think>` block is
/// extracted as a reasoning item before the tool-call scan. Mirrors
/// vLLM's `DeepSeekV3ReasoningParser` with `chat_template_kwargs.thinking=True`,
/// which delegates to the R1 reasoning shape. The `<think>` opener is
/// optional – chat templates often inject it into the prompt, in which
/// case the model emits only `</think>` to close.
struct DeepSeekV3Parser: ResponseFormatParser {
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
  private static let jsonFenceOpen = "```json\n"
  private static let jsonFenceClose = "\n```"

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
    /// Allocated lazily, once the name has been parsed and we're about
    /// to emit `output_item.added`. Truncation between
    /// `<｜tool▁call▁begin｜>` and the name leaves this nil so no slot is
    /// consumed and the next item's index stays consecutive.
    var outputIndex: Int?
    var name: String?
    var argsEmitted: String = ""
    var closed: Bool = false
  }

  /// - Parameters:
  ///   - acceptThink: When true, scans for a leading `<think>...</think>`
  ///     reasoning preamble before the tool-call body. Mirrors vLLM's
  ///     `DeepSeekV3ReasoningParser` with `chat_template_kwargs.thinking=True`.
  ///     Off by default; base V3 emits no reasoning markers.
  ///   - initialState: Used by continuation requests on thinking-enabled
  ///     checkpoints. ``InitialState/reasoning`` starts already inside
  ///     the reasoning preamble (continuation mid-`<think>`);
  ///     ``InitialState/normal`` skips the preamble entirely
  ///     (continuation post-`</think>`). Ignored when
  ///     ``acceptThink`` is false.
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
    events.append(contentsOf: emitNormalText(isEnd: isEnd))

    while parsedIdx < buffer.count {
      let slice = buffer.dropFirst(parsedIdx)

      if slice.hasPrefix(DeepSeekV3Parser.toolCallsBegin) {
        parsedIdx += DeepSeekV3Parser.toolCallsBegin.count
        insideToolCallsEnvelope = true
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        continue
      }
      if slice.hasPrefix(DeepSeekV3Parser.toolCallsEnd) {
        parsedIdx += DeepSeekV3Parser.toolCallsEnd.count
        insideToolCallsEnvelope = false
        continue
      }
      if slice.hasPrefix(DeepSeekV3Parser.toolCallBegin) {
        parsedIdx += DeepSeekV3Parser.toolCallBegin.count
        insideSingleToolCall = true
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
        ))
        continue
      }
      if slice.hasPrefix(DeepSeekV3Parser.toolCallEnd) {
        parsedIdx += DeepSeekV3Parser.toolCallEnd.count
        insideSingleToolCall = false
        if let index = toolCalls.indices.last, !toolCalls[index].closed,
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

  private mutating func parseFunctionHeader(
    at start: Int,
    callIndex: Int,
    events: inout [ResponseStreamingEvent],
  ) -> Bool {
    let bufChars = Array(buffer)
    let sepChars = Array(DeepSeekV3Parser.toolSep)
    guard let sepIdx = bufChars.firstIndexOf(substring: DeepSeekV3Parser.toolSep, after: start) else {
      return false
    }
    let nameStart = sepIdx + sepChars.count

    var nameEnd: Int? = nil
    var i = nameStart
    while i < bufChars.count {
      if bufChars[i] == "\n" { nameEnd = i; break }
      i += 1
    }
    guard let nameEnd else { return false }
    // sglang's streaming detector strips the captured name; R1, V3.1,
    // and V3.2 mirror that. Match here so consumers do not see leading
    // or trailing whitespace surfacing inside `<｜tool▁sep｜> name \n`.
    let name = String(bufChars[nameStart ..< nameEnd])
      .trimmingCharacters(in: .whitespacesAndNewlines)

    let fenceOpenChars = Array(DeepSeekV3Parser.jsonFenceOpen)
    guard let fenceIdx = bufChars.firstIndexOf(substring: DeepSeekV3Parser.jsonFenceOpen, after: nameEnd) else {
      return false
    }
    let argsStart = fenceIdx + fenceOpenChars.count

    if name.isEmpty {
      // Drop malformed nameless calls before allocating an output slot.
      // This mirrors the defensive V3.1/V3.2 behavior and keeps
      // downstream dispatch from seeing an unusable function call.
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
    let fenceCloseChars = Array(DeepSeekV3Parser.jsonFenceClose)
    let endTokenChars = Array(DeepSeekV3Parser.toolCallEnd)

    let fenceIdx = bufChars.firstIndexOf(substring: DeepSeekV3Parser.jsonFenceClose, after: start)
    let endIdx = bufChars.firstIndexOf(substring: DeepSeekV3Parser.toolCallEnd, after: start)

    let stopIdx: Int?
    let stopIsFence: Bool
    switch (fenceIdx, endIdx) {
      case let (.some(f), .some(e)) where f <= e:
        stopIdx = f; stopIsFence = true
      case (.some, .some):
        stopIdx = endIdx; stopIsFence = false
      case (.some, .none):
        stopIdx = fenceIdx; stopIsFence = true
      case (.none, .some):
        stopIdx = endIdx; stopIsFence = false
      case (.none, .none):
        stopIdx = nil; stopIsFence = false
    }

    let safeEnd: Int
    if let stopIdx {
      safeEnd = stopIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let fenceOverlap = partialOverlap(suffixOf: bufChars, with: fenceCloseChars)
      let endOverlap = partialOverlap(suffixOf: bufChars, with: endTokenChars)
      safeEnd = bufChars.count - Swift.max(fenceOverlap, endOverlap)
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

    guard let stopIdx, stopIdx == safeEnd else {
      parsedIdx = safeEnd
      return false
    }
    if stopIsFence {
      parsedIdx = stopIdx + fenceCloseChars.count
    } else {
      parsedIdx = stopIdx
    }
    return true
  }

  private mutating func emitNormalText(isEnd: Bool) -> [ResponseStreamingEvent] {
    if insideToolCallsEnvelope { return [] }

    let bufChars = Array(buffer)
    let beginChars = Array(DeepSeekV3Parser.toolCallsBegin)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: DeepSeekV3Parser.toolCallsBegin, after: parsedIdx) {
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
      DeepSeekV3Parser.toolCallsBegin,
      DeepSeekV3Parser.toolCallsEnd,
      DeepSeekV3Parser.toolCallBegin,
      DeepSeekV3Parser.toolCallEnd,
      DeepSeekV3Parser.toolSep,
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
        name: name, arguments: call.argsEmitted, sequenceNumber: takeSequence(),
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
