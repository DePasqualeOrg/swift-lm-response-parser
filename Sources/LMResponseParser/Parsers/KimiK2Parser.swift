// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Kimi K2 tool-call format.
///
/// **Wire shape.** Tool calls live inside a `<|tool_calls_section_begin|>`
/// envelope, with each individual call delimited by
/// `<|tool_call_begin|>` … `<|tool_call_end|>`. The function identifier
/// uses the `functions.NAME:INDEX` convention, and arguments live after a
/// `<|tool_call_argument_begin|>` separator as JSON:
///
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// Plain text outside the envelope is normal message content. Multiple
/// calls in a single envelope each emit their own `function_call` item.
///
/// **Reasoning preamble (Kimi-K2-Thinking).** When constructed with
/// ``InitialState/reasoning``, the parser begins in a reasoning phase
/// that ends at the first `</think>` *or* `<|tool_calls_section_begin|>`.
/// A leading `<think>` opener is optional (consumed if present). This
/// mirrors vLLM's `KimiK2ReasoningParser` and SGLang's `KimiK2Detector`,
/// where the tool-call section start is treated as an implicit reasoning
/// end. The default ``InitialState/normal`` matches Kimi-K2-Instruct,
/// which emits no reasoning preamble.
struct KimiK2Parser: ResponseFormatParser {
  /// Initial reasoning phase. Default ``normal`` matches
  /// Kimi-K2-Instruct (no reasoning preamble). Pass ``reasoning`` for
  /// Kimi-K2-Thinking, where the chat template configures the model to
  /// begin output inside an implicit reasoning block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let sectionBegin = "<|tool_calls_section_begin|>"
  private static let sectionEnd = "<|tool_calls_section_end|>"
  private static let callBegin = "<|tool_call_begin|>"
  private static let callEnd = "<|tool_call_end|>"
  private static let argBegin = "<|tool_call_argument_begin|>"
  private static let thinkStart = "<think>"
  private static let thinkEnd = "</think>"

  /// Active suffix that has not yet been proven safe to discard.
  private var buffer: String = ""
  private var parsedIdx: Int = 0

  private enum Phase { case reasoning, normal }
  private var phase: Phase

  private var openReasoning: OpenReasoning?
  private var openMessage: OpenMessage?
  private var toolCalls: [OpenToolCall] = []

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private var insideSection: Bool = false
  private var insideSingleCall: Bool = false

  /// Tracks whether we've already consumed an optional leading
  /// `<think>` opener in the reasoning phase. Once true, any further
  /// `<think>` text is treated as reasoning content rather than a
  /// marker.
  private var consumedThinkOpener: Bool = false

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
    /// Allocated lazily, once the function name has been parsed
    /// (truncation between `<|tool_call_begin|>` and the name leaves
    /// this nil so no slot is consumed).
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
    defer { pruneConsumedPrefix() }

    if phase == .reasoning {
      events.append(contentsOf: scanReasoning(isEnd: isEnd))
      if phase == .reasoning { return events }
    }
    events.append(contentsOf: emitNormalText(isEnd: isEnd))

    while parsedIdx < buffer.count {
      let slice = buffer.dropFirst(parsedIdx)
      if slice.hasPrefix(KimiK2Parser.sectionBegin) {
        parsedIdx += KimiK2Parser.sectionBegin.count
        insideSection = true
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        continue
      }
      if slice.hasPrefix(KimiK2Parser.sectionEnd) {
        parsedIdx += KimiK2Parser.sectionEnd.count
        insideSection = false
        continue
      }
      if slice.hasPrefix(KimiK2Parser.callBegin) {
        parsedIdx += KimiK2Parser.callBegin.count
        insideSingleCall = true
        toolCalls.append(OpenToolCall(
          id: IDFactory.make(.functionCall),
          callId: IDFactory.make(.callId),
        ))
        continue
      }
      if slice.hasPrefix(KimiK2Parser.callEnd) {
        parsedIdx += KimiK2Parser.callEnd.count
        insideSingleCall = false
        if let index = toolCalls.indices.last, !toolCalls[index].closed,
           toolCalls[index].name != nil
        {
          events.append(contentsOf: closeToolCall(at: index, status: .completed))
        }
        continue
      }

      if insideSingleCall, let last = toolCalls.indices.last, !toolCalls[last].closed {
        if toolCalls[last].name == nil {
          if !parseFunctionId(callIndex: last, events: &events) { return events }
          continue
        } else {
          let advanced = parseArguments(callIndex: last, isEnd: isEnd, events: &events)
          if advanced { continue } else { return events }
        }
      }

      if insideSection {
        if slice.first == "<" {
          // A `<` that's still ambiguous as a marker prefix (e.g.,
          // `<` alone, or `<|t`) must hold until more bytes
          // arrive. A `<` that can't grow into any marker (e.g.,
          // `<html>` inside dropped malformed-call args) is just
          // a content byte and we step past it.
          if couldStillBecomeATag(slice: String(slice)) {
            if !isEnd { return events }
          }
          parsedIdx += 1
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

  /// Reasoning phase scan. Emits reasoning deltas until either
  /// `</think>` or `<|tool_calls_section_begin|>` is observed; either
  /// terminates the reasoning block and transitions to the normal
  /// phase. A leading `<think>` opener is consumed (once) and not
  /// emitted as reasoning text. Mirrors vLLM's `KimiK2ReasoningParser`
  /// and SGLang's `KimiK2Detector`, both of which treat the tool-call
  /// section start as an implicit reasoning end.
  private mutating func scanReasoning(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    let bufChars = Array(buffer)

    // Strip an optional leading `<think>` opener exactly once. While
    // the buffer prefix is still a strict prefix of `<think>`, hold
    // rather than emit a partial-marker fragment as reasoning text.
    if !consumedThinkOpener {
      let openerChars = Array(KimiK2Parser.thinkStart)
      let here = Array(bufChars[parsedIdx ..< bufChars.count])
      if here.count >= openerChars.count {
        if Array(here.prefix(openerChars.count)) == openerChars {
          parsedIdx += openerChars.count
        }
        consumedThinkOpener = true
      } else if !isEnd, openerChars.starts(with: here) {
        return events
      } else {
        consumedThinkOpener = true
      }
    }

    let endChars = Array(KimiK2Parser.thinkEnd)
    let sectionChars = Array(KimiK2Parser.sectionBegin)

    let endIdx = bufChars.firstIndexOf(substring: KimiK2Parser.thinkEnd, after: parsedIdx)
    let sectionIdx = bufChars.firstIndexOf(substring: KimiK2Parser.sectionBegin, after: parsedIdx)

    let terminatorIdx: Int?
    let terminatorIsThinkEnd: Bool
    switch (endIdx, sectionIdx) {
      case let (e?, s?):
        if e <= s { terminatorIdx = e; terminatorIsThinkEnd = true }
        else { terminatorIdx = s; terminatorIsThinkEnd = false }
      case let (e?, nil): terminatorIdx = e; terminatorIsThinkEnd = true
      case let (nil, s?): terminatorIdx = s; terminatorIsThinkEnd = false
      case (nil, nil): terminatorIdx = nil; terminatorIsThinkEnd = false
    }

    let safeEnd: Int
    if let terminatorIdx {
      safeEnd = terminatorIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let overlap = Swift.max(
        partialOverlap(suffixOf: bufChars, with: endChars),
        partialOverlap(suffixOf: bufChars, with: sectionChars),
      )
      safeEnd = bufChars.count - overlap
    }

    if safeEnd > parsedIdx {
      let chunk = String(bufChars[parsedIdx ..< safeEnd])
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
      parsedIdx = safeEnd
    }

    guard let terminatorIdx, terminatorIdx == safeEnd else { return events }

    events.append(contentsOf: closeReasoning(status: .completed))
    if terminatorIsThinkEnd {
      // Consume `</think>`. Anything after it is normal-phase content
      // or the tool-call section, handled by `emitNormalText` /
      // tool-call scanning.
      parsedIdx += endChars.count
    }
    // For the implicit `<|tool_calls_section_begin|>` end, leave the
    // marker in place so the normal-phase scan picks it up as the
    // section opener.
    phase = .normal
    return events
  }

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

  /// Parse `functions.NAME:INDEX` (or `NAME:INDEX`) up to the
  /// `<|tool_call_argument_begin|>` separator. Returns true when the
  /// header was consumed and the cursor advanced to the start of args.
  private mutating func parseFunctionId(
    callIndex: Int,
    events: inout [ResponseStreamingEvent],
  ) -> Bool {
    let bufChars = Array(buffer)
    let argChars = Array(KimiK2Parser.argBegin)
    guard let argIdx = bufChars.firstIndexOf(substring: KimiK2Parser.argBegin, after: parsedIdx) else {
      return false
    }
    let header = String(bufChars[parsedIdx ..< argIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
    var name = header
    if name.hasPrefix("functions.") { name.removeFirst("functions.".count) }

    // The Kimi-K2 protocol requires a `:INDEX` (digit suffix) on the
    // function ID. An ID without it (e.g., `functions.foo.0`) is
    // malformed; skip the call rather than emitting it under a guessed
    // name. Because `outputIndex` is allocated lazily below, no slot
    // is consumed for the malformed call and downstream indexes stay
    // consecutive without recycling.
    guard let colon = name.firstIndex(of: ":") else {
      parsedIdx = argIdx + argChars.count
      toolCalls[callIndex].closed = true
      return true
    }
    let suffix = name[name.index(after: colon)...]
    guard !suffix.isEmpty, suffix.allSatisfy({ $0.isNumber }) else {
      parsedIdx = argIdx + argChars.count
      toolCalls[callIndex].closed = true
      return true
    }
    name = String(name[..<colon])

    // Mirror sglang's `tool_call_id_regex`: the name body must match
    // `[\w.\-]+` – Python's `\w` accepts Unicode word characters by
    // default, so we allow any letter or digit (Swift's `isLetter` and
    // `isNumber` cover the same set), plus underscore, dot, or hyphen.
    // Anything else (whitespace, slashes, special chars) is a malformed
    // ID; drop the call instead of forwarding garbage as a function name.
    guard !name.isEmpty, name.allSatisfy({ ch in
      ch.isLetter || ch.isNumber || ch == "_" || ch == "." || ch == "-"
    }) else {
      parsedIdx = argIdx + argChars.count
      toolCalls[callIndex].closed = true
      return true
    }

    var call = toolCalls[callIndex]
    call.name = name
    // The Kimi K2 chat template inserts `tool_call.id` verbatim into
    // `<|tool_call_begin|>{id}<|tool_call_argument_begin|>…`. Preserving
    // the wire-format header (`functions.NAME:INDEX` or `NAME:INDEX`)
    // here means re-rendering an assistant turn from history reproduces
    // the format the model was trained on. Mirrors vLLM's per-parser
    // `tool_call_id` capture.
    call.callId = header
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

    parsedIdx = argIdx + argChars.count
    return true
  }

  /// Stream argument bytes up to the next `<|tool_call_end|>` boundary.
  private mutating func parseArguments(
    callIndex: Int,
    isEnd: Bool,
    events: inout [ResponseStreamingEvent],
  ) -> Bool {
    let bufChars = Array(buffer)
    let endChars = Array(KimiK2Parser.callEnd)

    let endIdx = bufChars.firstIndexOf(substring: KimiK2Parser.callEnd, after: parsedIdx)
    let safeEnd: Int
    if let endIdx {
      safeEnd = endIdx
    } else if isEnd {
      safeEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: endChars)
      safeEnd = bufChars.count - overlap
    }

    let newArgs = String(bufChars[parsedIdx ..< safeEnd])
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
    parsedIdx = safeEnd
    return endIdx != nil
  }

  private mutating func emitNormalText(isEnd: Bool) -> [ResponseStreamingEvent] {
    if insideSection { return [] }

    let bufChars = Array(buffer)
    let beginChars = Array(KimiK2Parser.sectionBegin)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: KimiK2Parser.sectionBegin, after: parsedIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      // Hold back any trailing partial that could complete into one
      // of the Kimi markers (so a stray `<|tool_` doesn't leak as
      // content). All five Kimi tokens are checked, not just the
      // section opener.
      let overlap = Swift.max(
        partialOverlap(suffixOf: bufChars, with: beginChars),
        partialOverlap(suffixOf: bufChars, with: Array(KimiK2Parser.sectionEnd)),
        partialOverlap(suffixOf: bufChars, with: Array(KimiK2Parser.callBegin)),
        partialOverlap(suffixOf: bufChars, with: Array(KimiK2Parser.callEnd)),
        partialOverlap(suffixOf: bufChars, with: Array(KimiK2Parser.argBegin)),
      )
      sendableEnd = bufChars.count - overlap
    }
    guard sendableEnd > parsedIdx else { return [] }

    var chunk = String(bufChars[parsedIdx ..< sendableEnd])
    parsedIdx = sendableEnd
    // Strip any complete Kimi tokens that landed in plain content
    // (e.g., a stray `<|tool_call_begin|>` outside a section). Mirrors
    // sglang's `_strip_special_tokens`. Inside a section, the scan
    // loop consumes these as markers; outside they're noise that
    // shouldn't reach the consumer as text.
    chunk = stripStraySpecialTokens(from: chunk)
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

  private func stripStraySpecialTokens(from text: String) -> String {
    var out = text
    for token in [
      KimiK2Parser.sectionBegin,
      KimiK2Parser.sectionEnd,
      KimiK2Parser.callBegin,
      KimiK2Parser.callEnd,
      KimiK2Parser.argBegin,
    ] {
      out = out.replacingOccurrences(of: token, with: "")
    }
    return out
  }

  private func couldStillBecomeATag(slice: String) -> Bool {
    for tag in [
      KimiK2Parser.sectionBegin,
      KimiK2Parser.sectionEnd,
      KimiK2Parser.callBegin,
      KimiK2Parser.callEnd,
      KimiK2Parser.argBegin,
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
