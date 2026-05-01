// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Hermes-2 / Nous tool-call format.
///
/// **Wire shape.** Tool calls are wrapped in `<tool_call>` … `</tool_call>`
/// XML tags, with a JSON object in between:
///
/// ```text
/// <tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>
/// ```
///
/// Plain text outside the tags is normal message content. Multiple tool
/// calls in one response are emitted in sequence with their own
/// `<tool_call>` wrapper each.
///
/// **Streaming algorithm.** On each call to ``process(_:)``, the parser
/// appends the new text to a full-output buffer, then re-scans the buffer
/// to find all `<tool_call>` regions and diffs them against per-tool
/// emitted state to produce only the new content / name / argument-fragment
/// events.
///
/// **Reasoning extraction is not in this parser.** The Hermes-2 and Nous
/// model families do not emit `<think>` reasoning markers; the Qwen 2.5 /
/// Qwen 3 base parser handles Hermes-style tool calls plus think-tag
/// reasoning together.
struct HermesParser: ResponseFormatParser {
  private let toolCallStart: String
  private let toolCallEnd: String
  private let argumentsMayBeJSONString: Bool

  // Full accumulated output. Re-scanned on every process() call.
  private var buffer: String = ""

  // Index into `buffer` of the last character that has been emitted as
  // plain (non-tool-call) message content.
  private var sentContentIdx: Int = 0

  // Open message item that buffered content gets appended to. nil means
  // no message has been opened yet (no plain text seen so far).
  private var openMessage: OpenMessage?

  // Per-tool-call state, in the order the tool calls appear in `buffer`.
  private var toolCalls: [OpenToolCall] = []

  // Sequential output_index assignment: messages count as one slot,
  // each tool_call counts as the next slot.
  private var nextOutputIndex: Int = 0

  // Parser-local sequence numbers. The driver (``ResponseStreamEmitter``)
  // substitutes response-scoped numbers in their place.
  private var nextSequence: Int = 0

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    /// Total characters of message text emitted so far via
    /// `output_text.delta`. Used to compute the closing
    /// `output_text.done` payload.
    var emittedText: String = ""
  }

  private struct OpenToolCall {
    var id: String
    var callId: String
    /// Allocated lazily, once the tool name has been parsed and we're
    /// about to emit `output_item.added`. Truncation before the name
    /// arrives leaves this nil so no slot is consumed and the next
    /// item's index stays consecutive.
    var outputIndex: Int?
    var name: String?
    /// Characters of the `arguments` value already emitted as
    /// `function_call_arguments.delta`.
    var argsEmitted: String = ""
    /// Whether `output_item.done` has been emitted.
    var closed: Bool = false
  }

  /// - Parameters:
  ///   - toolCallStart: The opening envelope token. Default
  ///     `<tool_call>` for Hermes / Nous; pass `<longcat_tool_call>`
  ///     for Meituan LongCat (mirrors vLLM's
  ///     `LongcatFlashToolParser`).
  ///   - toolCallEnd: The closing envelope token. Must pair with
  ///     `toolCallStart`.
  ///   - argumentsMayBeJSONString: When true, accept a wire form
  ///     where the `arguments` value is a JSON-encoded string instead
  ///     of an object (e.g. `"arguments": "{\"x\":1}"`). The parser
  ///     decodes the string at call close so the emitted `arguments`
  ///     text always matches the canonical object form. Mirrors
  ///     vLLM's `Granite4ToolParser` `dump_args` behavior. Mid-call
  ///     argument-delta streaming is suppressed for the string-encoded
  ///     case since the wire bytes don't match the canonical bytes.
  init(
    toolCallStart: String = "<tool_call>",
    toolCallEnd: String = "</tool_call>",
    argumentsMayBeJSONString: Bool = false,
    startingOutputIndex: Int = 0,
  ) {
    self.toolCallStart = toolCallStart
    self.toolCallEnd = toolCallEnd
    self.argumentsMayBeJSONString = argumentsMayBeJSONString
    nextOutputIndex = startingOutputIndex
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return scan()
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = scan(isEnd: true)
    // Close anything still open at EOS.
    if let _ = openMessage {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    for index in toolCalls.indices where !toolCalls[index].closed {
      // An open tool call at EOS is truncated. The scan() at end of
      // buffer already emitted the best-effort name + partial args;
      // close out as incomplete so the consumer can see the
      // truncation signal.
      events.append(contentsOf: closeToolCall(at: index, status: .incomplete))
    }
    return events
  }

  // MARK: Scan loop

  /// Walk the buffer from the last consumed position forward, emitting
  /// content and tool-call events.
  private mutating func scan(isEnd: Bool = false) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []

    // Process every tool-call region the buffer currently shows. Plain
    // text is flushed before each region so any text that landed
    // between two consecutive `<tool_call>...</tool_call>` blocks in a
    // single chunk is emitted as a message before the next region's
    // events. A trailing flush handles text after the last closed
    // region.
    let regions = extractToolCallRegions()
    for (index, region) in regions.enumerated() {
      events.append(contentsOf: flushContent(isEnd: isEnd))
      if index >= toolCalls.count {
        // First time we've seen this region – track it without
        // taking an output_index slot yet. Allocation happens
        // lazily once the name is parsed (see `processRegion`).
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

  /// Emit any plain content (text outside tool-call regions) that is safe
  /// to forward. Content is "safe" up to either the next `<tool_call>` start
  /// or, when no start exists in the buffer, up to the buffer end minus any
  /// suffix that could be a partial `<tool_call>` or `</tool_call>` marker.
  /// A stray `</tool_call>` literal (without a matching opener) is stripped
  /// from the emitted chunk; mirrors sglang's `_clean_normal_text`.
  private mutating func flushContent(isEnd: Bool) -> [ResponseStreamingEvent] {
    let bufChars = Array(buffer)
    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: toolCallStart, after: sentContentIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      // Hold back any partial-tag overlap on either marker so a
      // chunk ending in `<tool_cal` or `</tool_cal` doesn't leak
      // raw bytes that would later complete to a real marker.
      let openOverlap = partialOverlap(suffixOf: bufChars, with: Array(toolCallStart))
      let closeOverlap = partialOverlap(suffixOf: bufChars, with: Array(toolCallEnd))
      sendableEnd = bufChars.count - Swift.max(openOverlap, closeOverlap)
    }
    guard sendableEnd > sentContentIdx else { return [] }

    var chunk = String(bufChars[sentContentIdx ..< sendableEnd])
    sentContentIdx = sendableEnd
    // Strip any stray `</tool_call>` literal that landed in plain
    // content. The cursor-advance in `processRegion` already skips
    // past close tags that legitimately follow open tags; this
    // handles bare close tags emitted by the model on their own.
    if chunk.contains(toolCallEnd) {
      chunk = chunk.replacingOccurrences(of: toolCallEnd, with: "")
    }
    if chunk.isEmpty { return [] }

    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem())
    }
    // Append delta to the open message.
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

  /// Extract the JSON region between each `<tool_call>` and (optionally)
  /// its matching `</tool_call>`. Each entry carries the JSON text, a
  /// flag for whether the JSON is structurally parseable, and – when
  /// present – the buffer position right after the closing tag (used to
  /// advance the content cursor and gate the close-events emission).
  private struct ToolCallRegion {
    var jsonText: String
    var isComplete: Bool
    var endIdxAfterClose: String.Index?
  }

  private func extractToolCallRegions() -> [ToolCallRegion] {
    var results: [ToolCallRegion] = []
    var pos = buffer.startIndex
    while let startRange = buffer.range(of: toolCallStart, range: pos ..< buffer.endIndex) {
      let jsonStart = startRange.upperBound
      if let endRange = buffer.range(of: toolCallEnd, range: jsonStart ..< buffer.endIndex) {
        let inner = buffer[jsonStart ..< endRange.lowerBound]
        results.append(ToolCallRegion(
          jsonText: inner.trimmingCharacters(in: .whitespacesAndNewlines),
          isComplete: true,
          endIdxAfterClose: endRange.upperBound,
        ))
        pos = endRange.upperBound
      } else {
        // Open tool call: JSON runs to end of buffer, minus any
        // partial `</tool_call>` suffix.
        var raw = String(buffer[jsonStart ..< buffer.endIndex])
        let endChars = Array(toolCallEnd)
        let overlap = partialOverlap(suffixOf: Array(raw), with: endChars)
        if overlap > 0 {
          raw = String(raw.dropLast(overlap))
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let isComplete = !trimmed.isEmpty && isValidJSON(trimmed)
        results.append(ToolCallRegion(
          jsonText: trimmed, isComplete: isComplete, endIdxAfterClose: nil,
        ))
        break
      }
    }
    return results
  }

  private mutating func processRegion(
    at index: Int,
    region: ToolCallRegion,
    isEnd: Bool,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var call = toolCalls[index]

    // Close any open message before opening the first tool call event,
    // so the message item's lifecycle ends before the tool's begins.
    if openMessage != nil, call.name == nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }

    // Emit `output_item.added` once, when the name is first available.
    // Allocate the output_index here too so a truncated header (no
    // name yet) doesn't reserve a permanent gap.
    if call.name == nil {
      if let name = extractToolName(from: region.jsonText) {
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
      }
    }

    // Stream any new arguments chars.
    if let outputIndex = call.outputIndex, call.name != nil {
      if let argsSoFar = extractArgumentsText(from: region.jsonText, isComplete: region.isComplete) {
        let canonicalArgs = canonicalizeArgs(
          argsSoFar,
          isComplete: region.isComplete,
        )
        if canonicalArgs.count > call.argsEmitted.count {
          let diffStart = canonicalArgs.index(canonicalArgs.startIndex, offsetBy: call.argsEmitted.count)
          let diff = String(canonicalArgs[diffStart...])
          call.argsEmitted = canonicalArgs
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

    // Emit close events when the closing tag has been seen (or the
    // region is structurally complete and we're at end of input).
    let regionClosed = region.endIdxAfterClose != nil
    if !call.closed, regionClosed || (isEnd && region.isComplete) {
      events.append(contentsOf: closeToolCall(at: index, status: .completed))
      // Advance the content cursor past `</tool_call>` so any
      // trailing text (`<tool_call>…</tool_call> trailing message`)
      // is picked up as a fresh message on the next scan.
      if let endIdx = region.endIdxAfterClose {
        let endOffset = buffer.distance(from: buffer.startIndex, to: endIdx)
        if endOffset > sentContentIdx {
          sentContentIdx = endOffset
        }
      }
    }

    return events
  }

  /// Apply Granite-4-style decoding when the `arguments` field is a
  /// JSON-encoded string instead of an object. For object-shaped args
  /// (or when the variant flag is off), pass through unchanged.
  ///
  /// When the variant is active and the args value begins with `"`, the
  /// wire bytes are a JSON string literal. Mid-stream the literal isn't
  /// fully written yet, so the canonical form is unknown – return the
  /// empty string so no delta is emitted; the close path will surface
  /// the decoded value in one shot. When the literal is complete, decode
  /// it and return the unescaped contents (typically the inner JSON
  /// object text).
  private func canonicalizeArgs(_ argsSoFar: String, isComplete: Bool) -> String {
    guard argumentsMayBeJSONString,
          let firstChar = argsSoFar.drop(while: { $0.isWhitespace }).first,
          firstChar == "\""
    else {
      return argsSoFar
    }
    guard isComplete,
          let data = argsSoFar.data(using: .utf8),
          let decoded = try? JSONSerialization.jsonObject(
            with: data, options: [.fragmentsAllowed],
          ) as? String
    else {
      return ""
    }
    return decoded
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
    let finalText = msg.emittedText
    let part = ResponseOutputText(text: finalText)
    return [
      .outputTextDone(.init(
        itemId: msg.id,
        outputIndex: msg.outputIndex,
        contentIndex: 0,
        text: finalText,
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
