// Copyright © Anthony DePasquale

import Foundation

/// Parser for the DeepSeek V3.2 base tool-call format (DSML).
///
/// **Wire shape.** V3.2 uses an XML-like DSML envelope distinct from V3.1:
///
/// ```text
/// <｜DSML｜function_calls>
///   <｜DSML｜invoke name="get_weather">
///     <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
///     <｜DSML｜parameter name="days" string="false">5</｜DSML｜parameter>
///   </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// ```
///
/// Or with a direct JSON body inside `<｜DSML｜invoke>`:
///
/// ```text
/// <｜DSML｜function_calls>
///   <｜DSML｜invoke name="get_weather">
///     {"city":"Paris","days":5}
///   </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// ```
///
/// **Parameter typing.** The `string=` attribute on a `<｜DSML｜parameter>`
/// tag is required; both vLLM and sglang reject parameters whose tag is
/// missing it. `string="true"` declares the value as a literal string;
/// any other `string=` value (`"false"` or otherwise) signals that the
/// body is a JSON literal – numbers, booleans, arrays, objects, and null
/// pass through as is. Mirrors sglang's permissive value dispatch (vLLM
/// is stricter and only accepts `"true"`/`"false"`).
///
/// **Truncation.** A `<｜DSML｜invoke>` whose `</｜DSML｜invoke>` never
/// arrives is silently dropped, matching both references – vLLM's
/// `_extract_delta_tool_calls` only emits via `invoke_complete_regex`,
/// and sglang's `parse_streaming_increment` waits for more chunks rather
/// than surfacing partial calls.
///
/// **Optional reasoning preamble.** When constructed with
/// ``acceptThink`` true, a leading `<think>...</think>` block is
/// extracted as a reasoning item before the tool-call scan. Mirrors
/// vLLM's `DeepSeekV3ReasoningParser` with `chat_template_kwargs.thinking=True`,
/// which delegates to the R1 reasoning shape.
struct DeepSeekV32Parser: ResponseFormatParser {
  /// Initial reasoning phase. Used by continuation requests on
  /// thinking-enabled checkpoints whose `priorOutput` ended either
  /// inside or after the `<think>...</think>` block.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private static let envelopeStart = "<｜DSML｜function_calls>"
  private static let envelopeEnd = "</｜DSML｜function_calls>"
  private static let invokeStart = "<｜DSML｜invoke"
  private static let invokeEnd = "</｜DSML｜invoke>"
  private static let parameterStart = "<｜DSML｜parameter"
  private static let parameterEnd = "</｜DSML｜parameter>"

  /// Active suffix that has not yet been proven safe to discard.
  private var buffer: String = ""
  private var sentContentIdx: Int = 0

  private var openMessage: OpenMessage?

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private let acceptThink: Bool
  private var thinkPreamble: ThinkPreambleExtractor

  /// Cursor (offset into `buffer`) past the last `</｜DSML｜invoke>` we've
  /// emitted, or past the envelope-start when no invoke has emitted yet.
  /// Nil when we're outside the function_calls envelope. Mirrors vLLM's
  /// `current_tool_index`-driven progression: each complete invoke is
  /// emitted as soon as `</｜DSML｜invoke>` arrives, without waiting for
  /// the outer `</｜DSML｜function_calls>` envelope to close.
  private var envelopeCursor: Int?

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
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
    if acceptThink {
      events.append(contentsOf: thinkPreamble.finalizeIfOpen(nextSequence: &nextSequence))
    }
    return events
  }

  // MARK: Scan loop

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    defer { pruneConsumedPrefix() }

    let bufChars = Array(buffer)
    let envStartChars = Array(DeepSeekV32Parser.envelopeStart)
    let envEndChars = Array(DeepSeekV32Parser.envelopeEnd)
    let invokeEndChars = Array(DeepSeekV32Parser.invokeEnd)

    while sentContentIdx < bufChars.count || envelopeCursor != nil {
      if envelopeCursor == nil {
        // Outside the envelope. Emit message content up to the
        // next `<｜DSML｜function_calls>`, holding back any
        // partial-marker suffix. `flushContent` advances
        // `sentContentIdx` to either the envelope start or the
        // safe content boundary.
        events.append(contentsOf: flushContent(isEnd: isEnd))

        // If we're now sitting on the envelope-start, enter the
        // envelope. Otherwise we're either still buffering or at
        // EOS without an envelope – exit the loop.
        let nextStart = bufChars.firstIndexOf(
          substring: DeepSeekV32Parser.envelopeStart, after: sentContentIdx,
        )
        guard let nextStart, nextStart == sentContentIdx else { break }
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        sentContentIdx = nextStart + envStartChars.count
        envelopeCursor = sentContentIdx
        continue
      }

      // Inside the envelope. Emit each invoke as `</｜DSML｜invoke>`
      // arrives, tracking position via `envelopeCursor`. Whichever
      // of `</｜DSML｜invoke>` or `</｜DSML｜function_calls>` is
      // encountered first decides the next step.
      guard let cursor = envelopeCursor else { break }
      let nextInvokeEnd = bufChars.firstIndexOf(
        substring: DeepSeekV32Parser.invokeEnd, after: cursor,
      )
      let nextEnvelopeEnd = bufChars.firstIndexOf(
        substring: DeepSeekV32Parser.envelopeEnd, after: cursor,
      )

      if let envEnd = nextEnvelopeEnd,
         nextInvokeEnd == nil || envEnd < nextInvokeEnd!
      {
        // Envelope closes before the next invoke-end. We're done
        // with this envelope.
        sentContentIdx = envEnd + envEndChars.count
        envelopeCursor = nil
        continue
      }

      guard let invokeEnd = nextInvokeEnd else {
        // No invoke close yet. Hold for more bytes (or, at EOS,
        // simply stop – any unclosed invoke is dropped, matching
        // vLLM's streaming behavior which only emits on a
        // complete `</｜DSML｜invoke>`).
        break
      }

      // Parse the invoke whose body runs from `cursor` up to
      // `invokeEnd`, then emit it. We pass the whole region
      // (including the open tag) to the existing extractor.
      let regionEnd = invokeEnd + invokeEndChars.count
      let region = String(bufChars[cursor ..< regionEnd])
      for invoke in extractInvokes(from: region) {
        events.append(contentsOf: emitToolCall(invoke: invoke))
      }
      envelopeCursor = regionEnd
    }

    return events
  }

  private mutating func pruneConsumedPrefix() {
    let dropCount = min(envelopeCursor ?? sentContentIdx, buffer.count)
    guard dropCount > 0 else { return }

    buffer.removeFirst(dropCount)
    sentContentIdx = max(0, sentContentIdx - dropCount)
    if let cursor = envelopeCursor {
      envelopeCursor = max(0, cursor - dropCount)
    }
  }

  private mutating func flushContent(isEnd: Bool) -> [ResponseStreamingEvent] {
    let bufChars = Array(buffer)
    let openChars = Array(DeepSeekV32Parser.envelopeStart)

    let sendableEnd: Int
    if let startIdx = bufChars.firstIndexOf(substring: DeepSeekV32Parser.envelopeStart, after: sentContentIdx) {
      sendableEnd = startIdx
    } else if isEnd {
      sendableEnd = bufChars.count
    } else {
      let overlap = partialOverlap(suffixOf: bufChars, with: openChars)
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
        itemId: msg.id, outputIndex: msg.outputIndex, contentIndex: 0,
        delta: chunk, sequenceNumber: takeSequence(),
      )))
    }
    return events
  }

  // MARK: Invoke extraction

  private struct ParsedInvoke {
    let name: String
    let argumentsJSON: String
  }

  private func extractInvokes(from envelope: String) -> [ParsedInvoke] {
    var results: [ParsedInvoke] = []
    var cursor = envelope.startIndex
    while let openRange = envelope.range(of: DeepSeekV32Parser.invokeStart, range: cursor ..< envelope.endIndex) {
      guard let openTagEnd = envelope.range(of: ">", range: openRange.upperBound ..< envelope.endIndex) else {
        break
      }
      guard let closeRange = envelope.range(of: DeepSeekV32Parser.invokeEnd, range: openTagEnd.upperBound ..< envelope.endIndex) else {
        break
      }
      let openTagAttrs = String(envelope[openRange.upperBound ..< openTagEnd.lowerBound])
      let inner = String(envelope[openTagEnd.upperBound ..< closeRange.lowerBound])
      // Drop empty/missing names – surfacing a function call with
      // `name: ""` to consumers is more harmful than a silent skip.
      // Same defensive choice we make in V3.1's parseFunctionHeader.
      if let name = extractAttribute(named: "name", from: openTagAttrs),
         !name.isEmpty
      {
        let argsJSON = renderInvokeArgs(inner: inner)
        results.append(ParsedInvoke(name: name, argumentsJSON: argsJSON))
      }
      cursor = closeRange.upperBound
    }
    return results
  }

  private func renderInvokeArgs(inner: String) -> String {
    let trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines)
    // Direct JSON body
    if trimmed.hasPrefix("{"), trimmed.hasSuffix("}") {
      if isValidJSON(trimmed) {
        return trimmed
      }
    }
    // XML parameter tags
    var pairs: [(key: String, json: String)] = []
    var cursor = inner.startIndex
    while let openRange = inner.range(of: DeepSeekV32Parser.parameterStart, range: cursor ..< inner.endIndex) {
      guard let openTagEnd = inner.range(of: ">", range: openRange.upperBound ..< inner.endIndex) else {
        break
      }
      guard let closeRange = inner.range(of: DeepSeekV32Parser.parameterEnd, range: openTagEnd.upperBound ..< inner.endIndex) else {
        break
      }
      let openTagAttrs = String(inner[openRange.upperBound ..< openTagEnd.lowerBound])
      let body = String(inner[openTagEnd.upperBound ..< closeRange.lowerBound])
      let bodyTrimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
      // Both vLLM and sglang require the `string=` attribute on
      // every `<｜DSML｜parameter>` tag – vLLM's regex hard-codes
      // `string="(?:true|false)"`, sglang's requires `string="..."`
      // to be present. A parameter without the attribute is dropped
      // by both references; we mirror that here. Within the value,
      // we follow sglang's dispatch: `"true"` → string mode, anything
      // else → JSON literal (vLLM additionally rejects values other
      // than `"true"`/`"false"`, but sglang's permissiveness is the
      // documented Swift behavior).
      if let paramName = extractAttribute(named: "name", from: openTagAttrs),
         !paramName.isEmpty,
         let stringFlag = extractAttribute(named: "string", from: openTagAttrs)
      {
        let json: String = if stringFlag == "true" {
          jsonEncodeString(bodyTrimmed)
        } else {
          if isValidJSON(bodyTrimmed) {
            bodyTrimmed
          } else {
            jsonEncodeString(bodyTrimmed)
          }
        }
        pairs.append((paramName, json))
      }
      cursor = closeRange.upperBound
    }
    var json = "{"
    for (i, pair) in pairs.enumerated() {
      if i > 0 { json += ", " }
      json += jsonEncodeString(pair.key) + ": " + pair.json
    }
    json += "}"
    return json
  }

  private func extractAttribute(named attr: String, from tagAttrs: String) -> String? {
    // Mirror vLLM/sglang: case-sensitive, double-quoted, at-least-one-char.
    // vLLM: `name="([^"]+)"`. sglang: `name="([^"]+)"`. Neither accepts
    // single quotes or zero-length values.
    let pattern = #"\#(attr)\s*=\s*"([^"]+)""#
    guard let regex = try? NSRegularExpression(pattern: pattern) else {
      return nil
    }
    let range = NSRange(tagAttrs.startIndex..., in: tagAttrs)
    if let match = regex.firstMatch(in: tagAttrs, range: range), match.numberOfRanges >= 2 {
      if let r = Range(match.range(at: 1), in: tagAttrs) {
        return String(tagAttrs[r])
      }
    }
    return nil
  }

  // MARK: Emission

  private mutating func emitToolCall(invoke: ParsedInvoke) -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.functionCall)
    let callId = IDFactory.make(.callId)
    let outputIndex = takeOutputIndex()
    let openItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: invoke.name, arguments: "", status: .inProgress,
    )
    let doneItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: invoke.name, arguments: invoke.argumentsJSON, status: .completed,
    )
    return [
      .outputItemAdded(.init(
        item: .functionCall(openItem),
        outputIndex: outputIndex,
        sequenceNumber: takeSequence(),
      )),
      .functionCallArgumentsDelta(.init(
        itemId: id, outputIndex: outputIndex,
        delta: invoke.argumentsJSON, sequenceNumber: takeSequence(),
      )),
      .functionCallArgumentsDone(.init(
        itemId: id, outputIndex: outputIndex,
        arguments: invoke.argumentsJSON,
        sequenceNumber: takeSequence(),
      )),
      .outputItemDone(.init(
        item: .functionCall(doneItem),
        outputIndex: outputIndex,
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

  private func jsonEncodeString(_ s: String) -> String {
    guard let data = try? JSONSerialization.data(
      withJSONObject: [s], options: [.withoutEscapingSlashes],
    ) else {
      return "\"\(s)\""
    }
    if let str = String(data: data, encoding: .utf8) {
      return String(str.dropFirst().dropLast())
    }
    return "\"\(s)\""
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
