// Copyright © Anthony DePasquale

import Foundation

/// Parser for Cohere Command-family wire formats (cmd3 / cmd4).
///
/// **Wire shape.** Cohere Command R7B and Command A Reasoning emit a
/// marker-driven sequence of region kinds:
///
/// - `<|START_THINKING|>` … `<|END_THINKING|>` — chain-of-thought
///   reasoning, streamed as a `reasoning` item.
/// - `<|START_ACTION|>` … `<|END_ACTION|>` — a JSON array of tool calls
///   shaped `{ "tool_call_id", "tool_name", "parameters" }`, each
///   element becomes its own `function_call` item.
/// - `<|START_RESPONSE|>` … `<|END_RESPONSE|>` (cmd3) or additionally
///   `<|START_TEXT|>` … `<|END_TEXT|>` (cmd4 — used by Cohere2 Vision)
///   — grounded answer, streamed as a `message` item with inline
///   `<co>…</co: …>` citations.
///
/// The model may interleave these regions within a single turn, so each
/// `THINKING` / `ACTION` / response region opens a fresh top-level item.
///
/// **cmd3 vs cmd4.** cmd4 is a superset of cmd3 — it registers all the
/// same markers plus `<|START_TEXT|>` / `<|END_TEXT|>`. The other
/// difference is the default initial state: cmd4 chat templates can
/// pre-inject `<|START_THINKING|>` via the `response_prefix` variable, so
/// a freshly-constructed cmd4 parser may start inside a reasoning block
/// rather than at rest. The factory passes that decision in via
/// ``CohereParser/InitialState``.
///
/// **Citations.** Grounded answers contain inline citation tags
/// `<co>span</co: 0:[1,2],1:[0]>`. The `<co>` (bare) and `<co: …>`
/// (legacy) open forms are both recognized; the trailing source list on a
/// legacy open is ignored — the close tag is authoritative. Each
/// `(tool_call_index, tool_result_indices)` group becomes a separate
/// ``ResponseOutputText/Annotation/cohereToolResultCitation(toolCallIndex:toolResultIndices:startIndex:endIndex:)``
/// annotation, attached to the surrounding `message` item's
/// `output_text` content part. UTF-16 indices match OpenAI's
/// `url_citation` convention.
///
/// **Streaming.** Tool calls follow the buffer-then-emit pattern used
/// elsewhere in this codebase (``JambaParser``, ``GraniteParser``):
/// accumulate the JSON array until `<|END_ACTION|>` arrives, then emit
/// one `arguments.delta` per call. Per-byte streaming of arguments —
/// melody's `raw-param` mode — is not implemented in v1.
struct CohereParser: ResponseFormatParser {
  /// Marker preset. ``cmd3`` covers Command R7B and Command A
  /// Reasoning; ``cmd4`` adds `<|START_TEXT|>` / `<|END_TEXT|>` for
  /// Cohere2 Vision multimodal text content.
  enum Variant: Equatable {
    case cmd3
    case cmd4
  }

  /// Initial parser state. ``groundedAnswer`` matches melody's cmd3
  /// `default_mode = GroundedAnswer`. ``reasoning`` matches cmd4's
  /// `default_mode = ToolReason` when the rendered prompt injected
  /// `<|START_THINKING|>`. The factory resolves the value by scanning
  /// `priorOutput`.
  enum InitialState: Equatable {
    case groundedAnswer
    case reasoning
  }

  // Marker strings. The full set is registered for cmd4; the cmd3
  // marker table is a subset (no `START_TEXT` / `END_TEXT`).
  private static let startThinking = "<|START_THINKING|>"
  private static let endThinking = "<|END_THINKING|>"
  private static let startAction = "<|START_ACTION|>"
  private static let endAction = "<|END_ACTION|>"
  private static let startResponse = "<|START_RESPONSE|>"
  private static let endResponse = "<|END_RESPONSE|>"
  private static let startText = "<|START_TEXT|>"
  private static let endText = "<|END_TEXT|>"

  private static let citationOpenPrefix = "<co"
  private static let citationCloseStart = "</co: "

  private enum Mode {
    case groundedAnswer
    case reasoning
    case toolAction
    case ignored
  }

  private var mode: Mode
  private let markerTable: [(marker: String, target: Mode)]
  private let markers: [String]

  private var buffer: String = ""
  private var actionBuffer: String = ""
  private var openMessage: OpenMessage?
  private var openReasoning: OpenReasoning?

  /// True while skipping leading whitespace on entry to a fresh
  /// reasoning region. Mirrors melody's `left_trimmed = true` reset in
  /// `apply_special_token_match` for the `ToolReason` transition.
  private var skipLeadingReasoningWhitespace: Bool = false

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
    /// UTF-16 length of `emittedText`. Maintained incrementally so
    /// every citation annotation can stamp its `startIndex` / `endIndex`
    /// in O(1) instead of recomputing the length each time.
    var utf16Length: Int = 0
    /// Annotations collected from `<co>…</co: …>` citations, in
    /// emission order. Carried on the parser so the terminal
    /// `output_item.done` payload survives the per-event accumulator's
    /// item-slot replacement; the streaming
    /// `output_text.annotation.added` deltas alone would be wiped at
    /// `output_item.done` time.
    var annotations: [ResponseOutputText.Annotation] = []
  }

  private struct OpenReasoning {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  init(variant: Variant, initialState: InitialState = .groundedAnswer) {
    switch initialState {
      case .groundedAnswer: mode = .groundedAnswer
      case .reasoning:
        mode = .reasoning
        // cmd4's `ToolReason` default sets `left_trimmed = true`; do
        // the same so a leading `\n  ` after the injected
        // `<|START_THINKING|>` doesn't appear in reasoning output.
        skipLeadingReasoningWhitespace = true
    }

    var table: [(marker: String, target: Mode)] = [
      (Self.startThinking, .reasoning),
      (Self.endThinking, .groundedAnswer),
      (Self.startAction, .toolAction),
      (Self.endAction, .ignored),
      (Self.startResponse, .groundedAnswer),
      (Self.endResponse, .ignored),
    ]
    if variant == .cmd4 {
      table.append((Self.startText, .groundedAnswer))
      table.append((Self.endText, .ignored))
    }
    markerTable = table
    markers = table.map(\.marker)
  }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    if chunk.text.isEmpty { return [] }
    buffer += chunk.text
    return scan(isEnd: false)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events = scan(isEnd: true)
    if mode == .toolAction, !actionBuffer.isEmpty {
      // EOS reached mid-action without a closing `<|END_ACTION|>`.
      // Best-effort parse: if the partial buffer is a valid JSON
      // array, emit the calls; otherwise drop. Truncated tool calls
      // have no spec-valid surface in our event stream.
      events.append(contentsOf: emitToolCalls(from: actionBuffer))
      actionBuffer = ""
    }
    if openReasoning != nil {
      events.append(contentsOf: closeReasoning(status: .incomplete))
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

      if let markerHit = findEarliestMarker() {
        // Emit pre-marker bytes according to current mode.
        if markerHit.startIndex != buffer.startIndex {
          let pre = String(buffer[buffer.startIndex ..< markerHit.startIndex])
          events.append(contentsOf: process(text: pre, isFinalSegment: true, isEnd: false))
        }
        // Transitioning out of a state with held resources flushes them.
        events.append(contentsOf: closePreviousMode(target: markerHit.target))
        // Consume the marker.
        buffer = String(buffer[markerHit.endIndex...])
        mode = markerHit.target
        if mode == .reasoning {
          skipLeadingReasoningWhitespace = true
        }
        didProgress = true
        continue
      }

      // No complete marker visible. Hold back any tail that's a
      // partial prefix of a marker so it survives to the next chunk;
      // process the rest in the current mode.
      if buffer.isEmpty { return events }
      let (safe, hold) = isEnd
        ? (buffer, "")
        : PrefixHold.split(text: buffer, markers: markers)
      buffer = hold
      if !safe.isEmpty {
        events.append(contentsOf: process(text: safe, isFinalSegment: false, isEnd: isEnd))
      }
      // `process` may re-buffer a citation tail by prepending to
      // `buffer`; leave it intact.
      return events
    }
    return events
  }

  /// Locate the earliest complete marker in the current buffer.
  /// Returns nil when none is found.
  private func findEarliestMarker()
    -> (startIndex: String.Index, endIndex: String.Index, target: Mode)?
  {
    var earliest: (startIndex: String.Index, endIndex: String.Index, target: Mode)?
    for (marker, target) in markerTable {
      guard let range = buffer.range(of: marker) else { continue }
      if let cur = earliest, range.lowerBound >= cur.startIndex { continue }
      earliest = (range.lowerBound, range.upperBound, target)
    }
    return earliest
  }

  // MARK: Per-mode content processing

  /// Process `text` (which contains no complete marker) according to
  /// the current mode. `isFinalSegment` is true when `text` is followed
  /// by a marker in this same scan iteration — partial-marker hold and
  /// trailing-whitespace hold are disabled in that case (the marker
  /// either fully proves what came before is content, or — for an
  /// `<|END_*|>` — entitles us to drop the held whitespace). `isEnd`
  /// signals the surrounding stream has terminated.
  private mutating func process(
    text: String,
    isFinalSegment: Bool,
    isEnd: Bool,
  ) -> [ResponseStreamingEvent] {
    switch mode {
      case .groundedAnswer:
        return processCitedText(text, isFinalSegment: isFinalSegment, isEnd: isEnd, asReasoning: false)
      case .reasoning:
        return processCitedText(text, isFinalSegment: isFinalSegment, isEnd: isEnd, asReasoning: true)
      case .toolAction:
        actionBuffer += text
        return []
      case .ignored:
        return []
    }
  }

  // MARK: Citation-aware text scanner

  /// Scan `text` for `<co>…</co: …>` citation pairs, emit the
  /// surrounding and inner text via the current mode's content path,
  /// and (for grounded text) emit one annotation per source group on
  /// the close tag. Mirrors melody's `parse_citations`, run for both
  /// `GroundedAnswer` and `ToolReason` — the only difference being
  /// that `ToolReason` suppresses annotations because the Open
  /// Responses spec has no annotation home on a reasoning item.
  ///
  /// `isFinalSegment` is true when this segment ends at a marker — so
  /// the trailing-whitespace hold and partial-marker hold are
  /// disabled. `isEnd` signals stream termination, which lets any
  /// held-back tail flush as content.
  private mutating func processCitedText(
    _ text: String,
    isFinalSegment: Bool,
    isEnd: Bool,
    asReasoning: Bool,
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var remaining = text

    // Reasoning entry trims leading whitespace.
    if asReasoning, skipLeadingReasoningWhitespace {
      let trimmed = remaining.drop(while: { $0.isWhitespace })
      if !trimmed.isEmpty || isFinalSegment || isEnd {
        skipLeadingReasoningWhitespace = false
      }
      remaining = String(trimmed)
    }

    while !remaining.isEmpty {
      if let openRange = remaining.range(of: Self.citationOpenPrefix) {
        // Emit pre-citation text.
        let pre = String(remaining[remaining.startIndex ..< openRange.lowerBound])
        if !pre.isEmpty {
          events.append(contentsOf: emitContent(pre, asReasoning: asReasoning, rightTrim: .none))
        }
        let citationSlice = String(remaining[openRange.lowerBound...])
        switch parseCitation(in: citationSlice, isEnd: isEnd || isFinalSegment) {
          case let .complete(consumed, spanText, sources):
            if asReasoning {
              // Reasoning: emit the inner span as text, drop the
              // markers and annotations.
              if !spanText.isEmpty {
                events.append(contentsOf: emitContent(spanText, asReasoning: true, rightTrim: .none))
              }
            } else {
              events.append(contentsOf: emitCitation(spanText: spanText, sources: sources))
            }
            remaining = String(citationSlice.dropFirst(consumed))
          case let .openOnly(consumed):
            // Open tag observed in full, no close found before the
            // segment ends. Strip the open and emit the post-open
            // text — mirrors melody's `get_partial_citation_text`
            // behavior for bare cmd3 opens.
            remaining = String(citationSlice.dropFirst(consumed))
          case .partial:
            // Wait for more data.
            buffer = citationSlice + buffer
            return events
        }
        continue
      }

      // No `<co` ahead. Right-trim policy mirrors melody:
      //
      // - Mid-stream (no marker / EOS yet): hold trailing whitespace
      //   for the next chunk so a newline tucked against a forthcoming
      //   `<|END_*|>` doesn't leak into output.
      // - Final segment (marker boundary) or EOS: drop trailing
      //   whitespace outright — the marker / EOS proves it was
      //   incidental, mirroring the
      //   `pre_special_token.len()`-wide drain in melody's
      //   `apply_special_token_match` (which discards held trailing
      //   bytes alongside the marker itself).
      let rightTrim: RightTrim = (isFinalSegment || isEnd) ? .drop : .hold
      events.append(contentsOf: emitContent(
        remaining,
        asReasoning: asReasoning,
        rightTrim: rightTrim,
        allowPartialOpenHold: !isFinalSegment && !isEnd,
      ))
      remaining = ""
    }

    return events
  }

  private enum RightTrim {
    /// Mid-stream: trim trailing whitespace from the emitted text and
    /// re-buffer it for the next chunk.
    case hold
    /// Final segment / EOS: trim trailing whitespace from the emitted
    /// text and discard it. Mirrors melody's drain-at-marker behavior
    /// for `right_trimmed = true` filters.
    case drop
    /// Don't touch trailing whitespace. Used for pre-citation text
    /// where the trailing whitespace sits before a `<co` that has
    /// already arrived, so it isn't subject to the marker drop rule.
    case none
  }

  private enum CitationParse {
    case complete(consumed: Int, spanText: String, sources: [Source])
    /// Open tag was parsed in full (`<co...>`) but no close arrived
    /// before the end of the searchable region. `consumed` is the
    /// character count of the open tag — caller strips that prefix.
    case openOnly(consumed: Int)
    case partial
  }

  private struct Source {
    var toolCallIndex: Int
    var toolResultIndices: [Int]
  }

  /// Parse a `<co…>` citation starting at the head of `slice`. Returns
  /// the consumed character count along with the span text and the
  /// close-tag's source list. The result of melody's
  /// `find_an_element` cmd3 path, adapted to character-indexed output.
  private func parseCitation(in slice: String, isEnd: Bool) -> CitationParse {
    let chars = Array(slice)
    // The open prefix is `<co` (3 chars). Find the `>` that closes the
    // open tag. Whatever sits between `<co` and that `>` is the
    // optional legacy source list — ignored.
    guard chars.count >= Self.citationOpenPrefix.count else {
      return isEnd ? .openOnly(consumed: chars.count) : .partial
    }
    var i = Self.citationOpenPrefix.count
    while i < chars.count, chars[i] != ">" {
      i += 1
    }
    guard i < chars.count else {
      // No `>` yet. If isEnd, strip the partial open.
      return isEnd ? .openOnly(consumed: chars.count) : .partial
    }
    let openEnd = i + 1 // position past the `>`

    // Look for `</co: ` after the open tag.
    let afterOpen = String(chars[openEnd...])
    guard let closeStartRange = afterOpen.range(of: Self.citationCloseStart) else {
      return isEnd ? .openOnly(consumed: openEnd) : .partial
    }
    let closeStartOffset = openEnd + afterOpen.distance(
      from: afterOpen.startIndex,
      to: closeStartRange.lowerBound,
    )
    // Span text is between open's `>` and `</co: `.
    let span = String(chars[openEnd ..< closeStartOffset])

    // Find the `>` that closes `</co: …>`. Source-list body sits
    // between `</co: ` and that `>`.
    let sourceListStart = closeStartOffset + Self.citationCloseStart.count
    var j = sourceListStart
    while j < chars.count, chars[j] != ">" {
      j += 1
    }
    guard j < chars.count else {
      return isEnd ? .openOnly(consumed: openEnd) : .partial
    }
    let sourceListBody = String(chars[sourceListStart ..< j])
    let sources = parseCmd3Sources(sourceListBody)
    return .complete(consumed: j + 1, spanText: span, sources: sources)
  }

  /// Parse a cmd3 close-tag source list: `tci:[rid,rid],tci:[rid]`.
  /// Mirrors melody's `convert_string_to_doc_indices`.
  ///
  /// Legacy-format close tags like `</co: 0,1>` have no `]` separator,
  /// so the split-on-`]` strategy yields an empty source list — same
  /// behavior as melody under cmd3 parsing. The span text is still
  /// emitted with a zero-source annotation list.
  private func parseCmd3Sources(_ s: String) -> [Source] {
    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
    let groups = trimmed.split(separator: "]", omittingEmptySubsequences: false).map(String.init)
    var sources: [Source] = []
    // The last split element is the trailing tail after the final `]`;
    // skip it. Empty groups (consecutive `]]`) are ignored.
    guard groups.count >= 2 else { return [] }
    for raw in groups[..<(groups.count - 1)] {
      var group = raw
      while group.hasPrefix(",") {
        group.removeFirst()
      }
      let parts = group.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
      guard parts.count == 2 else { continue }
      let tciText = parts[0].trimmingCharacters(in: .whitespaces)
      var ridText = parts[1].trimmingCharacters(in: .whitespaces)
      while ridText.hasPrefix("[") {
        ridText.removeFirst()
      }
      guard let tci = Int(tciText), tci >= 0 else { continue }
      let rids: [Int] = ridText
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        .filter { $0 >= 0 }
      sources.append(Source(toolCallIndex: tci, toolResultIndices: rids))
    }
    return sources
  }

  // MARK: Mode close

  private mutating func closePreviousMode(target: Mode) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    // Same-mode marker transitions keep the open item: a leading
    // preamble already streamed into `groundedAnswer` should flow into
    // the same message item when the model later emits
    // `<|START_RESPONSE|>`. Only mode changes (or `<|END_*|>`
    // transitioning to `.ignored`) close the item.
    if mode == target { return [] }
    switch mode {
      case .groundedAnswer:
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
      case .reasoning:
        if openReasoning != nil {
          events.append(contentsOf: closeReasoning(status: .completed))
        }
      case .toolAction:
        events.append(contentsOf: emitToolCalls(from: actionBuffer))
        actionBuffer = ""
      case .ignored:
        break
    }
    skipLeadingReasoningWhitespace = false
    return events
  }

  // MARK: Tool calls

  private struct ParsedCall {
    let toolCallId: String?
    let name: String
    let arguments: String
  }

  private mutating func emitToolCalls(from body: String) -> [ResponseStreamingEvent] {
    let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
          let calls = parseToolCallArray(trimmed)
    else { return [] }
    var events: [ResponseStreamingEvent] = []
    for call in calls {
      events.append(contentsOf: emitToolCall(call))
    }
    return events
  }

  private func parseToolCallArray(_ arrayText: String) -> [ParsedCall]? {
    guard let data = arrayText.data(using: .utf8),
          let array = try? JSONSerialization.jsonObject(with: data) as? [Any]
    else { return nil }
    if array.isEmpty { return [] }
    var calls: [ParsedCall] = []
    for entry in array {
      guard let dict = entry as? [String: Any],
            let name = dict["tool_name"] as? String
      else { return nil }
      let id = dict["tool_call_id"] as? String
      let paramsValue = dict["parameters"] ?? [String: Any]()
      guard let argsJSON = serializeJSONArgument(paramsValue) else { return nil }
      calls.append(ParsedCall(toolCallId: id, name: name, arguments: argsJSON))
    }
    return calls
  }

  private func serializeJSONArgument(_ value: Any) -> String? {
    guard JSONSerialization.isValidJSONObject([value]),
          let data = try? JSONSerialization.data(
            withJSONObject: value,
            options: [.sortedKeys, .withoutEscapingSlashes, .fragmentsAllowed],
          )
    else { return nil }
    return String(data: data, encoding: .utf8)
  }

  // MARK: Event emission

  private mutating func emitContent(
    _ text: String,
    asReasoning: Bool,
    rightTrim: RightTrim,
    allowPartialOpenHold: Bool = false,
  ) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var safe = text
    var hold = ""
    if allowPartialOpenHold {
      // Hold back any tail that could grow into a `<co...` citation
      // open. (Partial marker holds are applied at the scan layer so
      // every mode gets them, including `.toolAction` and `.ignored`.)
      let (emit, h) = PrefixHold.split(text: safe, markers: [Self.citationOpenPrefix])
      hold = h + hold
      safe = emit
    }
    switch rightTrim {
      case .hold:
        let (e, h) = splitTrailingWhitespace(safe)
        safe = e
        hold = h + hold
      case .drop:
        safe = trimTrailingWhitespace(safe)
      case .none:
        break
    }
    if !hold.isEmpty {
      buffer = hold + buffer
    }
    if safe.isEmpty { return [] }
    return asReasoning ? emitReasoningDelta(text: safe) : emitMessageDelta(text: safe)
  }

  private func trimTrailingWhitespace(_ text: String) -> String {
    var copy = text
    while let last = copy.last, last.isWhitespace {
      copy.removeLast()
    }
    return copy
  }

  private mutating func emitCitation(
    spanText: String,
    sources: [Source],
  ) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem())
    }
    let startUtf16 = openMessage?.utf16Length ?? 0
    if !spanText.isEmpty {
      events.append(contentsOf: emitMessageDelta(text: spanText))
    }
    let endUtf16 = openMessage?.utf16Length ?? startUtf16
    guard let msg = openMessage else { return events }
    if sources.isEmpty {
      // Span recognized but no source list — emit a single annotation
      // with `toolCallIndex = 0` and empty `toolResultIndices` so the
      // citation surface still reaches consumers. Mirrors melody's
      // emit-citation-with-empty-sources behavior under cmd3 parsing
      // of a legacy close tag.
      let annotation = ResponseOutputText.Annotation.cohereToolResultCitation(
        toolCallIndex: 0,
        toolResultIndices: [],
        startIndex: startUtf16,
        endIndex: endUtf16,
      )
      events.append(emitAnnotation(annotation, to: msg))
      updateAnnotations { $0.append(annotation) }
    } else {
      for source in sources {
        let annotation = ResponseOutputText.Annotation.cohereToolResultCitation(
          toolCallIndex: source.toolCallIndex,
          toolResultIndices: source.toolResultIndices,
          startIndex: startUtf16,
          endIndex: endUtf16,
        )
        events.append(emitAnnotation(annotation, to: msg))
        updateAnnotations { $0.append(annotation) }
      }
    }
    return events
  }

  private mutating func updateAnnotations(_ mutate: (inout [ResponseOutputText.Annotation]) -> Void) {
    if var m = openMessage {
      mutate(&m.annotations)
      openMessage = m
    }
  }

  private mutating func emitAnnotation(
    _ annotation: ResponseOutputText.Annotation,
    to msg: OpenMessage,
  ) -> ResponseStreamingEvent {
    .outputTextAnnotationAdded(.init(
      itemId: msg.id,
      outputIndex: msg.outputIndex,
      contentIndex: 0,
      annotationIndex: msg.annotations.count,
      annotation: annotation,
      sequenceNumber: takeSequence(),
    ))
  }

  private mutating func emitMessageDelta(text: String) -> [ResponseStreamingEvent] {
    if text.isEmpty { return [] }
    var events: [ResponseStreamingEvent] = []
    if openMessage == nil {
      events.append(contentsOf: openMessageItem())
    }
    if var msg = openMessage {
      msg.emittedText += text
      msg.utf16Length += text.utf16.count
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
    let part = ResponseOutputText(text: msg.emittedText, annotations: msg.annotations)
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

  private mutating func emitToolCall(_ call: ParsedCall) -> [ResponseStreamingEvent] {
    let id = IDFactory.make(.functionCall)
    // Cohere emits `tool_call_id` (typically the positional index "0",
    // "1", …) on every call. Cohere chat templates default to
    // `regen_tool_call_ids = true`, which regenerates IDs positionally
    // on subsequent turns regardless of what we send back, so the
    // model-emitted value has no semantic weight beyond ordering.
    // Mirror melody and vLLM's behavior: forward the raw value
    // verbatim into `call_id`. Mint a fresh `call_…` only when the
    // model omitted the field entirely.
    let callId = call.toolCallId ?? IDFactory.make(.callId)
    let outputIndex = takeOutputIndex()
    let openItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: call.name, arguments: "", status: .inProgress,
    )
    let doneItem = ResponseFunctionToolCall(
      id: id, callId: callId, name: call.name, arguments: call.arguments, status: .completed,
    )
    var events: [ResponseStreamingEvent] = []
    events.append(.outputItemAdded(.init(
      item: .functionCall(openItem), outputIndex: outputIndex, sequenceNumber: takeSequence(),
    )))
    if !call.arguments.isEmpty {
      events.append(.functionCallArgumentsDelta(.init(
        itemId: id, outputIndex: outputIndex, delta: call.arguments, sequenceNumber: takeSequence(),
      )))
    }
    events.append(.functionCallArgumentsDone(.init(
      itemId: id, outputIndex: outputIndex, arguments: call.arguments, sequenceNumber: takeSequence(),
    )))
    events.append(.outputItemDone(.init(
      item: .functionCall(doneItem), outputIndex: outputIndex, sequenceNumber: takeSequence(),
    )))
    return events
  }

  // MARK: Whitespace helpers

  /// Split `text` into the leading non-trailing-whitespace portion and
  /// the trailing run of whitespace. The trailing whitespace is held
  /// back so it can either be absorbed by the next chunk's content or
  /// dropped when a marker / EOS proves it was incidental.
  private func splitTrailingWhitespace(_ text: String) -> (emit: String, hold: String) {
    if text.isEmpty { return ("", "") }
    let chars = Array(text)
    var i = chars.count
    while i > 0, chars[i - 1].isWhitespace {
      i -= 1
    }
    if i == chars.count { return (text, "") }
    let pivot = text.index(text.startIndex, offsetBy: i)
    return (String(text[..<pivot]), String(text[pivot...]))
  }

  // MARK: ID helpers

  private mutating func takeOutputIndex() -> Int {
    defer { nextOutputIndex += 1 }
    return nextOutputIndex
  }

  private mutating func takeSequence() -> Int {
    defer { nextSequence += 1 }
    return nextSequence
  }
}
