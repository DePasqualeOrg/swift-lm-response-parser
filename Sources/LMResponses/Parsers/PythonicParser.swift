// Copyright © Anthony DePasquale

import Foundation

/// Parser for the Pythonic tool-call format used by Llama 4 and LFM2.
///
/// **Wire shape.** Tool calls use Python function-call syntax inside a
/// list literal:
///
/// ```text
/// [tool1(arg1=val1, arg2=val2), tool2(arg1=val3)]
/// ```
///
/// Arguments are Python literals: strings (single or double quoted), ints,
/// floats, booleans (`True` / `False`), `None`, lists, and dicts. The
/// parser converts each value to its JSON equivalent before composing
/// the `arguments` field. The model may wrap the call list in optional
/// start/end marker tokens; both are stripped before parsing. The
/// markers are configurable via ``init(startTag:endTag:acceptJSON:newlineSeparated:)``
/// because different model families use different literal strings:
///
/// - Llama 4: `<|python_start|>` / `<|python_end|>` (the defaults).
/// - LFM2 (Liquid AI): `<|tool_call_start|>` / `<|tool_call_end|>`.
/// - OLMo 3 (Allen AI): `<function_calls>` / `</function_calls>` with
///   newline-separated calls (no bracket list); see `newlineSeparated`.
///
/// **JSON dual-format mode.** Some families (LFM2) also emit calls as
/// JSON inside the same wrapper tokens – either a list of objects
/// `[{"name": "...", "arguments": {...}}, ...]` or a single object
/// `{"name": "...", "arguments": {...}}`. Pass ``acceptJSON: true`` to
/// enable both shapes. Mirrors sglang's `Lfm2Detector` which tries JSON
/// first when the wrapped content begins with `{` or `[{` and falls back
/// to Pythonic.
///
/// **Newline-separated mode.** OLMo 3 wraps calls in
/// `<function_calls>...</function_calls>` and separates them with newlines
/// instead of commas (no surrounding `[...]`). Pass ``newlineSeparated:
/// true`` to enable: in this mode the wrapper is the only signal – the
/// parser does not detect bare bracket-lists, treats text outside the
/// wrapper as plain content, and parses the inner content by splitting
/// on newlines and reusing the comma-separated pythonic parser. Mirrors
/// vLLM's `Olmo3PythonicToolParser`, which performs the same
/// splitlines / `", ".join` / wrap-in-`[...]` transform.
///
/// Plain text outside the bracketed call list is normal message content.
/// During streaming, the parser waits for the bracket-balanced call list
/// before emitting any tool calls and forwards safe content prefixes
/// chunk by chunk. LFM2 can opt into `requiresWrapper`, which makes the
/// wrapper structural and waits for the complete envelope before parsing
/// the Pythonic/JSON payload inside it.
struct PythonicParser: ResponseFormatParser {
  /// Initial reasoning state for `acceptThink` mode. Mirrors the
  /// V3-family parsers: `.reasoning` for fresh requests where the chat
  /// template injects `<think>` into the prompt, `.normal` for
  /// continuation requests whose prior output already contained
  /// `</think>`.
  enum InitialState: Equatable {
    case normal
    case reasoning
  }

  private let startTag: String
  private let endTag: String
  private let acceptJSON: Bool
  private let newlineSeparated: Bool
  private let acceptThink: Bool
  private let requiresWrapper: Bool
  private let acceptBarePythonicCall: Bool

  /// Buffer of bytes we haven't decided what to do with yet. Bytes get
  /// drained from the front as they're emitted (as content) or consumed
  /// (as part of a tool call list).
  private var buffer: String = ""

  private var openMessage: OpenMessage?

  private var nextOutputIndex: Int = 0
  private var nextSequence: Int = 0

  private var thinkPreamble: ThinkPreambleExtractor

  private struct OpenMessage {
    var id: String
    var outputIndex: Int
    var emittedText: String = ""
  }

  /// Construct a Pythonic-format parser. Defaults match Llama 4's
  /// `<|python_start|>` / `<|python_end|>` wrapper tokens. Pass
  /// alternative tags for variants like LFM2 that use the same Pythonic
  /// inner syntax with different wrapper markers. Pass `acceptJSON:
  /// true` to also accept JSON-shaped call lists or single objects (the
  /// LFM2 dual-format protocol). Pass `newlineSeparated: true` for
  /// OLMo 3, whose `<function_calls>` envelope contains newline-separated
  /// calls instead of a bracket-list. Pass `acceptThink: true` (with
  /// `newlineSeparated: true`) to also extract a leading `<think>...</think>`
  /// reasoning preamble – matches OLMo 3 Think variants whose chat
  /// template injects `<think>` into the prompt. Pass
  /// `requiresWrapper: true` when the wrapper is mandatory (LFM2), and
  /// `acceptBarePythonicCall: true` when the envelope content may be a
  /// single `fn(...)` call instead of a bracketed call list.
  init(
    startTag: String = "<|python_start|>",
    endTag: String = "<|python_end|>",
    acceptJSON: Bool = false,
    newlineSeparated: Bool = false,
    acceptThink: Bool = false,
    initialState: InitialState = .reasoning,
    requiresWrapper: Bool = false,
    acceptBarePythonicCall: Bool = false,
  ) {
    self.startTag = startTag
    self.endTag = endTag
    self.acceptJSON = acceptJSON
    self.newlineSeparated = newlineSeparated
    self.acceptThink = acceptThink
    self.requiresWrapper = requiresWrapper
    self.acceptBarePythonicCall = acceptBarePythonicCall
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
    if acceptThink {
      events.append(contentsOf: emitContentBeforeExplicitThinkIfNeeded())
      events.append(contentsOf: thinkPreamble.drain(
        buffer: &buffer,
        isEnd: false,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
      // While reasoning is still open, hold the rest of the buffer.
      if thinkPreamble.phase != .done { return events }
    }
    events.append(contentsOf: scan(isEnd: false))
    return events
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    if acceptThink {
      events.append(contentsOf: emitContentBeforeExplicitThinkIfNeeded())
      events.append(contentsOf: thinkPreamble.drain(
        buffer: &buffer,
        isEnd: true,
        nextSequence: &nextSequence,
        nextOutputIndex: &nextOutputIndex,
      ))
    }
    events.append(contentsOf: scan(isEnd: true))
    // Anything left in the buffer at finalize is plain content.
    if !buffer.isEmpty {
      events.append(contentsOf: emitMessageDelta(text: buffer))
      buffer = ""
    }
    if openMessage != nil {
      events.append(contentsOf: closeMessage(status: .completed))
    }
    if acceptThink {
      events.append(contentsOf: thinkPreamble.finalizeIfOpen(nextSequence: &nextSequence))
    }
    return events
  }

  // MARK: Scan

  private mutating func scan(isEnd: Bool) -> [ResponseStreamingEvent] {
    if newlineSeparated {
      return scanNewlineSeparated(isEnd: isEnd)
    }
    if requiresWrapper {
      return scanRequiredWrapper(isEnd: isEnd)
    }
    var events: [ResponseStreamingEvent] = []
    // Loop while we make progress: each iteration either emits a
    // content chunk, drops a wrapper token, or consumes a tool call.
    var didProgress = true
    while didProgress {
      didProgress = false

      // 1. Drop a complete wrapper token at the head of the buffer.
      if buffer.hasPrefix(startTag) {
        buffer.removeFirst(startTag.count)
        didProgress = true
        continue
      }
      if buffer.hasPrefix(endTag) {
        buffer.removeFirst(endTag.count)
        didProgress = true
        continue
      }

      // 2. If a wrapper tag appears later in the buffer, emit the
      // text before it as message content, then consume the tag on
      // the next loop iteration. Mirrors sglang's `pythonic_detector`,
      // which strips wrappers wherever they occur – not just at the
      // head – so that any preamble emitted before the tool-call
      // envelope reaches the consumer cleanly. Diverges from vLLM,
      // which only recognizes wrappers at the buffer head.
      let startTagIdx = buffer.range(of: startTag)?.lowerBound
      let endTagIdx = buffer.range(of: endTag)?.lowerBound
      let bracketIdx = earliestBracketStart()

      // Pick the earliest interesting position; tags win ties so we
      // don't accidentally swallow a wrapper as bracket-list content.
      let candidates: [(idx: String.Index, kind: Int)] =
        [(startTagIdx, 0), (endTagIdx, 0), (bracketIdx, 1)]
          .compactMap { pair in pair.0.map { ($0, pair.1) } }
      guard let earliest = candidates.min(by: { lhs, rhs in
        if lhs.idx != rhs.idx { return lhs.idx < rhs.idx }
        return lhs.kind < rhs.kind
      }) else {
        // No wrapper, no bracket – fall through to the partial-
        // suffix handling below.
        let partial = trailingPartialWrapperOverlap(of: buffer)
        let safeEnd = isEnd ? buffer.count : buffer.count - partial
        if safeEnd > 0 {
          let safeText = String(buffer.prefix(safeEnd))
          events.append(contentsOf: emitMessageDelta(text: safeText))
          buffer.removeFirst(safeEnd)
          didProgress = safeEnd > 0
        }
        return events
      }

      if earliest.kind == 0 {
        // Wrapper tag is the next interesting position. Emit the
        // text before it (if any), then loop to consume the tag.
        let preTag = String(buffer[buffer.startIndex ..< earliest.idx])
        if !preTag.isEmpty {
          events.append(contentsOf: emitMessageDelta(text: preTag))
          buffer.removeFirst(preTag.count)
        }
        didProgress = true
        continue
      }

      // Bracket is the earliest interesting position.
      let preBracket = String(buffer[buffer.startIndex ..< earliest.idx])
      if !preBracket.isEmpty {
        events.append(contentsOf: emitMessageDelta(text: preBracket))
        buffer.removeFirst(preBracket.count)
        didProgress = true
        continue
      }
      // Buffer starts with `[` (or `{` when acceptJSON). Find the
      // matching close based on which kind we're looking at.
      let openChar = buffer.first!
      guard let closeIdx = matchingCloseBracket(in: buffer, openChar: openChar) else {
        // No close yet. At end-of-stream, emit as content.
        if isEnd {
          events.append(contentsOf: emitMessageDelta(text: buffer))
          buffer = ""
          didProgress = true
        }
        return events
      }
      let callListText = String(buffer[buffer.startIndex ... closeIdx])
      let consumed = buffer.distance(from: buffer.startIndex, to: closeIdx) + 1
      // Try parsing as a tool-call list. If parsing fails, retry
      // with any embedded wrapper tags stripped, mirroring sglang's
      // `_text_strip` pre-processing for tools that occasionally
      // surface `<|python_end|>` inside the bracket list.
      var calls = tryParseCalls(callListText)
      if calls?.isEmpty ?? true {
        let stripped = callListText
          .replacingOccurrences(of: startTag, with: "")
          .replacingOccurrences(of: endTag, with: "")
        if stripped != callListText, let retry = tryParseCalls(stripped), !retry.isEmpty {
          calls = retry
        }
      }
      if let calls, !calls.isEmpty {
        if openMessage != nil {
          events.append(contentsOf: closeMessage(status: .completed))
        }
        for call in calls {
          events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
        }
      } else {
        // Not a valid call list – forward as plain content.
        events.append(contentsOf: emitMessageDelta(text: callListText))
      }
      buffer.removeFirst(consumed)
      didProgress = true
      continue
    }
    return events
  }

  /// vLLM's OLMo 3 reasoning buffer treats text before an explicit
  /// `<think>` opener as ordinary content. The shared
  /// `ThinkPreambleExtractor` models the DeepSeek-style implicit
  /// reasoning preamble, so OLMo's content-before-opener case is
  /// handled here before the shared extractor drains reasoning.
  private mutating func emitContentBeforeExplicitThinkIfNeeded() -> [ResponseStreamingEvent] {
    guard thinkPreamble.phase != .done,
          let startRange = buffer.range(of: "<think>")
    else { return [] }
    if let endRange = buffer.range(of: "</think>"),
       endRange.lowerBound < startRange.lowerBound
    {
      return []
    }

    let preThink = String(buffer[buffer.startIndex ..< startRange.lowerBound])
    guard !preThink.isEmpty else { return [] }
    var events = emitMessageDelta(text: preThink)
    events.append(contentsOf: closeMessage(status: .completed))
    let consumed = buffer.distance(from: buffer.startIndex, to: startRange.upperBound)
    buffer.removeFirst(consumed)
    return events
  }

  /// LFM2's wrapper tokens are structural, not merely cosmetic. SGLang
  /// only starts parsing after `<|tool_call_start|>` and waits for the
  /// matching `<|tool_call_end|>`, but accepts either a Pythonic list, a
  /// single bare Pythonic call, or JSON inside the envelope.
  private mutating func scanRequiredWrapper(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var didProgress = true
    while didProgress {
      didProgress = false

      if buffer.hasPrefix(startTag) {
        let afterStartIdx = buffer.index(buffer.startIndex, offsetBy: startTag.count)
        let rest = buffer[afterStartIdx...]
        guard let endRange = rest.range(of: endTag) else {
          if isEnd {
            events.append(contentsOf: emitMessageDelta(text: buffer))
            buffer = ""
            didProgress = true
          }
          return events
        }

        let inner = String(rest[rest.startIndex ..< endRange.lowerBound])
        let consumed = startTag.count + inner.count + endTag.count
        let wrappedCalls = parseWrappedContentCalls(inner)
        if let calls = wrappedCalls, !calls.isEmpty {
          if openMessage != nil {
            events.append(contentsOf: closeMessage(status: .completed))
          }
          for call in calls {
            events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
          }
        } else if wrappedCalls == nil,
                  !inner.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
          let envelope = String(buffer.prefix(consumed))
          events.append(contentsOf: emitMessageDelta(text: envelope))
        }
        buffer.removeFirst(consumed)
        didProgress = true
        continue
      }

      if buffer.hasPrefix(endTag) {
        buffer.removeFirst(endTag.count)
        didProgress = true
        continue
      }

      if let startRange = buffer.range(of: startTag) {
        let pre = String(buffer[buffer.startIndex ..< startRange.lowerBound])
        if !pre.isEmpty {
          events.append(contentsOf: emitMessageDelta(text: pre))
          buffer.removeFirst(pre.count)
        }
        didProgress = true
        continue
      }

      let partial = trailingPartialWrapperOverlap(of: buffer)
      let safeEnd = isEnd ? buffer.count : buffer.count - partial
      if safeEnd > 0 {
        let safeText = String(buffer.prefix(safeEnd))
        events.append(contentsOf: emitMessageDelta(text: safeText))
        buffer.removeFirst(safeEnd)
        didProgress = safeEnd > 0
      }
      return events
    }
    return events
  }

  /// Newline-separated scan path for OLMo 3. The wrapper is the only
  /// signal: text outside `<function_calls>...</function_calls>` is
  /// always plain content and we never look for a bare bracket-list.
  /// Inside the wrapper, calls are split by `\n`, then handed to the
  /// existing comma-separated pythonic parser via a transformation
  /// equivalent to vLLM's splitlines / `", ".join` / wrap-in-`[...]`.
  private mutating func scanNewlineSeparated(isEnd: Bool) -> [ResponseStreamingEvent] {
    var events: [ResponseStreamingEvent] = []
    var didProgress = true
    while didProgress {
      didProgress = false

      if buffer.hasPrefix(startTag) {
        // Look for the matching closer in the rest of the buffer.
        let afterStartIdx = buffer.index(buffer.startIndex, offsetBy: startTag.count)
        let rest = buffer[afterStartIdx...]
        if let endRange = rest.range(of: endTag) {
          let inner = String(rest[rest.startIndex ..< endRange.lowerBound])
          let consumed = startTag.count + inner.count + endTag.count
          let calls = parseNewlineSeparatedCalls(inner)
          if let calls, !calls.isEmpty {
            if openMessage != nil {
              events.append(contentsOf: closeMessage(status: .completed))
            }
            for call in calls {
              events.append(contentsOf: emitToolCall(name: call.name, arguments: call.arguments))
            }
          } else {
            // Parse failed – forward the whole envelope as plain
            // content so the user still sees the model output.
            let envelope = String(buffer.prefix(consumed))
            events.append(contentsOf: emitMessageDelta(text: envelope))
          }
          buffer.removeFirst(consumed)
          didProgress = true
          continue
        }
        // No closer yet. At end-of-stream the envelope never closed –
        // treat the whole thing as content. Otherwise keep buffering.
        if isEnd {
          events.append(contentsOf: emitMessageDelta(text: buffer))
          buffer = ""
          didProgress = true
        }
        return events
      }

      // No `<function_calls>` at the head. Look for it later in the
      // buffer; forward any preceding text as content.
      if let startRange = buffer.range(of: startTag) {
        let pre = String(buffer[buffer.startIndex ..< startRange.lowerBound])
        if !pre.isEmpty {
          events.append(contentsOf: emitMessageDelta(text: pre))
          buffer.removeFirst(pre.count)
        }
        didProgress = true
        continue
      }

      // No wrapper opener anywhere. Forward content with the trailing
      // partial-tag overlap held back so a chunk boundary inside
      // `<function_calls>` doesn't leak as content.
      let partial = trailingPartialWrapperOverlap(of: buffer)
      let safeEnd = isEnd ? buffer.count : buffer.count - partial
      if safeEnd > 0 {
        let safeText = String(buffer.prefix(safeEnd))
        events.append(contentsOf: emitMessageDelta(text: safeText))
        buffer.removeFirst(safeEnd)
        didProgress = safeEnd > 0
      }
      return events
    }
    return events
  }

  /// Parse newline-separated calls by transforming to the comma-
  /// separated bracket-list form the existing pythonic parser already
  /// understands. Mirrors vLLM's `Olmo3PythonicToolParser`, which does
  /// the same `splitlines` / `", ".join` / `f"[{...}]"` rewrite.
  private func parseNewlineSeparatedCalls(_ inner: String) -> [ParsedCall]? {
    let lines = inner
      .split(omittingEmptySubsequences: false, whereSeparator: { $0 == "\n" })
      .map { $0.trimmingCharacters(in: .whitespaces) }
      .filter { !$0.isEmpty }
    if lines.isEmpty { return [] }
    let listText = "[" + lines.joined(separator: ", ") + "]"
    return parsePythonicCalls(listText)
  }

  /// Index of the next bracket-list opener in the buffer. With
  /// `acceptJSON`, both `[` and `{` are valid openers (LFM2's JSON
  /// shape can be a single object). Without, only `[` (the Pythonic
  /// list opener).
  private func earliestBracketStart() -> String.Index? {
    let lb = buffer.firstIndex(of: "[")
    guard acceptJSON else { return lb }
    let cb = buffer.firstIndex(of: "{")
    switch (lb, cb) {
      case let (.some(l), .some(c)): return Swift.min(l, c)
      case (.some, .none): return lb
      case (.none, .some): return cb
      case (.none, .none): return nil
    }
  }

  /// Try to parse the bracket text as either Pythonic or JSON calls.
  /// Mirrors sglang's `Lfm2Detector._parse_tool_calls_content`: tries
  /// JSON first when the content begins with `{` or `[{` (after
  /// trimming), then falls back to Pythonic.
  private func tryParseCalls(_ text: String) -> [ParsedCall]? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if acceptJSON, trimmed.hasPrefix("[{") || trimmed.hasPrefix("{") {
      if let json = parseJSONCalls(trimmed) { return json }
    }
    return parsePythonicCalls(text)
  }

  private func parseWrappedContentCalls(_ text: String) -> [ParsedCall]? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return [] }
    if let calls = tryParseCalls(trimmed) { return calls }
    if acceptBarePythonicCall {
      return parsePythonicCalls("[\(trimmed)]")
    }
    return nil
  }

  private func trailingPartialWrapperOverlap(of text: String) -> Int {
    let chars = Array(text)
    let s1 = partialOverlap(suffixOf: chars, with: Array(startTag))
    let s2 = partialOverlap(suffixOf: chars, with: Array(endTag))
    return Swift.max(s1, s2)
  }

  private func matchingCloseBracket(in slice: String, openChar: Character) -> String.Index? {
    let closeChar: Character = openChar == "[" ? "]" : "}"
    var depth = 0
    var inString = false
    var quote: Character = "\""
    var escape = false
    var i = slice.startIndex
    while i < slice.endIndex {
      let ch = slice[i]
      if escape { escape = false; i = slice.index(after: i); continue }
      if ch == "\\" { escape = true; i = slice.index(after: i); continue }
      if inString {
        if ch == quote { inString = false }
        i = slice.index(after: i)
        continue
      }
      if ch == "\"" || ch == "'" {
        inString = true; quote = ch
      } else if ch == openChar {
        depth += 1
      } else if ch == closeChar {
        depth -= 1
        if depth == 0 { return i }
      }
      i = slice.index(after: i)
    }
    return nil
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

  // MARK: Pythonic literal parsing

  private struct ParsedCall {
    let name: String
    let arguments: String
  }

  /// Parse JSON-shaped tool calls used by LFM2's dual-format protocol.
  /// Accepts a list `[{name, arguments}, ...]` or a single object
  /// `{name, arguments}`. Returns nil if the JSON doesn't decode or has
  /// the wrong shape; returns an empty array if the JSON parses but
  /// contains no name-bearing entries.
  private func parseJSONCalls(_ text: String) -> [ParsedCall]? {
    guard let data = text.data(using: .utf8),
          let parsed = try? JSONSerialization.jsonObject(with: data)
    else { return nil }

    let dicts: [[String: Any]]
    if let array = parsed as? [Any] {
      dicts = array.compactMap { $0 as? [String: Any] }
      if dicts.count != array.count { return nil }
    } else if let single = parsed as? [String: Any] {
      dicts = [single]
    } else {
      return nil
    }

    var calls: [ParsedCall] = []
    for dict in dicts {
      guard let name = dict["name"] as? String, !name.isEmpty else {
        return nil
      }
      // sglang's `parse_base_json` accepts either `arguments` or
      // `parameters`; mirror that.
      let argsValue = dict["arguments"] ?? dict["parameters"] ?? [String: Any]()
      let argsJSON: String = if let s = argsValue as? String {
        s
      } else if let data = try? JSONSerialization.data(
        withJSONObject: argsValue,
        // `.sortedKeys` for deterministic key order across
        // Foundation dictionaries; matches the rest of the
        // parser library.
        options: [.sortedKeys, .withoutEscapingSlashes],
      ), let s = String(data: data, encoding: .utf8) {
        s
      } else {
        "{}"
      }
      calls.append(ParsedCall(name: name, arguments: argsJSON))
    }
    return calls
  }

  private func parsePythonicCalls(_ text: String) -> [ParsedCall]? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix("["), trimmed.hasSuffix("]") else { return nil }
    let inner = String(trimmed.dropFirst().dropLast())
    if inner.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return [] }

    var calls: [ParsedCall] = []
    var i = inner.startIndex
    while i < inner.endIndex {
      while i < inner.endIndex, inner[i].isWhitespace || inner[i] == "," {
        i = inner.index(after: i)
      }
      if i >= inner.endIndex { break }
      guard let parenOpen = inner[i...].firstIndex(of: "(") else { return nil }
      let name = String(inner[i ..< parenOpen]).trimmingCharacters(in: .whitespacesAndNewlines)
      guard isValidIdentifier(name) else { return nil }
      guard let parenCloseIdx = matchingClosingParen(in: inner, openAt: parenOpen) else { return nil }
      let argsText = String(inner[inner.index(after: parenOpen) ..< parenCloseIdx])
      guard let argsJSON = pythonKeywordsToJSON(argsText) else { return nil }
      calls.append(ParsedCall(name: name, arguments: argsJSON))
      i = inner.index(after: parenCloseIdx)
    }
    return calls
  }

  private func isValidIdentifier(_ name: String) -> Bool {
    guard let first = name.first, first.isLetter || first == "_" else { return false }
    return name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
  }

  private func matchingClosingParen(in text: String, openAt: String.Index) -> String.Index? {
    var depth = 0
    var inString = false
    var quote: Character = "\""
    var escape = false
    var i = openAt
    while i < text.endIndex {
      let ch = text[i]
      if escape { escape = false; i = text.index(after: i); continue }
      if ch == "\\" { escape = true; i = text.index(after: i); continue }
      if inString {
        if ch == quote { inString = false }
        i = text.index(after: i)
        continue
      }
      if ch == "\"" || ch == "'" {
        inString = true; quote = ch
      } else if ch == "(" {
        depth += 1
      } else if ch == ")" {
        depth -= 1
        if depth == 0 { return i }
      }
      i = text.index(after: i)
    }
    return nil
  }

  private func pythonKeywordsToJSON(_ text: String) -> String? {
    let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if stripped.isEmpty { return "{}" }

    var pairs: [(key: String, value: String)] = []
    var i = stripped.startIndex
    while i < stripped.endIndex {
      while i < stripped.endIndex, stripped[i].isWhitespace || stripped[i] == "," {
        i = stripped.index(after: i)
      }
      if i >= stripped.endIndex { break }
      let keyStart = i
      while i < stripped.endIndex, stripped[i].isLetter || stripped[i].isNumber || stripped[i] == "_" {
        i = stripped.index(after: i)
      }
      let key = String(stripped[keyStart ..< i])
      guard !key.isEmpty else { return nil }
      while i < stripped.endIndex, stripped[i].isWhitespace {
        i = stripped.index(after: i)
      }
      guard i < stripped.endIndex, stripped[i] == "=" else { return nil }
      i = stripped.index(after: i)
      while i < stripped.endIndex, stripped[i].isWhitespace {
        i = stripped.index(after: i)
      }
      guard let (valueJSON, end) = PythonLiteral.parseValue(in: stripped, from: i) else { return nil }
      pairs.append((key: key, value: valueJSON))
      i = end
    }

    var out = "{"
    for (idx, pair) in pairs.enumerated() {
      if idx > 0 { out += ", " }
      out += encodeJSONStringLiteral(pair.key) + ": " + pair.value
    }
    out += "}"
    return out
  }

  private func encodeJSONStringLiteral(_ s: String) -> String {
    if let data = try? JSONSerialization.data(withJSONObject: [s], options: []),
       let encoded = String(data: data, encoding: .utf8),
       encoded.count >= 2
    {
      return String(encoded.dropFirst().dropLast())
    }
    var out = "\""
    for ch in s {
      switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out.append(ch)
      }
    }
    out += "\""
    return out
  }

  // MARK: Tool-call emission

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
