// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses
import Tokenizers

/// Multi-turn chat session that produces one Responses envelope per logical
/// assistant turn (across any number of internal generation passes for tool
/// dispatch).
///
/// One session call corresponds to one logical assistant turn. The
/// conversation history is re-applied through the model's chat template
/// on every turn; the prior turn's KV tail is reused via LCP against
/// the rendered prompt (see `runTurn`).
///
/// Not thread-safe — one task per session.
///
/// **Cancellation.** Mirror MLX's behavior: cancelling a turn closes the
/// stream silently. *Chat history* is only mutated on a clean turn
/// finish, so a cancelled turn leaves `state.history` unchanged.
/// *KV cache* state is committed on every exit path — the
/// `defer { state.cachedKVTokens = cachedKV }` reflects what the engine
/// actually decoded into the underlying `LlamaContext`'s KV memory,
/// regardless of whether the turn finished cleanly. The next turn's
/// LCP machinery depends on this: if we committed empty after a cancel
/// but KV still held the in-flight prompt + partial generation,
/// `keepTokens` would skip truncating the stale tail and `llama_decode`
/// would fail with "inconsistent sequence positions." This is a
/// deliberate asymmetry with ``MultimodalResponseChatSession``, which
/// resets `cachedSignatures` to `[]` on cancel because it can't
/// pinpoint partial mtmd-chunk progress — the text path can track
/// partial progress at token granularity, mtmd can't.
public final class ResponseChatSession: @unchecked Sendable {
  private let context: LlamaContext
  private let modelName: String

  public var instructions: String?
  public var generateParameters: GenerateParameters
  public var additionalContext: [String: any Sendable]?
  public var tools: [ToolSpec]?
  public var toolDispatch: ToolDispatch?
  public var format: ResponseFormat?

  /// Extra stop-token strings to inject into the engine's halt set, in
  /// addition to the model's EOS and any format-required halt tokens.
  public var extraEOSTokens: Set<String>

  /// Per-session state: conversation history plus the tokens currently
  /// in the KV cache (so the next turn can prefix-reuse them). Lives
  /// inside a ``SessionCacheStore`` so the lock-protected mutation also
  /// serializes concurrent turn submissions.
  fileprivate struct State {
    var history: [Tokenizers.Message] = []
    /// Tokens currently in the KV cache at positions
    /// `[0, cachedKVTokens.count)`. Includes both prompt tokens and
    /// previously generated tokens (everything decoded into the KV;
    /// EOS/EOT tokens that halted generation without being decoded are
    /// not included). Used to compute the longest common prefix with
    /// the next turn's rendered prompt.
    var cachedKVTokens: [Int32] = []
  }

  private let stateStore: SessionCacheStore<State>

  /// The terminal ``/LMResponses/Response`` snapshot from the most recently finalized
  /// turn, or `nil` until the first turn finalizes. Updated after the
  /// consumer has been yielded the terminal response event.
  public var lastResponse: Response? {
    lastResponseBox.current
  }

  private let lastResponseBox = LastResponseBox()

  public typealias ToolDispatch =
    @Sendable (ResponseFunctionToolCall) async throws -> ResponseFunctionCallOutput.Output

  public init(
    context: LlamaContext,
    modelName: String,
    instructions: String? = nil,
    generateParameters: GenerateParameters = .init(),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
    extraEOSTokens: Set<String> = [],
  ) {
    self.context = context
    self.modelName = modelName
    self.instructions = instructions
    self.generateParameters = generateParameters
    self.additionalContext = additionalContext
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.format = format
    self.extraEOSTokens = extraEOSTokens
    stateStore = SessionCacheStore<State>(State())
  }

  /// Persist the session's current KV cache to `url` so a future
  /// session can resume without re-decoding the prompt prefix.
  ///
  /// The file holds the KV state for sequence 0 plus the token list at
  /// positions `[0, n)` — both are restored by
  /// ``init(context:modelName:instructions:history:cache:generateParameters:additionalContext:tools:toolDispatch:format:extraEOSTokens:)``.
  /// History (the chat-template messages) is **not** in the file; the
  /// caller is responsible for persisting `history` (e.g. as JSON)
  /// alongside the cache file and passing both at re-load time.
  ///
  /// Acquires the session lock; do not call from inside a running
  /// `toolDispatch` closure.
  public func saveCache(to url: URL) async throws {
    let context = context
    try await stateStore.read { state in
      try await context.saveCache(to: url, tokens: state.cachedKVTokens)
    }
  }

  /// Initialize with a pre-existing conversation history and KV cache
  /// file produced by ``saveCache(to:)`` on a previous session.
  ///
  /// The first turn rehydrates both: the chat template renders against
  /// `history` (system message prepended via `instructions` if set), and
  /// the LCP between the new rendered prompt and the cache's saved
  /// tokens skips re-decoding the matching prefix. If `history` and the
  /// cache file's contents diverge, the LCP path still produces correct
  /// output — just with less reuse.
  public init(
    context: LlamaContext,
    modelName: String,
    instructions: String? = nil,
    history: [Tokenizers.Message],
    cache cacheURL: URL,
    generateParameters: GenerateParameters = .init(),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
    extraEOSTokens: Set<String> = [],
  ) async throws {
    self.context = context
    self.modelName = modelName
    self.instructions = instructions
    self.generateParameters = generateParameters
    self.additionalContext = additionalContext
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.format = format
    self.extraEOSTokens = extraEOSTokens
    let tokens = try await context.loadCache(
      from: cacheURL,
      capacity: Int(context.contextLength),
    )
    let filtered = Self.droppingSystemMessages(history)
    stateStore = SessionCacheStore<State>(
      State(history: filtered, cachedKVTokens: tokens),
    )
  }

  /// Initialize with a pre-existing conversation history.
  ///
  /// `history` is a sequence of chat-template messages (`role` + `content`,
  /// plus optional `tool_calls` / `tool_call_id` for tool-use turns) that
  /// the next turn will render through the model's chat template alongside
  /// the new user message. System messages should be passed via
  /// `instructions` rather than included here; any `"role": "system"`
  /// entries in `history` are dropped to match the empty-init invariant.
  ///
  /// No KV cache state is restored — the first turn will decode the full
  /// rendered prompt from scratch. For cross-launch KV reuse, use the
  /// `cache:` overload above instead.
  public init(
    context: LlamaContext,
    modelName: String,
    instructions: String? = nil,
    history: [Tokenizers.Message],
    generateParameters: GenerateParameters = .init(),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
    extraEOSTokens: Set<String> = [],
  ) {
    self.context = context
    self.modelName = modelName
    self.instructions = instructions
    self.generateParameters = generateParameters
    self.additionalContext = additionalContext
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.format = format
    self.extraEOSTokens = extraEOSTokens
    let filtered = Self.droppingSystemMessages(history)
    stateStore = SessionCacheStore<State>(State(history: filtered, cachedKVTokens: []))
  }

  // MARK: Public stream entry points

  public func streamResponseEvents(
    prompt: String,
    role: Role = .user,
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<ResponseStreamingEvent, Error> {
    runTurn(prompt: prompt, role: role.rawValue, config: config) { events in events }
  }

  public func streamResponseItems(
    prompt: String,
    role: Role = .user,
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<[ResponseOutputItem], Error> {
    let accumulator = TurnItemsBox()
    return runTurn(prompt: prompt, role: role.rawValue, config: config) { events in
      accumulator.ingest(events)
      return [accumulator.snapshot]
    }
  }

  /// Streams the assistant's reply as plain text chunks, filtering out
  /// reasoning, tool calls, and lifecycle events. Use
  /// ``streamResponseEvents(prompt:role:config:)`` for the full typed
  /// event stream, or ``respond(to:role:config:)`` then `.outputText`
  /// for the whole reply at once.
  public func streamText(
    prompt: String,
    role: Role = .user,
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<String, Error> {
    runTurn(prompt: prompt, role: role.rawValue, config: config) { events in
      events.compactMap { event in
        if case let .outputTextDelta(e) = event { return e.delta }
        return nil
      }
    }
  }

  public func respond(
    to prompt: String,
    role: Role = .user,
    config: ResponseStreamConfig? = nil,
  ) async throws -> Response {
    var terminal: Response?
    for try await event in streamResponseEvents(prompt: prompt, role: role, config: config) {
      if let response = event.terminalResponse {
        terminal = response
      }
    }
    guard let terminal else {
      throw ResponseChatSessionError.passDidNotFinish
    }
    return terminal
  }

  /// Clear history and reset the KV cache.
  ///
  /// **Blocking.** Awaits the session lock; if a turn is mid-stream this
  /// call suspends until the turn finishes (potentially many seconds for
  /// a long generation). It does NOT interrupt the in-flight turn — to
  /// stop a turn, cancel its consumer `Task` first, then call `clear`.
  ///
  /// Do not call from inside a running `toolDispatch` closure (would
  /// deadlock — the dispatch holds the session lock).
  public func clear() async {
    let context = context
    // `try?` — if the caller's task is cancelled while waiting for
    // the session lock, abandon the clear silently rather than
    // bubbling the cancellation. Cleanup operations are best-effort
    // by convention and a thrown public API would surprise callers.
    try? await stateStore.update { state in
      state.history = []
      state.cachedKVTokens = []
      await context.clearCache()
    }
  }

  /// Drop the cached KV state without clearing conversation history.
  /// The next turn will re-decode the full prompt from scratch rather
  /// than prefix-reusing the previous turn's cache. Mostly useful for
  /// tests verifying that prefix reuse doesn't change output, and for
  /// recovering from a known-bad cache after an external mutation
  /// (e.g. the caller invoked `clearCache` on the underlying context
  /// directly).
  ///
  /// **Blocking.** Awaits the session lock — same semantics as
  /// ``clear()``.
  public func discardCachedPrefix() async {
    let context = context
    try? await stateStore.update { state in
      state.cachedKVTokens = []
      await context.clearCache()
    }
  }

  /// Wait until in-flight turns drain.
  public func synchronize() async {
    try? await stateStore.read { _ in }
  }

  // MARK: Restart loop

  private func runTurn<Element: Sendable>(
    prompt: String,
    role: String,
    config: ResponseStreamConfig?,
    transform: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
  ) -> AsyncThrowingStream<Element, Error> {
    let (stream, continuation) = AsyncThrowingStream<Element, Error>.makeStream()

    let userMessage: Tokenizers.Message = ["role": role, "content": prompt]
    let userBox = SendableBox(userMessage)

    let task = Task { [
      context, modelName, instructions,
      additionalContext, tools, toolDispatch, generateParameters,
      format, extraEOSTokens, lastResponseBox, stateStore,
    ] in
      do {
        try await stateStore.update { state in
          let tokenizer = LibLlamaTokenizer(model: context.model)
          let resolvedConfig = config ?? ResponseStreamConfig(
            model: modelName,
            instructions: instructions,
            tools: tools ?? [],
          )

          // Build the working history for this turn: start with whatever
          // exists, optionally prepend system instructions, then append
          // the user's new message.
          var workingHistory: [Tokenizers.Message] = []
          if let instructions, !state.history.contains(where: { ($0["role"] as? String) == "system" }) {
            workingHistory.append(["role": "system", "content": instructions])
          }
          workingHistory.append(contentsOf: state.history)
          workingHistory.append(userBox.consume())

          let envelope = ResponseTurnEnvelope(config: resolvedConfig)
          let usageBox = UsageBox()
          let itemsBox = TurnItemsBox()
          var dispatchedCallIds: Set<String> = []
          var lastFinishInfo: PassFinishInfo?

          for batch in transform(envelope.start()) {
            continuation.yield(batch)
          }

          // Track tokens we expect to be in the KV. Starts at whatever
          // was cached from the previous turn (or empty on first turn).
          //
          // The defer block commits the snapshot back to `state` on
          // every exit path — clean finish, cancellation, or thrown
          // error. Without this, a cancelled or thrown turn would
          // leave the underlying KV memory mutated (with decoded
          // prompt + partial generation tokens) while `state.cachedKVTokens`
          // stayed empty, so the next turn's LCP machinery would skip
          // truncating the stale KV and llama_decode would fail with
          // "inconsistent sequence positions."
          var cachedKV = state.cachedKVTokens
          defer { state.cachedKVTokens = cachedKV }

          restart: while true {
            if Task.isCancelled { break restart }

            // Render the prompt through the chat template. If it
            // overflows the model's context window minus the
            // generation budget, drop the earliest non-system,
            // non-current-user-turn message and re-render until it
            // fits. The earliest entry is normally an old user turn
            // (or the assistant's reply to it); the loop preserves the
            // current user message at the tail so it always ships.
            let promptIds = try Self.renderWithinBudget(
              workingHistory: &workingHistory,
              model: context.model,
              tools: tools,
              additionalContext: additionalContext,
              budget: Int(context.contextLength) - generateParameters.maxTokens,
            )

            // Find the longest prefix of `promptIds` that's already in
            // the KV cache (matches `cachedKV` byte-for-byte). Truncate
            // the KV beyond that prefix; runOnePass will decode only
            // the new suffix.
            var lcp = longestCommonPrefixLength(cachedKV, promptIds)
            if lcp < cachedKV.count {
              // Some cached tail diverged from the new prompt — drop it.
              // `keepTokens(count:)` is `llama_memory_seq_rm(seq, count, -1)`:
              // remove positions `[count, ∞)`, keeping `[0, count)`. If
              // it returns false (e.g. the model uses a recurrent state
              // that can't be partially truncated), wipe the cache and
              // re-decode the full prompt — correct but slow, vs.
              // silently letting `llama_decode` fail at the next call.
              let kept = await context.keepTokens(count: LlamaPosition(lcp))
              if !kept {
                await context.clearCache()
                cachedKV = []
                lcp = 0
              }
            }

            // Collect the tokens the engine yields so we can update the
            // cached-KV snapshot for the next pass. After this pass the
            // KV holds `promptIds` followed by all yielded generation
            // tokens (excluding EOS/EOT, which halt without being
            // decoded into the cache).
            let yieldedBox = YieldedTokensBox()

            // Snapshot the turn-wide item count so we can slice out the
            // delta this pass produced. Without this, the assistant
            // message reconstruction below would re-include every prior
            // pass's `output_text`, duplicating earlier content into
            // workingHistory.
            let priorItemCount = itemsBox.snapshot.count

            let passInfo = try await runOnePass(
              on: context,
              prompt: promptIds,
              tokenizer: tokenizer,
              generateParameters: generateParameters,
              modelName: modelName,
              tools: tools ?? [],
              envelope: envelope,
              transform: transform,
              continuation: continuation,
              usageBox: usageBox,
              itemsBox: itemsBox,
              format: format,
              extraEOSTokens: extraEOSTokens,
              cachedPrefixLength: lcp,
              tokenSink: { id in yieldedBox.append(id) },
            )
            lastFinishInfo = passInfo
            cachedKV = promptIds + yieldedBox.snapshot.map(Int32.init)
            // The engine yields each sampled token before deciding
            // whether to decode it. On `.length` exits and on `.stop`
            // with `includeStopTokenInStream`, the trailing yielded
            // token is not in KV — trim to match the authoritative
            // engine position count.
            let kvCount = await Int(context.seqPosMax()) + 1
            let overcount = cachedKV.count - kvCount
            assert(
              overcount == 0 || overcount == 1,
              "cachedKV drift: \(cachedKV.count) tracked vs \(kvCount) actual",
            )
            if overcount > 0 {
              cachedKV.removeLast(overcount)
            }

            if Task.isCancelled { break restart }

            // Reconstruct an assistant message from this pass's items so
            // the next pass's chat template sees what the model just
            // emitted. Slice from `priorItemCount` to exclude prior
            // passes' items (whose text + completed function calls
            // already shipped as their own assistant messages).
            let allItems = itemsBox.snapshot
            let passItems = Array(allItems[priorItemCount...])
            let assistantMessage = MessageBuilder.assistantMessage(
              from: passItems,
              alreadyDispatched: dispatchedCallIds,
            )
            workingHistory.append(assistantMessage)

            // Dispatch any new function calls.
            let pending: [ResponseFunctionToolCall] = passItems.compactMap { item in
              guard case let .functionCall(call) = item else { return nil }
              guard call.status == .completed else { return nil }
              guard !dispatchedCallIds.contains(call.callId) else { return nil }
              return call
            }

            guard let toolDispatch, !pending.isEmpty else {
              break restart
            }

            for call in pending {
              if Task.isCancelled { break restart }
              let result: ResponseFunctionCallOutput.Output
              do {
                result = try await toolDispatch(call)
              } catch {
                // Surface non-cancellation dispatch failures to the
                // model as the function's output so it can apologize
                // / retry / continue. The alternative — throwing —
                // kills the turn mid-flight, strands the orphan
                // function_call in the consumer's history with no
                // paired output, and matches neither the OpenAI spec
                // nor common SDK behavior.
                if error is CancellationError { throw error }
                result = .string("Error: \(error.localizedDescription)")
              }
              if Task.isCancelled { break restart }
              dispatchedCallIds.insert(call.callId)
              workingHistory.append(MessageBuilder.toolMessage(for: call, result: result))
              let toolEvents = envelope.emitToolResult(.init(
                id: IDFactory.make(.functionCallOutput),
                callId: call.callId,
                output: result,
              ))
              itemsBox.ingest(toolEvents)
              for batch in transform(toolEvents) {
                continuation.yield(batch)
              }
            }
          }

          if Task.isCancelled {
            continuation.finish()
            return
          }

          // Every reachable break-restart path runs after at least
          // one `runOnePass` returns. Throw a typed error rather than
          // synthesizing telemetry so any regression surfaces with
          // diagnostic detail.
          guard let finishReason = lastFinishInfo?.finishReason else {
            throw ResponseChatSessionError.passInfoMissing
          }
          let terminal = envelope.finalize(
            info: usageBox.finalInfo(finishReason: finishReason),
          )
          var terminalResponse: Response?
          for event in terminal {
            if let response = event.terminalResponse {
              terminalResponse = response
            }
          }
          for batch in transform(terminal) {
            continuation.yield(batch)
          }
          if let terminalResponse {
            lastResponseBox.set(terminalResponse)
          }

          // Commit the working history (minus the prepended system
          // message, which lives in `instructions`). `cachedKVTokens`
          // is committed by the `defer` above on every exit path.
          state.history = workingHistory.filter {
            ($0["role"] as? String) != "system"
          }

          continuation.finish()
        }
      } catch is CancellationError {
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }

    continuation.onTermination = { _ in
      task.cancel()
    }

    return stream
  }

  /// Drop any `role: "system"` entries from a history list. System
  /// messages live on the session's ``instructions`` property and
  /// are prepended at render time; the empty-init invariant is "no
  /// system entries in `state.history`", and the `history:` inits
  /// enforce it here.
  private static func droppingSystemMessages(
    _ history: [Tokenizers.Message],
  ) -> [Tokenizers.Message] {
    history.filter { ($0["role"] as? String) != "system" }
  }

  // MARK: History reconstruction

  /// Wrapper around ``renderHistoryWithinBudget`` returning `[Int32]`
  /// (the form `LlamaContext.generate` consumes).
  private static func renderWithinBudget(
    workingHistory: inout [Tokenizers.Message],
    model: LlamaModel,
    tools: [ToolSpec]?,
    additionalContext: [String: any Sendable]?,
    budget: Int,
  ) throws -> [Int32] {
    let result = try renderHistoryWithinBudget(
      workingHistory: &workingHistory,
      messageOf: { $0 },
      model: model,
      tools: tools,
      additionalContext: additionalContext,
      budget: budget,
    )
    return result.tokens.map(Int32.init)
  }
}

/// Render `workingHistory` through the chat template embedded in the GGUF
/// and ensure the tokenized result stays under `budget` tokens. Drops the
/// earliest non-system entry (preserving the tail user turn) and
/// re-renders until it fits. Mutates `workingHistory` in place.
///
/// Generic over the history element so text sessions (Tokenizers.Message)
/// and multimodal sessions (HistoryEntry) share the trim algorithm;
/// `messageOf` projects each element to the chat-template dict.
///
/// Returns both the rendered string and its token IDs — the text session
/// uses tokens for `LlamaContext.generate`, the multimodal session passes
/// the rendered string to `MultimodalInput`.
///
/// Drops the earliest non-system entry plus the rest of its turn block
/// (any assistant + tool messages up to the next user turn). Dropping
/// a `user` in isolation while leaving its `assistant_with_tool_calls`
/// + `tool_result` follow-ups strands those at the head of the new
/// history, which Jinja templates for tool-using models (Qwen,
/// gpt-oss, Llama 3.x) generally reject.
///
/// Throws ``/Llama/LlamaError/missingChatTemplate`` when the GGUF has no
/// embedded template, or
/// ``ResponseChatSessionError/promptExceedsContext(promptTokens:budget:)``
/// when even system + last-user-turn alone don't fit.
func renderHistoryWithinBudget<Element>(
  workingHistory: inout [Element],
  messageOf: (Element) -> Tokenizers.Message,
  model: LlamaModel,
  tools: [ToolSpec]?,
  additionalContext: [String: any Sendable]?,
  budget: Int,
) throws -> (rendered: String, tokens: [Int]) {
  guard let template = model.defaultChatTemplate else {
    throw LlamaError.missingChatTemplate
  }
  let specialTokens = ChatTemplate.SpecialTokens(
    bos: model.tokenText(for: model.bosToken),
    eos: model.tokenText(for: model.eosToken),
  )
  while true {
    let messages = workingHistory.map(messageOf)
    let rendered = try ChatTemplate.render(
      template: template,
      messages: messages,
      tools: tools,
      additionalContext: additionalContext,
      specialTokens: specialTokens,
    )
    let tokens = try model.tokenize(rendered, addSpecial: false, parseSpecial: true).map(Int.init)
    if tokens.count <= budget {
      return (rendered, tokens)
    }
    let lastIndex = workingHistory.count - 1
    var dropIndex: Int?
    for (i, entry) in workingHistory.enumerated() {
      if i == lastIndex { break }
      if (messageOf(entry)["role"] as? String) == "system" { continue }
      dropIndex = i
      break
    }
    guard let dropIndex else {
      throw ResponseChatSessionError.promptExceedsContext(
        promptTokens: tokens.count,
        budget: budget,
      )
    }
    var dropEnd = dropIndex
    while dropEnd + 1 < lastIndex {
      let nextRole = messageOf(workingHistory[dropEnd + 1])["role"] as? String
      if nextRole == "user" || nextRole == "system" { break }
      dropEnd += 1
    }
    workingHistory.removeSubrange(dropIndex ... dropEnd)
  }
}

/// Lock-protected accumulator for token IDs the engine yields during a pass.
/// Used by the session to reconstruct what's now in the KV cache so the
/// next turn can prefix-reuse it.
final class YieldedTokensBox: @unchecked Sendable {
  private let lock = NSLock()
  private var tokens: [Int] = []

  func append(_ id: Int) {
    lock.lock(); defer { lock.unlock() }
    tokens.append(id)
  }

  var snapshot: [Int] {
    lock.lock(); defer { lock.unlock() }
    return tokens
  }
}

extension ResponseFunctionCallOutput.Output {
  var toolMessageText: String {
    switch self {
      case let .string(text):
        text
      case let .content(parts):
        parts.compactMap { part -> String? in
          switch part {
            case let .inputText(text):
              return text.text
            case let .inputImage(image):
              if let imageURL = image.imageURL { return "[image: \(imageURL)]" }
              if let fileId = image.fileId { return "[image: \(fileId)]" }
              return "[image]"
            case let .inputFile(file):
              if let filename = file.filename { return "[file: \(filename)]" }
              if let fileURL = file.fileURL { return "[file: \(fileURL)]" }
              if let fileId = file.fileId { return "[file: \(fileId)]" }
              return "[file]"
          }
        }.joined(separator: "\n")
    }
  }
}
