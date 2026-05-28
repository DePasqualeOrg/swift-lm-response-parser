// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses
import Tokenizers

/// Multi-turn chat session over a multimodal model. Mirrors
/// ``ResponseChatSession`` but accepts image attachments per user turn
/// and dispatches generation through ``/Llama/LlamaMtmdContext``.
///
/// **Marker convention.** The session renders messages through the
/// model's chat template, which inserts the model's own image marker
/// token (e.g. `<|image|>` for Gemma 4). The caller must configure the
/// ``/Llama/LlamaMtmdContext`` to recognize that exact marker via
/// ``/Llama/MtmdContextParameters/mediaMarker``; otherwise mtmd will look for
/// its default `<__media__>` instead and the markers won't match.
///
/// **Cache lifecycle.** Chunk-level KV prefix reuse via
/// ``/Llama/MtmdChunkSignature``: the new turn's prepared chunks are matched
/// element-wise against the prior turn's, and the matching prefix is
/// kept in KV (with intra-chunk token-LCP on the first divergent text
/// chunk). Image/audio chunks are opaque — only chunk-level reuse.
///
/// Not thread-safe — one task per session.
///
/// **Cancellation.** Cancelling a turn closes the stream silently.
/// *History* is committed only on clean finish — cancellation leaves
/// the conversation transcript unchanged. *Chunk signatures* and
/// the underlying KV cache are RESET to empty on any non-clean exit,
/// because mtmd chunk evaluation isn't a token-by-token process and
/// we can't pinpoint how many positions made it into KV before the
/// interruption. This is a deliberate asymmetry with
/// ``ResponseChatSession``, which commits the partial KV/snapshot
/// pair on cancel (text generation has per-token granularity for
/// reconstructing "what's in KV"). The trade-off here is: cancelled
/// multimodal turns lose all prefix-reuse benefit for the next turn
/// (full re-encode of every image), but the next turn is guaranteed
/// to start from a consistent state — no `llama_decode` failures
/// from stale chunk positions.
public final class MultimodalResponseChatSession: @unchecked Sendable {
  /// Versioned envelope written to disk as the `.sigs` sidecar. The
  /// version is checked on load; mismatches throw
  /// ``ResponseChatSessionError/cacheVersionMismatch(found:expected:)``
  /// rather than attempting a best-effort decode that could silently
  /// produce wrong KV-reuse decisions. Bump `currentVersion` whenever
  /// the signature wire format changes in a way that's incompatible
  /// with prior caches.
  private struct CacheSidecar: Codable {
    static let currentVersion = 1
    let version: Int
    let signatures: [MtmdChunkSignature]
  }

  private let context: LlamaMtmdContext
  private let modelName: String

  public var instructions: String?
  public var generateParameters: GenerateParameters
  public var additionalContext: [String: any Sendable]?
  public var tools: [ToolSpec]?
  public var toolDispatch: ToolDispatch?
  public var format: ResponseFormat?
  public var extraEOSTokens: Set<String>

  public typealias ToolDispatch =
    @Sendable (ResponseFunctionToolCall) async throws -> ResponseFunctionCallOutput.Output

  /// One entry in a multimodal conversation history. The `message` is a
  /// swift-tokenizers chat-template dict; for user turns that included
  /// images, the images live alongside on `images` because the dict's
  /// `content` array is rendered by Jinja — the images themselves don't
  /// survive the round-trip and must be re-attached on every turn so
  /// mtmd can swap them back into embeddings at the image-marker
  /// positions.
  ///
  /// **No `.system` factory.** System messages should be passed to the
  /// session via the ``instructions`` property, not via history. Any
  /// `role: "system"` entries supplied through `history:` are dropped
  /// by both `init(history:)` constructors so the on-the-wire turn
  /// shape stays consistent across re-renders.
  public struct HistoryEntry: Sendable {
    public var message: Tokenizers.Message
    public var images: [VisionImage]

    public init(message: Tokenizers.Message, images: [VisionImage] = []) {
      self.message = message
      self.images = images
    }

    /// Build a user-role entry. When `images` is non-empty, the message
    /// content is structured as a Jinja parts array (one `{"type":"image"}`
    /// per image, followed by `{"type":"text", "text": ...}`) so the chat
    /// template renders the model's image-marker tokens at the right
    /// positions. The images themselves are stored on the entry for the
    /// mtmd input on every replay.
    public static func user(text: String, images: [VisionImage] = []) -> HistoryEntry {
      HistoryEntry(message: userMessage(text: text, imageCount: images.count), images: images)
    }

    /// Build an assistant-role entry. Plain text only (no images).
    public static func assistant(text: String) -> HistoryEntry {
      HistoryEntry(message: ["role": "assistant", "content": text], images: [])
    }
  }

  fileprivate struct State {
    var history: [HistoryEntry] = []
    /// Chunk signatures from the most recently evaluated turn. The next
    /// turn's signatures are compared element-wise against this to find
    /// the longest matching prefix; chunks in the prefix don't need to
    /// be re-encoded.
    var cachedSignatures: [MtmdChunkSignature] = []
  }

  private let stateStore: SessionCacheStore<State>

  public var lastResponse: Response? {
    lastResponseBox.current
  }

  private let lastResponseBox = LastResponseBox()

  public init(
    context: LlamaMtmdContext,
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

  /// Persist the session's current KV cache + chunk signatures to disk
  /// so a future session can resume without re-encoding images or
  /// re-decoding the text prompt prefix.
  ///
  /// Two files are written: `url` (the llama.cpp KV state file, format
  /// compatible with llama-server's session files) and `url + ".sigs"`
  /// (a JSON sidecar holding the chunk signatures). Both are restored
  /// by ``init(context:modelName:instructions:history:cache:generateParameters:additionalContext:tools:toolDispatch:format:extraEOSTokens:)``.
  ///
  /// History (the chat-template messages, including images) is **not**
  /// in either file; the caller persists `history` separately and
  /// passes it at re-load time. Acquires the session lock.
  public func saveCache(to url: URL) async throws {
    let context = context
    try await stateStore.read { state in
      // Two-file save with best-effort cross-file atomicity. Pure
      // filesystem primitives can't perfectly atomically swap two
      // files (no directory-rename here), but we minimize the
      // inconsistency window and preserve the user's prior pair on
      // any failure.
      //
      // Order:
      // 1. Encode sigs in memory (fail-fast on serialization).
      // 2. Write KV to a temp path (preserves any existing KV at `url`).
      // 3. Write sigs atomically to its final path.
      // 4. Move KV temp into final path.
      //
      // The only remaining inconsistency window is between step 3 and
      // step 4 — if (4) fails, disk briefly holds new sigs + old KV
      // until the next save retries. Rename-after-successful-write
      // almost never fails on the same volume, so this is a tiny tail
      // risk vs. the alternatives (writing KV first leaves the prior
      // session's data poisoned by partial overwrite on KV-write
      // failure; writing sigs first without a KV temp leaves
      // new-sigs + stale-KV on KV failure).
      //
      // llama.cpp's state file embeds a token list as caller-side
      // metadata (stored separately from the KV blob, not used to
      // validate the load). For multimodal, image/audio positions
      // don't correspond to text tokens, so any pseudo-token list we
      // pass here would be misleading — the `.sigs` sidecar is the
      // authoritative source of "what's at each KV position." Pass
      // an empty list.
      let sigsURL = url.appendingPathExtension("sigs")
      let kvTempURL = url.appendingPathExtension("kvtmp")
      let sidecar = CacheSidecar(
        version: CacheSidecar.currentVersion,
        signatures: state.cachedSignatures,
      )
      let sigsData = try JSONEncoder().encode(sidecar)

      try await context.textContext.saveCache(to: kvTempURL, tokens: [])

      do {
        try sigsData.write(to: sigsURL, options: .atomic)
        // `replaceItemAt` is an atomic rename (POSIX `rename(2)`), so
        // `url` always points at either the prior KV file or the new
        // one — never absent. `moveItem` for the no-prior-file case.
        if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
          _ = try FileManager.default.replaceItemAt(url, withItemAt: kvTempURL)
        } else {
          try FileManager.default.moveItem(at: kvTempURL, to: url)
        }
      } catch {
        try? FileManager.default.removeItem(at: kvTempURL)
        throw error
      }
    }
  }

  /// Initialize with a pre-existing multimodal conversation history and
  /// a cache pair previously written by ``saveCache(to:)``. The KV state
  /// for the matched chunk prefix is restored from `cache`, and the
  /// signature sidecar (`cache + ".sigs"`) is loaded so the first turn
  /// can compute LCP against it.
  public init(
    context: LlamaMtmdContext,
    modelName: String,
    instructions: String? = nil,
    history: [HistoryEntry],
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
    // Validate the sigs sidecar fully (file present, decodable,
    // version-matched) BEFORE calling loadCache. The text context is
    // shared mutable state held by the caller; if we touched its KV
    // first and then threw on a bad sidecar, the init would leave the
    // caller's context in a populated-but-unclaimed state that they
    // can't easily detect or recover from.
    let sigsURL = cacheURL.appendingPathExtension("sigs")
    let sigsData = try Data(contentsOf: sigsURL)
    let sidecar = try JSONDecoder().decode(CacheSidecar.self, from: sigsData)
    guard sidecar.version == CacheSidecar.currentVersion else {
      throw ResponseChatSessionError.cacheVersionMismatch(
        found: sidecar.version,
        expected: CacheSidecar.currentVersion,
      )
    }
    let signatures = sidecar.signatures

    _ = try await context.textContext.loadCache(
      from: cacheURL,
      capacity: Int(context.textContext.contextLength),
    )

    let filtered = Self.droppingSystemMessages(history)
    stateStore = SessionCacheStore<State>(
      State(history: filtered, cachedSignatures: signatures),
    )
  }

  /// Initialize with a pre-existing multimodal conversation history.
  ///
  /// Each ``HistoryEntry`` carries a swift-tokenizers chat-template
  /// message and any images attached to that turn. System messages
  /// should be passed via `instructions` rather than included here; any
  /// `"role": "system"` entries are dropped to match the empty-init
  /// invariant.
  ///
  /// No KV cache state is restored — the first turn will re-encode every
  /// image and re-decode the full rendered prompt from scratch.
  public init(
    context: LlamaMtmdContext,
    modelName: String,
    instructions: String? = nil,
    history: [HistoryEntry],
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
    stateStore = SessionCacheStore<State>(State(history: filtered, cachedSignatures: []))
  }

  // MARK: Public stream entry points

  public func streamResponseEvents(
    prompt: String,
    images: [VisionImage] = [],
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<ResponseStreamingEvent, Error> {
    runTurn(prompt: prompt, images: images, config: config) { events in events }
  }

  public func streamResponseItems(
    prompt: String,
    images: [VisionImage] = [],
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<[ResponseOutputItem], Error> {
    let accumulator = TurnItemsBox()
    return runTurn(prompt: prompt, images: images, config: config) { events in
      accumulator.ingest(events)
      return [accumulator.snapshot]
    }
  }

  /// Streams the assistant's reply as plain text chunks, filtering out
  /// reasoning, tool calls, and lifecycle events. Use
  /// ``streamResponseEvents(prompt:images:config:)`` for the full typed
  /// event stream, or ``respond(to:images:config:)`` then `.outputText`
  /// for the whole reply at once.
  public func streamText(
    prompt: String,
    images: [VisionImage] = [],
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<String, Error> {
    runTurn(prompt: prompt, images: images, config: config) { events in
      events.compactMap { event in
        if case let .outputTextDelta(e) = event { return e.delta }
        return nil
      }
    }
  }

  public func respond(
    to prompt: String,
    images: [VisionImage] = [],
    config: ResponseStreamConfig? = nil,
  ) async throws -> Response {
    var terminal: Response?
    for try await event in streamResponseEvents(prompt: prompt, images: images, config: config) {
      if let response = event.terminalResponse {
        terminal = response
      }
    }
    guard let terminal else { throw ResponseChatSessionError.passDidNotFinish }
    return terminal
  }

  /// Clear history, chunk signatures, and the underlying KV cache.
  ///
  /// **Blocking.** Awaits the session lock; if a turn is mid-stream this
  /// suspends until the turn finishes. Cancel the consuming `Task`
  /// first to interrupt the in-flight turn.
  public func clear() async {
    let context = context
    // Best-effort: swallow cancellation rather than throwing from
    // a public non-throwing API. Matches the text session's clear().
    try? await stateStore.update { state in
      state.history = []
      state.cachedSignatures = []
      await context.textContext.clearCache()
    }
  }

  /// Drop the cached chunk signatures without clearing conversation
  /// history. The next turn will re-encode every chunk from scratch.
  /// Useful for tests and recovery after external context mutation.
  ///
  /// **Blocking.** Awaits the session lock — same semantics as
  /// ``clear()``.
  public func discardCachedPrefix() async {
    let context = context
    try? await stateStore.update { state in
      state.cachedSignatures = []
      await context.textContext.clearCache()
    }
  }

  public func synchronize() async {
    try? await stateStore.read { _ in }
  }

  // MARK: Turn loop

  private func runTurn<Element: Sendable>(
    prompt: String,
    images: [VisionImage],
    config: ResponseStreamConfig?,
    transform: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
  ) -> AsyncThrowingStream<Element, Error> {
    let (stream, continuation) = AsyncThrowingStream<Element, Error>.makeStream()

    let userEntry = HistoryEntry(
      message: Self.userMessage(text: prompt, imageCount: images.count),
      images: images,
    )
    let entryBox = SendableBox(userEntry)

    let task = Task { [
      context, modelName, instructions,
      additionalContext, tools, toolDispatch, generateParameters, format,
      extraEOSTokens, lastResponseBox, stateStore,
    ] in
      do {
        try await stateStore.update { state in
          let tokenizer = LibLlamaTokenizer(model: context.textContext.model)
          let resolvedConfig = config ?? ResponseStreamConfig(
            model: modelName,
            instructions: instructions,
            tools: tools ?? [],
          )

          // Build the working history: optional system message, prior
          // turns, then this turn's user entry.
          var workingHistory: [HistoryEntry] = []
          if let instructions, !state.history.contains(where: { ($0.message["role"] as? String) == "system" }) {
            workingHistory.append(HistoryEntry(
              message: ["role": "system", "content": instructions],
              images: [],
            ))
          }
          workingHistory.append(contentsOf: state.history)
          workingHistory.append(entryBox.consume())

          // One envelope/items/usage trio for the whole turn — each
          // restart pass appends into them so the final envelope
          // reflects everything the model + tool dispatch produced.
          let envelope = ResponseTurnEnvelope(config: resolvedConfig)
          let usageBox = UsageBox()
          let itemsBox = TurnItemsBox()
          var dispatchedCallIds: Set<String> = []
          var lastFinishInfo: PassFinishInfo?

          // Tracks the signature snapshot reflecting what's currently
          // in the KV cache. Updated after each successful pass. The
          // defer commits this to `state` on clean exit, or resets to
          // empty on any thrown exit (cancellation / error) — multimodal
          // can't pinpoint partial-decode progress, so we err on the
          // side of dropping prefix-cache info on any non-clean exit.
          var committedSignatures = state.cachedSignatures
          var didCommitTurn = false
          defer {
            state.cachedSignatures = didCommitTurn ? committedSignatures : []
          }

          for batch in transform(envelope.start()) {
            continuation.yield(batch)
          }

          restart: while true {
            try Task.checkCancellation()

            // Render the current working history through the chat
            // template. If it overflows the model's context window
            // minus the generation budget, drop the earliest entry
            // (along with any images attached to it) and re-render
            // until it fits. The current user turn at the tail always
            // ships.
            let renderedPrompt = try Self.renderWithinBudget(
              workingHistory: &workingHistory,
              model: context.textContext.model,
              tools: tools,
              additionalContext: additionalContext,
              budget: Int(context.textContext.contextLength) - generateParameters.maxTokens,
            )

            let allImages = workingHistory.flatMap(\.images)
            let input = MultimodalInput(
              prompt: renderedPrompt,
              media: allImages.map { .image($0) },
              addSpecialTokens: false,
              parseSpecialTokens: true,
            )

            let prepared = try await context.prepare(input: input)
            let lcpChunks = longestCommonPrefixLength(committedSignatures, prepared.signatures)
            var effectiveNPast: LlamaPosition = 0
            for sig in prepared.signatures.prefix(lcpChunks) {
              effectiveNPast += sig.nPos
            }
            var effectiveStartChunk = lcpChunks

            // No committed signatures means the LCP-truncation
            // branches below are skipped entirely, so any KV residue
            // (e.g. from a cancelled prior turn) would survive past
            // the new turn's mtmd write head. mtmd's causal mask
            // happens to hide such residue, but that's an
            // implementation detail; clear here so correctness
            // doesn't depend on it.
            if committedSignatures.isEmpty {
              await context.textContext.clearCache()
            }

            // Intra-text-chunk reuse: if the first divergent chunk on
            // both sides is `.text`, the new chunk often shares a token
            // prefix with the old one (e.g. multi-turn vision chats
            // where text grows by one user message). Keep that prefix
            // in KV, decode only the new tokens, and skip the chunk in
            // mtmd's eval. For non-text divergences (image/audio) the
            // chunk is opaque — fall back to chunk-level reuse only.
            if lcpChunks < committedSignatures.count,
               lcpChunks < prepared.signatures.count,
               case let .text(oldTokens) = committedSignatures[lcpChunks].kind,
               case let .text(newTokens) = prepared.signatures[lcpChunks].kind
            {
              let tokenLCP = longestCommonPrefixLength(oldTokens, newTokens)
              let keepCount = effectiveNPast + LlamaPosition(tokenLCP)
              let kept = await context.textContext.keepTokens(count: keepCount)
              if !kept {
                // Model doesn't support partial KV truncation. Fall
                // back to a full re-eval — correct but loses all
                // prefix-reuse for this turn.
                await context.textContext.clearCache()
                committedSignatures = []
                effectiveNPast = 0
                effectiveStartChunk = 0
              } else if tokenLCP > 0 {
                if tokenLCP < newTokens.count {
                  // Decode the suffix of this text chunk directly via
                  // the text engine — bypassing mtmd for this one chunk
                  // since we only want a partial eval, not the full
                  // chunk's worth.
                  let suffix = Array(newTokens[tokenLCP...])
                  try await context.textContext.decode(
                    tokens: suffix,
                    startingAt: effectiveNPast + LlamaPosition(tokenLCP),
                    logitsOnLastOnly: true,
                  )
                }
                effectiveStartChunk = lcpChunks + 1
                effectiveNPast += prepared.signatures[lcpChunks].nPos
              }
            } else if lcpChunks < committedSignatures.count {
              // No intra-chunk match available — drop the diverged tail
              // (matches the original chunk-level behavior). Same
              // fallback as above if keepTokens is rejected.
              let kept = await context.textContext.keepTokens(count: effectiveNPast)
              if !kept {
                await context.textContext.clearCache()
                committedSignatures = []
                effectiveNPast = 0
                effectiveStartChunk = 0
              }
            }

            // If the LCP fully covers the new prompt (identical resend,
            // or intra-chunk advance landed on the last chunk), mtmd
            // would no-op and sample would pull stale logits left over
            // from the prior turn's decode. Drop the final chunk's KV
            // positions so the pass has at least one chunk to evaluate
            // and sample sees fresh logits at the last position. The
            // text-only path handles the analogous case inside
            // `LlamaContext.runGeneration`.
            if effectiveStartChunk >= prepared.signatures.count, effectiveStartChunk > 0 {
              let lastSig = prepared.signatures[effectiveStartChunk - 1]
              effectiveStartChunk -= 1
              effectiveNPast -= lastSig.nPos
              let kept = await context.textContext.keepTokens(count: effectiveNPast)
              if !kept {
                await context.textContext.clearCache()
                committedSignatures = []
                effectiveNPast = 0
                effectiveStartChunk = 0
              }
            }

            // Snapshot the turn-wide item count so we can slice out the
            // delta this pass produced (see ResponseChatSession.runTurn
            // for the rationale).
            let priorItemCount = itemsBox.snapshot.count

            let passInfo = try await runOneMtmdPass(
              on: context,
              prepared: prepared,
              startingAtChunk: effectiveStartChunk,
              nPast: effectiveNPast,
              promptText: renderedPrompt,
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
            )
            lastFinishInfo = passInfo
            // KV post-pass layout:
            //
            //   [0, prepared.totalPositions)         = signature-covered
            //                                          (text-chunk tokens
            //                                          + image/audio
            //                                          embedding positions)
            //   [prepared.totalPositions,
            //     prepared.totalPositions + gen)     = generation residue
            //
            // `committedSignatures` describes only the signature-covered
            // span; generation tokens have no chunk identity, so they
            // sit anonymously past the frontier.
            //
            // On the next turn, the LCP path may decode new chunks
            // starting at some `effectiveNPast` < the current frontier.
            // mtmd writes new tokens at those positions, overwriting
            // the generation residue from this turn — but only up to
            // the new chunks' cumulative span. If the new chunks span
            // fewer positions than this turn's generation, the trailing
            // tail of residue lingers. That tail is never attended to
            // (causal masks exclude positions > current write head) and
            // gets overwritten one slot at a time as the next sample
            // loop advances. Hence "correct in practice, despite the
            // KV frontier briefly extending past what signatures
            // describe."
            committedSignatures = prepared.signatures

            try Task.checkCancellation()

            // Reconstruct an assistant message from this pass's items so
            // the next pass's chat template sees what the model emitted.
            // Slice from `priorItemCount` to exclude prior passes' items.
            let allItems = itemsBox.snapshot
            let passItems = Array(allItems[priorItemCount...])
            let assistantMessage = MessageBuilder.assistantMessage(
              from: passItems,
              alreadyDispatched: dispatchedCallIds,
            )
            workingHistory.append(HistoryEntry(message: assistantMessage, images: []))

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
              try Task.checkCancellation()
              let result: ResponseFunctionCallOutput.Output
              do {
                result = try await toolDispatch(call)
              } catch {
                // See ResponseChatSession.runTurn for the rationale —
                // surface non-cancellation tool failures to the model
                // as a function output rather than killing the turn.
                if error is CancellationError { throw error }
                result = .string("Error: \(error.localizedDescription)")
              }
              try Task.checkCancellation()
              dispatchedCallIds.insert(call.callId)
              workingHistory.append(HistoryEntry(
                message: MessageBuilder.toolMessage(for: call, result: result),
                images: [],
              ))
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

          // See ResponseChatSession.runTurn for the rationale.
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

          // Commit the working history (minus any system entry, stored
          // in `instructions`). `state.cachedSignatures` is committed
          // by the defer above.
          state.history = workingHistory.filter {
            ($0.message["role"] as? String) != "system"
          }
          didCommitTurn = true

          continuation.finish()
        }
      } catch is CancellationError {
        // The defer inside the closure already reset
        // `state.cachedSignatures = []`. Clear the underlying KV under
        // the lock so the next turn starts from a known-empty state.
        // `try?` — best-effort cleanup; if even this cleanup is
        // cancelled, the original cancellation has already done its job.
        try? await stateStore.update { _ in
          await context.textContext.clearCache()
        }
        continuation.finish()
      } catch {
        try? await stateStore.update { _ in
          await context.textContext.clearCache()
        }
        continuation.finish(throwing: error)
      }
    }

    continuation.onTermination = { _ in task.cancel() }
    return stream
  }

  /// Drop any `HistoryEntry` whose wrapped message has `role:
  /// "system"`. System messages live on the session's
  /// ``instructions`` property; this enforces the empty-init
  /// invariant at the `history:` boundary.
  private static func droppingSystemMessages(_ history: [HistoryEntry]) -> [HistoryEntry] {
    history.filter { ($0.message["role"] as? String) != "system" }
  }

  // MARK: Message construction

  /// Wrapper around ``renderHistoryWithinBudget`` returning the rendered
  /// prompt as `String` (mtmd consumes the prompt as text, re-tokenizing
  /// internally around media markers).
  ///
  /// Note: mtmd budget accounting is approximate here — text-chunk
  /// tokenization counts but image-marker tokens expand into many more
  /// positions inside mtmd. Size the context with headroom proportional
  /// to the expected image count.
  private static func renderWithinBudget(
    workingHistory: inout [HistoryEntry],
    model: LlamaModel,
    tools: [ToolSpec]?,
    additionalContext: [String: any Sendable]?,
    budget: Int,
  ) throws -> String {
    let result = try renderHistoryWithinBudget(
      workingHistory: &workingHistory,
      messageOf: \.message,
      model: model,
      tools: tools,
      additionalContext: additionalContext,
      budget: budget,
    )
    return result.rendered
  }

  private static func userMessage(text: String, imageCount: Int) -> Tokenizers.Message {
    guard imageCount > 0 else {
      return ["role": "user", "content": text]
    }
    var parts: [[String: any Sendable]] = []
    for _ in 0 ..< imageCount {
      parts.append(["type": "image"])
    }
    parts.append(["type": "text", "text": text])
    return ["role": "user", "content": parts as [any Sendable]]
  }
}
