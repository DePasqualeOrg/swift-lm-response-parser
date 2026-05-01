// Copyright © Anthony DePasquale

import CoreGraphics
import Foundation
import LMResponseParser
import MLX
import MLXLMCommon

/// Multi-turn session that mirrors `ChatSession`'s ergonomics with
/// Responses-shaped output.
///
/// One session call corresponds to one logical assistant turn and produces
/// exactly one Responses envelope, regardless of how many internal
/// generation passes happen for tool dispatch. KV cache is reused across
/// turns. Not thread-safe – one task per session, mirroring
/// `MLXLMCommon.ChatSession`.
///
/// `modelType` and `modelConfig` are explicit because mlx-swift-lm's
/// `LLMModelFactory._load` decodes `model_type` and the raw `config.json`
/// dict for dispatch and discards both. The bridge needs both to drive
/// ``/LMResponseParser/ResponseFormat/infer(modelName:modelType:modelConfig:)``.
///
/// - TODO: Drop both parameters if `ModelConfiguration` ever exposes the
///   decoded `model_type` and raw config dict directly – they become
///   derivable from the passed-in model context.
///
/// **Cancellation and the KV cache.** When a turn is cancelled while a
/// ``toolDispatch`` is in flight (or while function calls are pending
/// dispatch in the restart loop), the KV cache retains pass-1 tokens
/// – the model's own function-call output – without any synthesized
/// `tool_result` folded back in. A subsequent turn on the same session
/// reuses that cache and the model "sees" unanswered function calls
/// followed by an unrelated user prompt; behavior is model-dependent
/// (often degraded output or repeated calls). To start fresh after such
/// a cancellation, call ``clear()`` before the next turn. Same
/// underlying behavior as `MLXLMCommon.ChatSession`; surfaced more often
/// here because cancelling mid-tool-dispatch is a more natural user
/// gesture.
public final class ResponseChatSession {
  enum Cache {
    case empty
    case kvcache([KVCache])
    case history([Chat.Message])
  }

  private let model: ModelContainer
  public var instructions: String?
  private let cache: SessionCacheStore<Cache>
  public var processing: UserInput.Processing
  public var generateParameters: GenerateParameters
  public var additionalContext: [String: any Sendable]?
  public var tools: [ToolSpec]?

  /// The terminal ``/LMResponseParser/Response`` snapshot from the most
  /// recently finalized turn, or `nil` until the first turn finalizes.
  ///
  /// Useful for reading post-turn metadata that is not otherwise
  /// surfaced by ``streamResponseItems(prompt:role:images:videos:config:)``
  /// – in particular `Response.usage` (input/output token counts) and
  /// `Response.status` (`completed` / `incomplete`) and
  /// `incompleteDetails`. For `streamResponseEvents`, the same data is
  /// available on the terminal response event; this accessor is the only
  /// path for items-style consumers.
  ///
  /// Updated only after the consumer has been yielded the turn's
  /// terminal response event. A silently closed turn (consumer
  /// cancellation) or a turn that throws an error does not update
  /// this accessor – the previously stored `Response` (if any)
  /// remains. Pair reads of `lastResponse` with the outcome of the
  /// iteration loop (clean exit, throw, or external cancellation)
  /// when distinguishing fresh from stale data matters. Reading
  /// while a turn is still streaming returns the previous turn's
  /// `Response` (or `nil`).
  public var lastResponse: Response? {
    lastResponseBox.current
  }

  private let lastResponseBox = LastResponseBox()

  public typealias ToolDispatch =
    @Sendable (ResponseFunctionToolCall) async throws -> ResponseFunctionCallOutput.Output

  /// Tool-dispatch callback. Differs from `ChatSession.toolDispatch` in
  /// type: receives a Responses-native ``/LMResponseParser/ResponseFunctionToolCall`` with
  /// `arguments` as a raw JSON string. Pre-parsing in the callback
  /// signature would force a lossy round-trip through
  /// `[String: JSONValue]` for callers who would otherwise decode
  /// straight into a typed Swift struct.
  ///
  /// This callback is the semantic validation boundary for tool use.
  /// Validate the function name, decode and check arguments, enforce
  /// permissions, and verify runtime preconditions here before executing
  /// the tool. Return ordinary text and JSON results as
  /// ``/LMResponseParser/ResponseFunctionCallOutput/Output/string(_:)``;
  /// return typed text, image, or file parts as
  /// ``/LMResponseParser/ResponseFunctionCallOutput/Output/content(_:)``.
  /// The emitted Responses events preserve that typed output. For the
  /// next local MLX generation pass, the bridge renders content parts into
  /// tool-message text and attaches `input_image.image_url` values as
  /// `UserInput.Image.url` when they can be represented as URLs; hosted
  /// file IDs and file contents remain application-level concerns.
  /// Recoverable validation failures should be returned as model-visible
  /// output so the next generation pass can correct the call. Throwing
  /// from this callback fails the response stream and is best reserved for
  /// cancellation or internal failures where the turn should not continue.
  ///
  /// > Important: This callback runs while the session holds its
  /// > internal cache lock. Do **not** call ``clear()``,
  /// > ``saveCache(to:)``, or ``synchronize()`` on the same session
  /// > from inside this closure – all four entry points acquire the
  /// > same lock and a reentrant call would deadlock the running
  /// > turn. External work (HTTP calls, database lookups, etc.) is
  /// > fine.
  public var toolDispatch: ToolDispatch?

  /// Optional explicit format override, used when `model_type` and
  /// `config.json` don't disambiguate the wire format the model emits.
  public var format: ResponseFormat?

  private let modelType: String?
  private let modelConfig: [String: any Sendable]?

  public init(
    _ model: ModelContainer,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    instructions: String? = nil,
    generateParameters: GenerateParameters = .init(),
    processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
  ) {
    self.model = model
    self.instructions = instructions
    cache = .init(.empty)
    self.processing = processing
    self.generateParameters = generateParameters
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.additionalContext = additionalContext
    self.format = format
    self.modelType = modelType
    self.modelConfig = modelConfig
  }

  public init(
    _ model: ModelContext,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    instructions: String? = nil,
    generateParameters: GenerateParameters = .init(),
    processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
  ) {
    self.model = ModelContainer(context: model)
    self.instructions = instructions
    cache = .init(.empty)
    self.processing = processing
    self.generateParameters = generateParameters
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.additionalContext = additionalContext
    self.format = format
    self.modelType = modelType
    self.modelConfig = modelConfig
  }

  /// Initialize with an existing message history (prompt re-hydration).
  public init(
    _ model: ModelContainer,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    instructions: String? = nil,
    history: consuming[Chat.Message],
    generateParameters: GenerateParameters = .init(),
    processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
  ) {
    self.model = model
    self.instructions = instructions
    cache = .init(.history(history))
    self.processing = processing
    self.generateParameters = generateParameters
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.additionalContext = additionalContext
    self.format = format
    self.modelType = modelType
    self.modelConfig = modelConfig
  }

  /// Initialize with an existing message history (prompt re-hydration).
  public init(
    _ model: ModelContext,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    instructions: String? = nil,
    history: [Chat.Message],
    generateParameters: GenerateParameters = .init(),
    processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
  ) {
    self.model = ModelContainer(context: model)
    self.instructions = instructions
    cache = .init(.history(history))
    self.processing = processing
    self.generateParameters = generateParameters
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.additionalContext = additionalContext
    self.format = format
    self.modelType = modelType
    self.modelConfig = modelConfig
  }

  /// Initialize with a pre-built KV cache (prefix caching). Same caveat
  /// as `ChatSession`: don't pass `instructions` if the cache already
  /// encodes a system prompt.
  public init(
    _ model: ModelContainer,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    instructions: String? = nil,
    cache: consuming[KVCache],
    generateParameters: GenerateParameters = .init(),
    processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
  ) {
    self.model = model
    self.instructions = instructions
    self.cache = .init(.kvcache(cache))
    self.processing = processing
    self.generateParameters = generateParameters
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.additionalContext = additionalContext
    self.format = format
    self.modelType = modelType
    self.modelConfig = modelConfig
  }

  /// Initialize with a pre-built KV cache (prefix caching). Same caveat
  /// as `ChatSession`: don't pass `instructions` if the cache already
  /// encodes a system prompt.
  public init(
    _ model: ModelContext,
    modelType: String? = nil,
    modelConfig: [String: any Sendable]? = nil,
    instructions: String? = nil,
    cache: consuming[KVCache],
    generateParameters: GenerateParameters = .init(),
    processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
    additionalContext: [String: any Sendable]? = nil,
    tools: [ToolSpec]? = nil,
    toolDispatch: ToolDispatch? = nil,
    format: ResponseFormat? = nil,
  ) {
    self.model = ModelContainer(context: model)
    self.instructions = instructions
    self.cache = .init(.kvcache(cache))
    self.processing = processing
    self.generateParameters = generateParameters
    self.tools = tools
    self.toolDispatch = toolDispatch
    self.additionalContext = additionalContext
    self.format = format
    self.modelType = modelType
    self.modelConfig = modelConfig
  }

  /// Stream Responses-API events for one assistant turn. The
  /// `images:` and `videos:` parameters are required (use `[]` when
  /// there are none) to disambiguate from the singular-image
  /// convenience overload below – the same pattern
  /// `MLXLMCommon.ChatSession.streamResponse(to:role:images:videos:)` uses.
  public func streamResponseEvents(
    prompt: String,
    role: Chat.Message.Role = .user,
    images: consuming [UserInput.Image],
    videos: consuming [UserInput.Video],
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<ResponseStreamingEvent, Error> {
    runTurn(prompt: prompt, role: role, images: images, videos: videos, config: config) { events in
      events
    }
  }

  /// Single-image / single-video convenience for
  /// ``streamResponseEvents(prompt:role:images:videos:config:)``, matching
  /// `ChatSession.streamResponse(to:image:video:)`. Both `image:` and
  /// `video:` default to `nil`, so the common "just a prompt" call –
  /// `session.streamResponseEvents(prompt: "…")` – dispatches here.
  public func streamResponseEvents(
    prompt: String,
    image: UserInput.Image? = nil,
    video: UserInput.Video? = nil,
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<ResponseStreamingEvent, Error> {
    streamResponseEvents(
      prompt: prompt,
      images: image.map { [$0] } ?? [],
      videos: video.map { [$0] } ?? [],
      config: config,
    )
  }

  /// Stream Responses-API items snapshots for one assistant turn. Each
  /// yielded value is the cumulative `[ResponseOutputItem]` as of that
  /// point in the stream. `images:` and `videos:` are required;
  /// see the singular convenience overload below.
  public func streamResponseItems(
    prompt: String,
    role: Chat.Message.Role = .user,
    images: consuming [UserInput.Image],
    videos: consuming [UserInput.Video],
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<[ResponseOutputItem], Error> {
    let accumulator = TurnItemsBox()
    return runTurn(prompt: prompt, role: role, images: images, videos: videos, config: config) { events in
      accumulator.ingest(events)
      return [accumulator.snapshot]
    }
  }

  /// Single-image / single-video convenience for
  /// ``streamResponseItems(prompt:role:images:videos:config:)``.
  public func streamResponseItems(
    prompt: String,
    image: UserInput.Image? = nil,
    video: UserInput.Video? = nil,
    config: ResponseStreamConfig? = nil,
  ) -> AsyncThrowingStream<[ResponseOutputItem], Error> {
    streamResponseItems(
      prompt: prompt,
      images: image.map { [$0] } ?? [],
      videos: video.map { [$0] } ?? [],
      config: config,
    )
  }

  /// Run one assistant turn end-to-end and return its terminal
  /// ``/LMResponseParser/Response``. Non-streaming convenience – drains
  /// ``streamResponseEvents(prompt:role:images:videos:config:)`` to
  /// completion and returns the ``/LMResponseParser/Response`` carried by
  /// that turn's terminal response event.
  /// Mirrors `MLXLMCommon.ChatSession.respond(to:role:images:videos:)`.
  ///
  /// Throws whatever ``streamResponseEvents(prompt:role:images:videos:config:)``
  /// throws. A consumer cancellation that closes the stream silently
  /// throws ``ResponseChatSessionError/passDidNotFinish`` here, since
  /// no terminal response was produced.
  public func respond(
    to prompt: String,
    role: Chat.Message.Role = .user,
    images: consuming [UserInput.Image] = [],
    videos: consuming [UserInput.Video] = [],
    config: ResponseStreamConfig? = nil,
  ) async throws -> Response {
    try await Self.drainTerminalResponse(from: streamResponseEvents(
      prompt: prompt,
      role: role,
      images: images,
      videos: videos,
      config: config,
    ))
  }

  static func drainTerminalResponse(
    from stream: AsyncThrowingStream<ResponseStreamingEvent, Error>,
  ) async throws -> Response {
    var terminalResponse: Response?
    for try await event in stream {
      if let response = event.terminalResponse {
        terminalResponse = response
      }
    }
    guard let terminalResponse else {
      throw ResponseChatSessionError.passDidNotFinish
    }
    return terminalResponse
  }

  /// Single-image / single-video convenience for
  /// ``respond(to:role:images:videos:config:)``.
  public func respond(
    to prompt: String,
    image: UserInput.Image? = nil,
    video: UserInput.Video? = nil,
    config: ResponseStreamConfig? = nil,
  ) async throws -> Response {
    try await respond(
      to: prompt,
      images: image.map { [$0] } ?? [],
      videos: video.map { [$0] } ?? [],
      config: config,
    )
  }

  /// Clear the session history and cache, preserving system instructions.
  ///
  /// Acquires the session's cache lock. Do not call from inside a
  /// running ``toolDispatch`` closure – see ``toolDispatch`` for the
  /// reentrancy contract.
  public func clear() async {
    await cache.update { cache in
      cache = .empty
    }
  }

  /// Wait until the cache lock is free – i.e., until every in-flight
  /// and queued turn has fully drained.
  ///
  /// After breaking out of a `streamResponse*` iteration early,
  /// callers must `await synchronize()` before assuming the producer
  /// task and MLX's underlying generation have finished cleaning up.
  /// Without this barrier, a follow-up `saveCache(to:)`, `clear()`,
  /// or another `streamResponse*` call could race the cancelled pass.
  /// Subsequent `streamResponse*` calls also serialize on the same
  /// lock, so `synchronize()` is only needed when the next operation
  /// isn't itself a `streamResponse*` call.
  ///
  /// Implementation: acquires the cache lock briefly via `read`. If
  /// other turns are already queued behind the in-flight turn,
  /// `synchronize()` queues at the tail and returns only once every
  /// preceding holder has released. In other words, the wait is
  /// bounded by the longest contiguous chain of queued turns; do not
  /// call `synchronize()` from inside a callback that is itself
  /// blocking a queued turn.
  public func synchronize() async {
    await cache.read { _ in }
  }

  /// Save the current KV cache to disk. Throws ``ResponseChatSessionError/noCacheAvailable``
  /// if no generation has occurred yet.
  ///
  /// Acquires the session's cache lock. Do not call from inside a
  /// running ``toolDispatch`` closure – see ``toolDispatch`` for the
  /// reentrancy contract.
  public func saveCache(to url: URL) async throws {
    try await cache.read { cache in
      switch cache {
        case let .kvcache(cache):
          try savePromptCache(url: url, cache: cache)
        default:
          throw ResponseChatSessionError.noCacheAvailable
      }
    }
  }

  // MARK: Restart loop

  private func runTurn<Element: Sendable>(
    prompt: String,
    role: Chat.Message.Role,
    images: consuming [UserInput.Image],
    videos: consuming [UserInput.Video],
    config: ResponseStreamConfig?,
    transform: @escaping @Sendable ([ResponseStreamingEvent]) -> [Element],
  ) -> AsyncThrowingStream<Element, Error> {
    let (stream, continuation) = AsyncThrowingStream<Element, Error>.makeStream()

    let messageBox = SendableBox(Chat.Message(
      role: role, content: prompt, images: images, videos: videos,
    ))

    let task = Task { [
      model, instructions, processing, tools, toolDispatch,
      additionalContext, cache, generateParameters, modelType, modelConfig,
      format, lastResponseBox,
    ] in
      do {
        try await cache.update { cacheState in
          let processor = await model.processor
          let modelConfiguration = await model.configuration
          let resolvedConfig = config ?? ResponseStreamConfig(
            model: modelConfiguration.name,
            instructions: instructions,
            tools: tools ?? [],
          )

          var messages: [Chat.Message] = []
          if let instructions {
            messages.append(.system(instructions))
          }

          // `kvCache` aliases the array stored in `cacheState`:
          // `[KVCache]` is a value type but each `KVCache` is a
          // class, so passing the array around copies the
          // reference graph and `runOnePass`'s mutations to
          // each `KVCache` instance propagate back into
          // `cacheState`. This is intentional – mirrors
          // `MLXLMCommon.ChatSession`'s same pattern.
          //
          // The cache is mutated to `.kvcache(...)` *before*
          // the first pass succeeds. If `processor.prepare` or
          // the pass itself throws on the very first turn, the
          // session retains an empty (newly minted) KV cache
          // rather than rolling back to `.empty` or `.history`.
          // Subsequent turns continue from that empty cache.
          // Same behavior as upstream `ChatSession`; documented
          // here for the avoidance of doubt.
          var kvCache: [KVCache]
          switch cacheState {
            case .empty:
              kvCache = await model.perform { context in
                SendableBox(context.model.newCache(parameters: generateParameters))
              }.consume()
              cacheState = .kvcache(kvCache)

            case let .kvcache(array):
              kvCache = array

            case let .history(history):
              kvCache = await model.perform { context in
                SendableBox(context.model.newCache(parameters: generateParameters))
              }.consume()
              cacheState = .kvcache(kvCache)
              messages.append(contentsOf: history)
          }

          // Single-use: `messageBox` is consumed exactly once
          // per turn here. The restart loop appends `tool`
          // messages directly without re-consuming the box, so
          // do not move this call into the loop body.
          messages.append(messageBox.consume())

          let envelopeBox = EnvelopeBox(ResponseTurnEnvelope(config: resolvedConfig))
          let usageBox = UsageBox()
          let itemsBox = TurnItemsBox()
          var dispatchedCallIds: Set<String> = []
          var lastFinishInfo: PassFinishInfo?

          for batch in transform(envelopeBox.start()) {
            continuation.yield(batch)
          }

          restart: while !messages.isEmpty {
            if Task.isCancelled { break restart }

            let userInput = UserInput(
              chat: messages,
              processing: processing,
              tools: tools,
              additionalContext: additionalContext,
            )
            let input = try await processor.prepare(input: userInput)
            messages.removeAll()

            let passInfo = try await runOnePass(
              on: model,
              input: input,
              kvCache: kvCache,
              generateParameters: generateParameters,
              modelType: modelType,
              modelConfig: modelConfig,
              tools: tools ?? [],
              envelopeBox: envelopeBox,
              transform: transform,
              continuation: continuation,
              usageBox: usageBox,
              itemsBox: itemsBox,
              format: format,
            )
            lastFinishInfo = passInfo

            if Task.isCancelled { break restart }

            // Compute pending calls from the turn-scoped
            // accumulator: completed function-calls whose
            // call_id we haven't dispatched yet. Earlier
            // passes' calls live in `itemsBox` too, but
            // `dispatchedCallIds` filters them out.
            let pending: [ResponseFunctionToolCall] = itemsBox.snapshot.compactMap { item in
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
              let result = try await toolDispatch(call)
              // Re-check cancellation after the dispatch
              // returns. A dispatcher that completes
              // synchronously (no real suspension point)
              // would otherwise let us proceed to mutate
              // turn state and yield tool-result events
              // even after the consumer has cancelled.
              // Yields would be no-ops once the iterator
              // is gone, but `itemsBox.ingest(...)` and
              // `envelopeBox.emitToolResult(...)` would
              // still mutate session-visible state.
              if Task.isCancelled { break restart }
              dispatchedCallIds.insert(call.callId)
              messages.append(result.toolMessage())
              let toolEvents = envelopeBox.emitToolResult(.init(
                id: IDFactory.make(.functionCallOutput),
                callId: call.callId,
                output: result,
              ))
              // Tool-result events get ingested into the
              // turn-scoped accumulator too – the snapshot
              // is what subsequent passes' pending-call
              // computations read.
              itemsBox.ingest(toolEvents)
              for batch in transform(toolEvents) {
                continuation.yield(batch)
              }
            }
          }

          if Task.isCancelled {
            // Silent close on cancellation. The cache lock is
            // released by the surrounding `update` block
            // returning; we've already awaited the in-flight
            // pass's cleanup before getting here. Explicit
            // `finish()` documents the silent-close intent
            // rather than relying on the continuation being
            // released when the Task body exits.
            continuation.finish()
            return
          }

          // The restart loop always entered at least once
          // (`messages` always carries the user's prompt), so
          // `runOnePass` ran at least once and either populated
          // `lastFinishInfo` or threw. Reaching the finalize
          // path with `lastFinishInfo` still nil would mean a
          // logic break in `runOnePass` itself.
          guard let lastFinishInfo else {
            preconditionFailure("runTurn reached finalize without any pass info")
          }
          let terminal = envelopeBox.finalize(
            info: usageBox.finalInfo(finishReason: lastFinishInfo.finishReason),
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
          // Write the box *after* the consumer has had every
          // terminal response event yielded to its
          // continuation. If a Task.cancellation lands between
          // the box write and the yield, the consumer would
          // never observe the terminal response event yet
          // `lastResponse` would be populated – the docstring's
          // "silently closed turn does not update this
          // accessor" contract would be violated. Yielding
          // first means the only way to populate
          // `lastResponse` is for the consumer to have either
          // received the terminal event or been gone already
          // (in which case the yields are no-ops and updating
          // the box has no observable effect anyway).
          if let terminalResponse {
            lastResponseBox.set(terminalResponse)
          }
          continuation.finish()
        }
      } catch is CancellationError {
        // Parent-task cancellation can propagate
        // `CancellationError` through any of the inner `await`s
        // (`processor.prepare`, `toolDispatch`, MLX iteration).
        // The session contract is "silent close on cancel" –
        // surface it as a clean stream end so the consumer's
        // `for try await` simply exits, rather than throwing a
        // structured-cancellation error the consumer would
        // otherwise have to filter out manually.
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }

    continuation.onTermination = { _ in
      // Cancel the outer turn task – do NOT release the cache lock
      // from here, and do NOT await the in-flight pass's cleanup
      // here. The lock is released only when the surrounding
      // `cache.update` block returns, which it does only after
      // `runOnePass` has awaited the cancelled pass's cleanup.
      // Releasing here would let a second turn enter `update`
      // while MLX's task is still draining.
      //
      // `task.cancel()` synchronously sets `Task.isCancelled` on
      // the outer task; the restart loop's pass-boundary checks
      // observe it on the next iteration, and any in-flight `await`
      // (token-loop iteration, parser-prepare, tool-dispatch) sees
      // cancellation propagate through Swift's structured
      // concurrency. Mirrors `ChatSession`'s `!Task.isCancelled`
      // guard.
      task.cancel()
    }

    return stream
  }
}

private extension ResponseFunctionCallOutput.Output {
  func toolMessage() -> Chat.Message {
    Chat.Message(role: .tool, content: toolMessageText, images: toolMessageImages)
  }

  var toolMessageText: String {
    switch self {
      case let .string(text):
        text

      case let .content(parts):
        parts.compactMap(\.toolMessageText).joined(separator: "\n")
    }
  }

  var toolMessageImages: [UserInput.Image] {
    guard case let .content(parts) = self else { return [] }
    return parts.compactMap(\.toolMessageImage)
  }
}

private extension ResponseFunctionCallOutput.Content {
  var toolMessageText: String? {
    switch self {
      case let .inputText(text):
        return text.text

      case let .inputImage(image):
        if let imageURL = image.imageURL {
          return "[image: \(imageURL)]"
        }
        if let fileId = image.fileId {
          return "[image: \(fileId)]"
        }
        return "[image]"

      case let .inputFile(file):
        if let filename = file.filename {
          return "[file: \(filename)]"
        }
        if let fileURL = file.fileURL {
          return "[file: \(fileURL)]"
        }
        if let fileId = file.fileId {
          return "[file: \(fileId)]"
        }
        return "[file]"
    }
  }

  var toolMessageImage: UserInput.Image? {
    guard case let .inputImage(image) = self, let imageURL = image.imageURL else {
      return nil
    }
    if let url = URL(string: imageURL), url.scheme != nil {
      return .url(url)
    }
    if imageURL.hasPrefix("/") {
      return .url(URL(fileURLWithPath: imageURL))
    }
    return nil
  }
}
