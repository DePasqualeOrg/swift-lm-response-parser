// Copyright © Anthony DePasquale

import Foundation

/// Per-token streaming response. Owns the parser, the streaming detokenizer,
/// the response-scoped envelope, and a live `[ResponseOutputItem]` snapshot
/// so consumers feeding their own model loop don't have to wire any of it
/// together.
///
/// Mirrors `openai-harmony`'s `StreamableParser` shape: feed one token at a
/// time, get back the events for that token, and read accumulated state off
/// properties on the same instance.
///
/// ```swift
/// let stream = ResponseStream(
///     format: .qwen3Xml,
///     config: ResponseStreamConfig(model: modelName, tools: tools),
///     tokenizer: tokenizer,
///     tools: tools
/// )
///
/// for event in stream.start() { yield(event) }
///
/// while let token = await model.nextToken() {
///     for event in stream.process(tokenId: token) { yield(event) }
///     render(stream.items)  // live snapshot for UI
/// }
///
/// for event in stream.finalize(finishReason: .stop, inputTokens: promptTokenCount) {
///     yield(event)
/// }
/// ```
///
/// Single-use: one instance per response. ``start()`` is called exactly
/// once before any ``process(tokenId:)``, and ``finalize(finishReason:inputTokens:cachedInputTokens:reasoningOutputTokens:)``
/// exactly once after the last token.
///
/// **Cancellation.** To cut a stream short, call
/// ``finalize(finishReason:inputTokens:cachedInputTokens:reasoningOutputTokens:)``
/// with ``FinishReason/cancelled``. The terminal events fire and
/// ``finalResponse`` is populated as on any other path. If the consumer
/// stops calling ``process(tokenId:)`` without finalizing, the stream
/// retains its last in-progress snapshot — ``items`` shows whatever
/// items were open with their `.inProgress` status, ``finalResponse``
/// stays nil.
public final class ResponseStream {
  private let emitter: ResponseStreamEmitter
  private var detokenizer: NaiveStreamingDetokenizer
  private var pendingTokenIds: [Int] = []
  private var outputTokenCount: Int = 0
  private var accumulator = ResponseItemsAccumulator()

  /// Response-scoped ID minted by the underlying emitter. Stable across
  /// every event the stream yields and every snapshot it constructs.
  public var responseId: String {
    emitter.responseId
  }

  /// Number of tokens fed into ``process(tokenId:)`` so far. Useful when
  /// the consumer wants to surface a running output count, or when
  /// constructing per-format usage breakdowns. Reflects every token
  /// passed in, including ones whose detokenized form was withheld
  /// pending a UTF-8 boundary.
  public var generatedTokens: Int {
    outputTokenCount
  }

  /// Live `[ResponseOutputItem]` snapshot of the response so far. Updated
  /// on every ``start()``, ``process(tokenId:)``, and
  /// ``finalize(finishReason:inputTokens:cachedInputTokens:reasoningOutputTokens:)``
  /// call. Items whose matching `output_item.done` has not yet arrived
  /// appear with their `inProgress` status preserved; filter on `status`
  /// to skip partial state.
  public var items: [ResponseOutputItem] {
    accumulator.items
  }

  /// Terminal `Response` envelope (usage, status, incomplete details)
  /// captured from the `response.completed` event. Nil until
  /// ``finalize(finishReason:inputTokens:cachedInputTokens:reasoningOutputTokens:)``
  /// runs; populated synchronously inside that call. Mirrors
  /// `ResponseStreamHandle.finalResponse()` and
  /// `ResponseChatSession.lastResponse` on the bridge side.
  public private(set) var finalResponse: Response?

  /// Construct a stream for `format`. The parser is built internally;
  /// pass `priorOutput` for continuation requests so the parser can
  /// resume an unclosed reasoning block from prior output.
  public convenience init(
    format: ResponseFormat,
    config: ResponseStreamConfig,
    tokenizer: any ParserTokenizer,
    tools: [ToolSpec] = [],
    priorOutput: String? = nil,
  ) {
    let parser = format.makeParser(tokenizer: tokenizer, tools: tools, priorOutput: priorOutput)
    self.init(parser: parser, config: config, tokenizer: tokenizer)
  }

  init(
    parser: any ResponseFormatParser,
    config: ResponseStreamConfig,
    tokenizer: any ParserTokenizer,
  ) {
    emitter = ResponseStreamEmitter(parser: parser, config: config)
    detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
  }

  /// Yield the lifecycle-envelope events that must appear at the head of
  /// the stream: `response.created` and `response.in_progress`. Call
  /// before any ``process(tokenId:)``.
  ///
  /// Consumers that only read ``items`` can ignore the return value; the
  /// items snapshot is updated regardless.
  @discardableResult
  public func start() -> [ResponseStreamingEvent] {
    let events = emitter.start()
    accumulator.ingest(events)
    return events
  }

  /// Feed one generated token through the detokenizer and parser. Returns
  /// the events to yield for this token, which is often empty (the
  /// detokenizer withholds chunks ending mid-Unicode-scalar until a
  /// following token completes them).
  ///
  /// Consumers that only read ``items`` can ignore the return value; the
  /// items snapshot is updated regardless.
  @discardableResult
  public func process(tokenId: Int) -> [ResponseStreamingEvent] {
    detokenizer.append(token: tokenId)
    pendingTokenIds.append(tokenId)
    outputTokenCount += 1
    guard let chunk = detokenizer.next() else { return [] }
    let events = emitter.process(text: chunk, tokenIds: pendingTokenIds)
    pendingTokenIds.removeAll(keepingCapacity: true)
    accumulator.ingest(events)
    return events
  }

  /// Flush parser state and yield the terminal `response.completed`
  /// event. Use ``FinishReason/length`` instead of ``FinishReason/stop``
  /// if generation hit `max_output_tokens`, and ``FinishReason/cancelled``
  /// if the consumer cut the stream short – these map to the correct
  /// `Response.status` and `incomplete_details`.
  ///
  /// `inputTokens` is the prompt length the engine reported. The stream
  /// supplies `outputTokens` from its own ``generatedTokens`` count.
  ///
  /// Consumers that only read ``items`` can ignore the return value; the
  /// items snapshot is updated regardless.
  @discardableResult
  public func finalize(
    finishReason: FinishReason,
    inputTokens: Int,
    cachedInputTokens: Int = 0,
    reasoningOutputTokens: Int = 0,
  ) -> [ResponseStreamingEvent] {
    let info = FinishInfo(
      finishReason: finishReason,
      inputTokens: inputTokens,
      outputTokens: outputTokenCount,
      cachedInputTokens: cachedInputTokens,
      reasoningOutputTokens: reasoningOutputTokens,
    )
    let events = emitter.finalize(info: info)
    accumulator.ingest(events)
    for event in events {
      if case let .responseCompleted(e) = event {
        finalResponse = e.response
      }
    }
    return events
  }
}
