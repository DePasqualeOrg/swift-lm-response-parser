// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser

/// A streaming handle returned by the bridge's low-level helpers.
///
/// Wraps an `AsyncStream` of generation output together with a cleanup
/// barrier consumers can await before reusing a passed-in `cache`. The
/// stream's `onTermination` callback is synchronous; it can `task.cancel()`
/// MLX's producer task but cannot `await task.value`. A consumer that
/// breaks the iteration loop and immediately reuses the same `cache`
/// therefore races MLX's drain – the cancelled task may still hold write
/// access for a brief window after the stream closes.
///
/// Conforms to `AsyncSequence`, so consumers iterate the handle directly:
///
/// ```swift
/// for await item in handle { … }
/// ```
///
/// rather than going through a property like `.events` whose name would lie
/// in items mode (where each yield is a `[ResponseOutputItem]` snapshot,
/// not a streaming event).
///
/// ``awaitCleanup()`` exposes the producer task's completion as a barrier
/// for non-terminal exit paths (consumer cancellation). On a clean finish
/// it returns immediately because the bridge already awaited the producer
/// task before yielding the terminal event.
///
/// ``finalResponse()`` exposes the terminal `Response` snapshot the bridge
/// generated internally for the `response.completed` event. Useful chiefly
/// for ``streamResponseItems(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``
/// callers that can't see lifecycle events but still want usage and status.
///
/// The stream is non-throwing: setup errors are surfaced synchronously by
/// the helper that mints this handle.
public struct ResponseStreamHandle<Element: Sendable>: AsyncSequence, Sendable {
  public typealias AsyncIterator = AsyncStream<Element>.Iterator

  private let stream: AsyncStream<Element>
  private let _awaitCleanup: @Sendable () async -> Void
  private let _finalResponse: @Sendable () async -> Response?

  /// Internal because handles are minted by the bridge's helpers.
  /// Consumers receive them; they don't construct them.
  init(
    stream: AsyncStream<Element>,
    awaitCleanup: @Sendable @escaping () async -> Void,
    finalResponse: @Sendable @escaping () async -> Response? = { nil },
  ) {
    self.stream = stream
    _awaitCleanup = awaitCleanup
    _finalResponse = finalResponse
  }

  public func makeAsyncIterator() -> AsyncStream<Element>.Iterator {
    stream.makeAsyncIterator()
  }

  /// Wait for the underlying generation task to drain. Required before
  /// reusing a passed-in `cache` after early consumer cancellation.
  /// Returns immediately if generation already finished.
  public func awaitCleanup() async {
    await _awaitCleanup()
  }

  /// The terminal ``/LMResponseParser/Response`` snapshot – usage, status, incomplete
  /// details – captured from the `response.completed` event the bridge
  /// generated internally. Awaits cleanup before returning, so callers
  /// can rely on the result reflecting the fully drained pass.
  ///
  /// Returns `nil` when the pass finished abnormally (consumer
  /// cancellation before any output, or an unexpected MLX contract
  /// violation that closes the stream silently). Once the bridge has
  /// captured even a partial pass-finish record it generates a
  /// `cancelled`-status terminal which is returned here.
  public func finalResponse() async -> Response? {
    await _finalResponse()
  }
}
