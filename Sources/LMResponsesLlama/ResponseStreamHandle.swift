// Copyright © Anthony DePasquale

import Foundation
import LMResponses

/// A streaming handle returned by the bridge's low-level helpers.
///
/// Wraps an `AsyncStream` of generation output together with a cleanup
/// barrier consumers can await before reusing the underlying context.
///
/// The stream is non-throwing: setup errors are surfaced synchronously by
/// the helper that mints this handle.
public struct ResponseStreamHandle<Element: Sendable>: AsyncSequence, Sendable {
  public typealias AsyncIterator = AsyncStream<Element>.Iterator

  private let stream: AsyncStream<Element>
  private let _awaitCleanup: @Sendable () async -> Void
  private let _finalResponse: @Sendable () async -> Response?

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
  /// reusing the underlying context after early consumer cancellation.
  /// Returns immediately if generation already finished.
  public func awaitCleanup() async {
    await _awaitCleanup()
  }

  /// The terminal ``/LMResponses/Response`` snapshot – usage, status,
  /// incomplete details – captured from the terminal response event the
  /// bridge generated internally.
  public func finalResponse() async -> Response? {
    await _finalResponse()
  }
}
