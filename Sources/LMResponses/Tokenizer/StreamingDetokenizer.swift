// Copyright © Anthony DePasquale

import Foundation

/// Streams a sequence of token IDs into incrementally emitted text chunks,
/// implementing the upstream Hugging Face Rust `step_decode_stream` algorithm.
///
/// Feed tokens one at a time through ``consume(_:)-(Int)``. Each call returns either
/// a non-empty text chunk that can be displayed, or `nil` to indicate that the
/// detokenizer has buffered the token because it does not yet form a complete
/// scalar. Internal state is bounded: after every successful emission the
/// stored prefix and id buffer are trimmed so memory does not grow with stream
/// length.
///
/// The detokenizer assumes that the underlying decoder is byte-prefix
/// monotonic — the decode of `ids + [newId]` must start with the decode of
/// `ids`. Wrapping a tokenizer that retroactively rewrites earlier text once
/// a following token arrives (e.g. a swift-tokenizers `PreTrainedTokenizer`
/// with `cleanUpTokenizationSpaces` enabled, routed through the public
/// `decode`) breaks the invariant and throws
/// ``StreamingDetokenizerError/invalidStreamingPrefix(tokenId:expectedPrefix:actualString:)``;
/// callers handle that by resetting the stream and retrying. Bridges that
/// already expose a "raw" decode path should prefer it for the wrapped
/// tokenizer.
///
/// `StreamingDetokenizer` is single-consumer by design and is therefore not
/// `Sendable`.
public final class StreamingDetokenizer {
  private let tokenizer: any ResponseTokenizer
  private let skipSpecialTokens: Bool
  private var ids: [Int]
  private var prefix: String
  private var prefixIndex: Int
  /// Set on a successful ``consume(_:)``. ``flush()`` keys off this to
  /// suppress seed-only emission — `prefix` alone can't distinguish
  /// "no consume yet" from "consumed but seed decoded to U+FFFD".
  private var hasConsumed: Bool

  /// Creates an empty stream over `tokenizer`.
  ///
  /// - Parameters:
  ///   - tokenizer: The tokenizer used to decode token IDs.
  ///   - skipSpecialTokens: When `true`, special tokens are omitted from emitted text.
  public convenience init(tokenizer: any ResponseTokenizer, skipSpecialTokens: Bool = false) {
    self.init(tokenizer: tokenizer, skipSpecialTokens: skipSpecialTokens, initialTokenIds: [])
  }

  /// Creates a stream seeded with prior token IDs.
  ///
  /// On the first ``consume(_:)-(Int)`` call, the detokenizer decodes the initial
  /// IDs to establish its prefix, and only the chunk produced by the new
  /// token is emitted. Use this when resuming a stream after an interruption,
  /// when the initial tokens have already been displayed and should not be
  /// re-emitted.
  ///
  /// - Parameters:
  ///   - tokenizer: The tokenizer used to decode token IDs.
  ///   - skipSpecialTokens: When `true`, special tokens are omitted from emitted text.
  ///   - initialTokenIds: Token IDs already shown to the user.
  public init(
    tokenizer: any ResponseTokenizer,
    skipSpecialTokens: Bool = false,
    initialTokenIds: [Int],
  ) {
    self.tokenizer = tokenizer
    self.skipSpecialTokens = skipSpecialTokens
    ids = initialTokenIds
    prefix = ""
    prefixIndex = 0
    hasConsumed = false
  }

  /// Consumes a token and returns any complete chunk it produced.
  ///
  /// - Parameter id: The next token ID to feed into the stream.
  /// - Returns: A non-empty chunk if the buffer can now emit one, or `nil`
  ///   if the buffer ends mid-scalar and the caller should feed the next
  ///   token. The returned chunk is guaranteed to be non-empty.
  /// - Throws: ``StreamingDetokenizerError/invalidStreamingPrefix(tokenId:expectedPrefix:actualString:)``
  ///   when the decoder produces output that does not begin with the cached
  ///   prefix, or any error propagated from the tokenizer's `decode` call. On
  ///   any throw, internal state is unchanged from before the call.
  public func consume(_ id: Int) throws -> String? {
    var workingIds = ids
    var workingPrefix = prefix
    var workingPrefixIndex = prefixIndex

    if workingPrefix.isEmpty && !workingIds.isEmpty {
      let seeded = try tokenizer.decode(
        tokenIds: workingIds,
        skipSpecialTokens: skipSpecialTokens,
      )
      if !seeded.hasSuffix("\u{fffd}") {
        workingPrefix = seeded
        workingPrefixIndex = workingIds.count
      }
    }

    workingIds.append(id)
    let string = try tokenizer.decode(
      tokenIds: workingIds,
      skipSpecialTokens: skipSpecialTokens,
    )

    if string.utf8.count <= workingPrefix.utf8.count || string.hasSuffix("\u{fffd}") {
      ids = workingIds
      prefix = workingPrefix
      prefixIndex = workingPrefixIndex
      hasConsumed = true
      return nil
    }

    guard string.utf8.starts(with: workingPrefix.utf8) else {
      throw StreamingDetokenizerError.invalidStreamingPrefix(
        tokenId: id,
        expectedPrefix: workingPrefix,
        actualString: string,
      )
    }

    let newChunk = String(
      decoding: string.utf8.dropFirst(workingPrefix.utf8.count),
      as: UTF8.self,
    )

    let trimmed = Array(workingIds[workingPrefixIndex...])
    let refreshed = try tokenizer.decode(
      tokenIds: trimmed,
      skipSpecialTokens: skipSpecialTokens,
    )

    ids = trimmed
    prefix = refreshed
    prefixIndex = trimmed.count
    hasConsumed = true
    return newChunk
  }

  /// Emit any text the buffer is still holding back because the last
  /// consumed token's decode ended mid-multibyte-UTF8 scalar.
  ///
  /// Call once at end-of-stream when no more tokens will arrive. The
  /// returned string includes the Unicode replacement character
  /// (U+FFFD) wherever a partial scalar terminates — better than
  /// silently dropping the bytes. Returns `nil` if there is nothing
  /// buffered.
  ///
  /// On any throw, internal state is unchanged. On success, the
  /// detokenizer is reset: the buffer is empty and subsequent
  /// ``consume(_:)-(Int)`` calls behave as if starting fresh.
  public func flush() throws -> String? {
    guard !ids.isEmpty else { return nil }

    // Seed-only buffer represents bytes the caller already displayed;
    // emitting it would leak the prompt tail as generated output.
    guard hasConsumed else {
      ids = []
      prefix = ""
      prefixIndex = 0
      hasConsumed = false
      return nil
    }

    let string = try tokenizer.decode(
      tokenIds: ids,
      skipSpecialTokens: skipSpecialTokens,
    )

    guard string.utf8.count > prefix.utf8.count else {
      // Nothing past the already-emitted prefix — reset and return nil.
      ids = []
      prefix = ""
      prefixIndex = 0
      hasConsumed = false
      return nil
    }
    guard string.utf8.starts(with: prefix.utf8) else {
      throw StreamingDetokenizerError.invalidStreamingPrefix(
        tokenId: ids.last ?? -1,
        expectedPrefix: prefix,
        actualString: string,
      )
    }
    let newChunk = String(
      decoding: string.utf8.dropFirst(prefix.utf8.count),
      as: UTF8.self,
    )
    ids = []
    prefix = ""
    prefixIndex = 0
    hasConsumed = false
    return newChunk
  }

  /// Consumes a batch of tokens. Returns the concatenation of every chunk
  /// produced, or `nil` if no chunk was produced.
  ///
  /// Equivalent to repeated calls to ``consume(_:)-(Int)``. Throws on the first
  /// failure; tokens already accumulated into the working chunk before the
  /// throw are discarded, and the detokenizer's internal state reflects only
  /// the successful steps that ran before the failing one.
  public func consume(_ ids: [Int]) throws -> String? {
    var combined = ""
    for id in ids {
      if let chunk = try consume(id) {
        combined.append(chunk)
      }
    }
    return combined.isEmpty ? nil : combined
  }
}

public extension ResponseTokenizer {
  /// Returns a fresh ``StreamingDetokenizer`` over this tokenizer.
  func streamingDetokenizer(skipSpecialTokens: Bool = false) -> StreamingDetokenizer {
    StreamingDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
  }

  /// Returns a ``StreamingDetokenizer`` seeded with `initialTokenIds`.
  func streamingDetokenizer(
    skipSpecialTokens: Bool = false,
    initialTokenIds: [Int],
  ) -> StreamingDetokenizer {
    StreamingDetokenizer(
      tokenizer: self,
      skipSpecialTokens: skipSpecialTokens,
      initialTokenIds: initialTokenIds,
    )
  }
}
