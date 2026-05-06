// Copyright © Anthony DePasquale

import Foundation

/// Streams a sequence of token IDs into incrementally emitted text chunks,
/// implementing the upstream Hugging Face Rust `step_decode_stream` algorithm.
///
/// Feed tokens one at a time through ``consume(_:)``. Each call returns either
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
  private let tokenizer: any ParserTokenizer
  private let skipSpecialTokens: Bool
  private var ids: [Int]
  private var prefix: String
  private var prefixIndex: Int

  /// Creates an empty stream over `tokenizer`.
  ///
  /// - Parameters:
  ///   - tokenizer: The tokenizer used to decode token IDs.
  ///   - skipSpecialTokens: When `true`, special tokens are omitted from emitted text.
  public convenience init(tokenizer: any ParserTokenizer, skipSpecialTokens: Bool = false) {
    self.init(tokenizer: tokenizer, skipSpecialTokens: skipSpecialTokens, initialTokenIds: [])
  }

  /// Creates a stream seeded with prior token IDs.
  ///
  /// On the first ``consume(_:)`` call, the detokenizer decodes the initial
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
    tokenizer: any ParserTokenizer,
    skipSpecialTokens: Bool = false,
    initialTokenIds: [Int],
  ) {
    self.tokenizer = tokenizer
    self.skipSpecialTokens = skipSpecialTokens
    ids = initialTokenIds
    prefix = ""
    prefixIndex = 0
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
    return newChunk
  }

  /// Consumes a batch of tokens. Returns the concatenation of every chunk
  /// produced, or `nil` if no chunk was produced.
  ///
  /// Equivalent to repeated calls to ``consume(_:)``. Throws on the first
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

public extension ParserTokenizer {
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
