// Copyright © Anthony DePasquale

import Foundation

/// Errors thrown by ``StreamingDetokenizer``.
///
/// The parser layer carries its own error type rather than reusing one
/// from `swift-tokenizers` or `mlx-swift-lm` because ``ParserTokenizer``
/// is deliberately decoupled from either ecosystem — a conforming
/// tokenizer can come from anywhere, so the streaming detokenizer needs
/// a parser-local error vocabulary that callers can switch on without
/// importing the upstream packages.
public enum StreamingDetokenizerError: LocalizedError, Equatable {
  /// The freshly decoded text after appending `tokenId` does not start
  /// with the previously decoded prefix. Indicates a tokenizer whose
  /// `decode` is not byte-prefix-monotonic — typically because the
  /// tokenizer retroactively rewrites earlier text once a following
  /// token arrives (e.g., late-applied cleanup or whitespace
  /// normalization). The streaming algorithm depends on monotonicity
  /// to extract chunks; this error is the canary that the wrapped
  /// tokenizer is incompatible with streaming.
  case invalidStreamingPrefix(tokenId: Int, expectedPrefix: String, actualString: String)

  public var errorDescription: String? {
    switch self {
      case let .invalidStreamingPrefix(tokenId, expectedPrefix, actualString):
        """
        Streaming detokenizer prefix invariant violated by token \(tokenId): \
        expected decoded text to start with "\(expectedPrefix)" but got "\(actualString)".
        """
    }
  }
}
