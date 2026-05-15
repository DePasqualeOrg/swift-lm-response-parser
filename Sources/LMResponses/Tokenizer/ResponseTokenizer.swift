// Copyright © Anthony DePasquale

import Foundation

/// The minimal tokenizer surface the parser layer needs.
///
/// The three signatures match the public `Tokenizers.Tokenizer` protocol
/// from `swift-tokenizers` exactly, so any concrete tokenizer type backed
/// by that package already satisfies the requirements and conforms with
/// a single empty extension on the concrete type:
///
/// ```swift
/// extension MyTokenizer: ResponseTokenizer {}
/// ```
///
/// Swift does not permit declaring protocol-to-protocol conformance in an
/// extension, so the extension must target a concrete type rather than
/// `Tokenizers.Tokenizer` or `MLXLMCommon.Tokenizer` directly. Engines
/// that ship their own tokenizer type can write a similarly trivial
/// conformance against whatever surface they expose.
///
/// The surface covers both sides of the parser boundary:
/// ``decode(tokenIds:skipSpecialTokens:)`` supports token-loop detokenization,
/// while ``convertTokenToId(_:)`` and ``encode(text:addSpecialTokens:)`` let
/// parsers resolve structural markers against the tokenizer vocabulary.
public protocol ResponseTokenizer: Sendable {
  /// Look up the integer ID of a single token string. Returns nil when
  /// the token is not in the vocabulary.
  ///
  /// Parsers can use this to recognize single-token structural markers.
  func convertTokenToId(_ token: String) -> Int?

  /// Encode a piece of text into a token-ID sequence. When
  /// `addSpecialTokens` is false, the tokenizer must not prepend any BOS
  /// or system tokens; callers depend on the returned IDs being exactly
  /// the encoding of the input string.
  ///
  /// Parsers can use this to compare against multi-token marker sequences.
  ///
  /// Throws when the underlying tokenizer cannot encode the input
  /// (malformed text, missing chat-template, FFI failure, etc.). The
  /// concrete error type is up to the conforming type; the parser
  /// surface keeps it untyped because conforming tokenizers come from
  /// different ecosystems (swift-tokenizers' typed `TokenizerError`,
  /// MLX's untyped errors) and we want to avoid a third typed error
  /// hierarchy here.
  func encode(text: String, addSpecialTokens: Bool) throws -> [Int]

  /// Decode a token-ID sequence back to text. Consumed by the
  /// package-level streaming detokenizer.
  ///
  /// Throws when the underlying tokenizer rejects the IDs (negative or
  /// out-of-range values, FFI failure, etc.). See ``encode(text:addSpecialTokens:)``
  /// for why this surface stays untyped.
  func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String
}
