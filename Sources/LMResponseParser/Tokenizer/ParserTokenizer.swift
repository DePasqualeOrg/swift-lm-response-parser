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
/// extension MyTokenizer: ParserTokenizer {}
/// ```
///
/// Swift does not permit declaring protocol-to-protocol conformance in an
/// extension, so the extension must target a concrete type rather than
/// `Tokenizers.Tokenizer` or `MLXLMCommon.Tokenizer` directly. Engines
/// that ship their own tokenizer type can write a similarly trivial
/// conformance against whatever surface they expose.
///
/// **What's actually used.** Only ``decode(tokenIds:skipSpecialTokens:)``
/// is read by the shipped code path: the package-level streaming
/// detokenizer calls it per token to do the U+FFFD-based UTF-8 boundary
/// withholding. None of the shipped parsers call ``convertTokenToId(_:)``
/// or ``encode(text:addSpecialTokens:)``; every per-format parser matches
/// markers on detokenized text. The two other methods remain on the
/// protocol because (a) the empty-extension conformance to
/// `swift-tokenizers.Tokenizer` requires them, and (b) they're the seam
/// for a future per-parser switch to token-ID matching, where the
/// package-level `ParserInput.tokenIds` field comes into play.
public protocol ParserTokenizer: Sendable {
  /// Look up the integer ID of a single token string. Returns nil when
  /// the token is not in the vocabulary.
  ///
  /// Not currently consumed by any shipped parser; preserved for future
  /// parsers that key off single reserved-token IDs (e.g., a Harmony
  /// variant that needs to disambiguate the structural tokens from
  /// regular text).
  func convertTokenToId(_ token: String) -> Int?

  /// Encode a piece of text into a token-ID sequence. When
  /// `addSpecialTokens` is false, the tokenizer must not prepend any BOS
  /// or system tokens; callers depend on the returned IDs being exactly
  /// the encoding of the input string.
  ///
  /// Not currently consumed by any shipped parser; preserved for future
  /// parsers that need to compare against multi-token marker sequences
  /// (Gemma 4's `<|channel>thought` is the design-doc example).
  func encode(text: String, addSpecialTokens: Bool) -> [Int]

  /// Decode a token-ID sequence back to text. Consumed by the
  /// package-level streaming detokenizer for U+FFFD-based UTF-8
  /// boundary withholding.
  func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
}
