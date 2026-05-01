// Copyright © Anthony DePasquale

import Foundation

/// One unit of input to a streaming parser: a chunk of detokenized text and,
/// optionally, the token IDs whose detokenized form is exactly that text.
///
/// **None of the shipped parsers read ``tokenIds``.** Every per-format
/// parser matches markers on ``text``: Hermes-family `<think>` markers,
/// Harmony's seven `<|...|>` reserved tokens, Gemma 4's `<|channel>` /
/// `<channel|>` pair, the CJK-bracketed DeepSeek tokens, and every other
/// shipped format have decoded text that is canonical and unambiguous,
/// so string matching gets the same boundaries as token-ID matching
/// without depending on tokenizer-specific IDs.
///
/// The field is preserved on the protocol surface as forward-looking
/// infrastructure for two scenarios where token-ID matching would be
/// strictly more robust than text matching:
///
/// 1. A model that emits a structural marker as literal text via regular
///    tokens (not the reserved special token) – for example, in response
///    to a prompt asking it to echo the marker string. The decoded text
///    is identical; only the token IDs differ. vLLM's Harmony and Gemma 4
///    parsers (`gptoss_reasoning_parser.py`, `gemma4_reasoning_parser.py`)
///    use token-ID matching for this reason.
/// 2. A consumer-side detokenizer configuration that strips special
///    tokens before they reach the parser (e.g., `skip_special_tokens=true`
///    in Hugging Face's tokenizer protocol).
///
/// When token-ID matching is supplied, the driver is responsible for
/// keeping the two fields aligned: when the streaming detokenizer withholds
/// a chunk because it would split a Unicode scalar, the driver must buffer
/// the contributing token IDs and flush both together when the next non-nil
/// chunk arrives. ``ResponseStream`` handles this for callers running
/// their own model loop.
package struct ParserInput {
  /// Detokenized text from one or more new tokens.
  package var text: String

  /// Token IDs of those tokens, in the order they were generated. nil when
  /// the consumer is not tracking token IDs (text-only parsing).
  package var tokenIds: [Int]?

  package init(text: String, tokenIds: [Int]? = nil) {
    self.text = text
    self.tokenIds = tokenIds
  }
}
