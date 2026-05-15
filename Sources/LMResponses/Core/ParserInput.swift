// Copyright © Anthony DePasquale

import Foundation

/// One unit of input to a streaming parser.
///
/// `text` is the parser's decoded payload. `tokenIds`, when present, are
/// aligned metadata: exactly the generated token IDs whose incremental
/// detokenization produced `text`.
///
/// That alignment lets parsers choose the right level of precision for their
/// format. A parser can stay text-only, use token IDs to recognize reserved
/// marker tokens, or combine both signals to distinguish a structural marker
/// token from ordinary content tokens that decode to the same characters.
///
/// When the streaming detokenizer withholds text because a token sequence
/// has not yet formed a complete Unicode scalar, the driver must withhold
/// the contributing IDs too and flush both fields together. ``ResponseStream``
/// handles that buffering for token-loop callers.
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
