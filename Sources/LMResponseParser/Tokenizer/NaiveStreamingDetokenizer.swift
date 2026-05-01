// Copyright © Anthony DePasquale

import Foundation

/// Streaming detokenizer that withholds chunks ending mid-Unicode-scalar.
///
/// The withholding strategy uses the U+FFFD (replacement character) check
/// that BPE tokenizers produce for incomplete scalar bytes: a multi-byte
/// UTF-8 scalar split across two tokens decodes as `…\u{fffd}` after the
/// first token, and as the real scalar after the second. The detokenizer
/// returns nil for the partial case, leaving the caller to wait for the
/// next token.
///
/// This guarantees that every non-nil chunk passed downstream is a complete
/// UTF-8 string at scalar boundaries, so per-format parsers can match
/// marker bytes safely without worrying about boundary splits.
package struct NaiveStreamingDetokenizer {
  let tokenizer: any ParserTokenizer

  var segmentTokens: [Int] = []
  var segment: String = ""

  package init(tokenizer: any ParserTokenizer) {
    self.tokenizer = tokenizer
  }

  /// Append a newly generated token to the buffer. Call ``next()`` after
  /// each append to read out any complete chunk it produced.
  package mutating func append(token: Int) {
    segmentTokens.append(token)
  }

  /// Return any complete chunk produced by the most-recent appended token,
  /// or nil if the buffer ends mid-scalar (in which case the caller waits
  /// for the next ``append(token:)``).
  package mutating func next() -> String? {
    let newSegment = tokenizer.decode(tokenIds: segmentTokens, skipSpecialTokens: false)
    // A few HF post-processors (e.g., `Lstrip`/`Rstrip` re-applied once a
    // following token is present) can cause the freshly decoded segment
    // to be shorter than the previously cached one. `String.suffix(_:)`
    // traps on negative counts, so clamp and treat the non-growing case
    // as "no new chars yet, hold."
    let diff = newSegment.count - segment.count
    if diff <= 0 { return nil }
    let new = newSegment.suffix(diff)

    // If the freshly decoded suffix ends in U+FFFD, the most recent
    // token's bytes did not complete a Unicode scalar. Withhold the
    // chunk and let a future token finish it.
    if new.last == "\u{fffd}" {
      return nil
    }

    if new.hasSuffix("\n") {
      startNewSegment()
    } else {
      segment = newSegment
    }

    return String(new)
  }

  /// Reset the segment after a newline boundary. Keeps the most-recent
  /// token in the buffer to preserve the decoder's left context, which
  /// some BPE tokenizers depend on when the next token straddles the
  /// reset point.
  private mutating func startNewSegment() {
    let lastToken = segmentTokens.last
    segmentTokens.removeAll()
    if let lastToken {
      segmentTokens.append(lastToken)
      segment = tokenizer.decode(tokenIds: segmentTokens, skipSpecialTokens: false)
    } else {
      segment = ""
    }
  }
}
