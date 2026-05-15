// Copyright © Anthony DePasquale

import Foundation

/// Splits a text buffer into the part that is safe to emit now and the part
/// that must be held back because it could still complete into a marker.
///
/// Detokenized text arrives in units that don't align with marker
/// boundaries, so a chunk ending in `<thi` must be held until the next
/// chunk arrives – `<thi` followed by `nk>` forms a real `<think>` marker,
/// but `<thi` alone is plain content. The function holds back the longest
/// suffix of `text` that is a prefix of any marker, and emits the rest.
package enum PrefixHold {
  /// - Parameters:
  ///   - text: The buffer to split.
  ///   - markers: Marker strings any of whose prefixes the buffer might
  ///     be in the middle of. Empty markers are ignored.
  /// - Returns: `(emitNow, hold)` where the concatenation of the two equals
  ///   the input. `emitNow` is safe to forward to consumers; `hold` must
  ///   be carried forward and prepended to the next chunk.
  package static func split(text: String, markers: [String]) -> (emit: String, hold: String) {
    if text.isEmpty {
      return ("", "")
    }

    var maxHold = 0
    let textChars = Array(text)

    for marker in markers {
      if marker.isEmpty { continue }
      let markerChars = Array(marker)
      // Cap suffix length at marker.count - 1: a complete marker
      // match is the per-format parser's responsibility, this helper
      // only holds back partial-marker prefixes.
      let limit = min(markerChars.count - 1, textChars.count)
      if limit <= maxHold { continue }

      var k = limit
      while k > maxHold {
        // Does `marker` start with the last `k` characters of `text`?
        if textChars[(textChars.count - k)...].elementsEqual(markerChars[..<k]) {
          maxHold = k
          break
        }
        k -= 1
      }
    }

    if maxHold == 0 {
      return (text, "")
    }
    let splitIndex = text.index(text.endIndex, offsetBy: -maxHold)
    return (String(text[..<splitIndex]), String(text[splitIndex...]))
  }
}
