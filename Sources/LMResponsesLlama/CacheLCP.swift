// Copyright © Anthony DePasquale

import Foundation

/// Longest common prefix length between two arrays. Used by both
/// session types to compute how much of the prior turn's cache
/// (KV tokens for text, chunk signatures for multimodal) the new
/// turn can reuse.
func longestCommonPrefixLength<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
  let n = min(a.count, b.count)
  var i = 0
  while i < n, a[i] == b[i] {
    i += 1
  }
  return i
}
