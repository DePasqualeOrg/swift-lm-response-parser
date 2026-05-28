// Copyright © Anthony DePasquale

@testable import LMResponsesLlama
import Testing

@Suite("longestCommonPrefixLength")
struct CacheLCPTests {
  @Test func `empty arrays return zero`() {
    #expect(longestCommonPrefixLength([Int](), [Int]()) == 0)
  }

  @Test func `disjoint arrays return zero`() {
    #expect(longestCommonPrefixLength([1, 2, 3], [4, 5, 6]) == 0)
  }

  @Test func `identical arrays return full length`() {
    #expect(longestCommonPrefixLength([1, 2, 3, 4], [1, 2, 3, 4]) == 4)
  }

  @Test func `first shorter returns its length`() {
    #expect(longestCommonPrefixLength([1, 2], [1, 2, 3, 4]) == 2)
  }

  @Test func `second shorter returns its length`() {
    #expect(longestCommonPrefixLength([1, 2, 3, 4], [1, 2]) == 2)
  }

  @Test func `divergence at middle returns divergence point`() {
    #expect(longestCommonPrefixLength([1, 2, 3, 9, 5], [1, 2, 3, 4, 5]) == 3)
  }

  @Test func `divergence at first returns zero`() {
    #expect(longestCommonPrefixLength([9, 2, 3], [1, 2, 3]) == 0)
  }

  @Test func `empty vs non empty returns zero`() {
    #expect(longestCommonPrefixLength([Int](), [1, 2, 3]) == 0)
    #expect(longestCommonPrefixLength([1, 2, 3], [Int]()) == 0)
  }

  @Test func `works on strings`() {
    #expect(longestCommonPrefixLength(["a", "b", "c"], ["a", "b", "x"]) == 2)
  }
}
