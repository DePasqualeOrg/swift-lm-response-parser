// Copyright © Anthony DePasquale

import Foundation

/// Describes a reasoning region delimited by a start marker and one or more
/// markers that end the region.
///
/// This covers model families whose parser state can be inferred by scanning
/// the already-rendered text immediately before a generated suffix. Families
/// with implicit reasoning preambles that do not require a start marker use
/// their own initial-state rules in ``ResponseFormat``.
package struct DelimitedReasoningBoundary: Equatable {
  /// A marker that closes reasoning only when the scanned text does not
  /// contain its matching end marker after the same start marker.
  package struct PairedImplicitEnd: Equatable {
    package var start: String
    package var end: String

    package init(start: String, end: String) {
      self.start = start
      self.end = end
    }

    package static let toolCall = Self(start: "<tool_call>", end: "</tool_call>")
  }

  package var start: String
  package var endTokens: [String]
  package var unpairedImplicitEnds: [PairedImplicitEnd]

  package init(
    start: String,
    end: String,
    implicitEndTokens: [String] = [],
    unpairedImplicitEnds: [PairedImplicitEnd] = [],
  ) {
    self.start = start
    endTokens = [end] + implicitEndTokens
    self.unpairedImplicitEnds = unpairedImplicitEnds
  }

  package static func think(
    implicitEndTokens: [String] = [],
    unpairedImplicitEnds: [PairedImplicitEnd] = [],
  ) -> Self {
    Self(
      start: "<think>",
      end: "</think>",
      implicitEndTokens: implicitEndTokens,
      unpairedImplicitEnds: unpairedImplicitEnds,
    )
  }

  package func isOpen(in precedingText: String?) -> Bool {
    suffixIfOpen(in: precedingText) != nil
  }

  package func suffixIfOpen(in precedingText: String?) -> String? {
    guard let precedingText, !precedingText.isEmpty else { return nil }
    guard let startRange = precedingText.range(of: start, options: .backwards) else {
      return nil
    }

    let afterStart = startRange.upperBound ..< precedingText.endIndex
    for endToken in endTokens {
      if precedingText.range(of: endToken, range: afterStart) != nil {
        return nil
      }
    }
    for implicitEnd in unpairedImplicitEnds {
      if containsUnpairedImplicitEnd(implicitEnd, in: precedingText, range: afterStart) {
        return nil
      }
    }

    return String(precedingText[startRange.lowerBound...])
  }

  private func containsUnpairedImplicitEnd(
    _ implicitEnd: PairedImplicitEnd,
    in text: String,
    range: Range<String.Index>,
  ) -> Bool {
    var searchRange = range
    while let startRange = text.range(of: implicitEnd.start, range: searchRange) {
      let afterImplicitStart = startRange.upperBound ..< range.upperBound
      if text.range(of: implicitEnd.end, range: afterImplicitStart) == nil {
        return true
      }
      searchRange = startRange.upperBound ..< range.upperBound
    }
    return false
  }
}
