// Copyright © Anthony DePasquale

import Foundation

/// Describes formats whose generated suffix starts inside a reasoning
/// preamble unless already-rendered text proves that preamble has ended.
///
/// Unlike ``DelimitedReasoningBoundary``, these formats do not require the
/// preceding text to contain a start marker. The opener may have been injected
/// by the chat template, or the model may emit raw reasoning text directly.
package struct ImplicitReasoningPreamble: Equatable {
  package var endTokens: [String]

  package init(endTokens: [String]) {
    self.endTokens = endTokens
  }

  package static func think(implicitEndTokens: [String] = []) -> Self {
    Self(endTokens: ["</think>"] + implicitEndTokens)
  }

  package func startsInReasoning(after precedingText: String?) -> Bool {
    !hasEnded(in: precedingText)
  }

  package func hasEnded(in precedingText: String?) -> Bool {
    guard let precedingText, !precedingText.isEmpty else { return false }
    return endTokens.contains { precedingText.range(of: $0) != nil }
  }
}
