// Copyright © Anthony DePasquale

import Foundation
import LMResponses

extension ResponseFormat {
  /// Derive parser boundary context from the prepared prompt token IDs.
  ///
  /// The parser-domain decision lives in `LMResponses`; this bridge
  /// method only decodes the prompt tokens into the rendered text that
  /// core can inspect.
  func promptBoundaryPriorText(
    fromPromptTokens tokens: [Int],
    tokenizer: any ResponseTokenizer,
  ) throws -> String? {
    guard requiresRenderedPromptBoundaryPriorText else { return nil }
    guard !tokens.isEmpty else { return nil }
    let renderedPrompt = try tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
    return promptBoundaryPriorText(fromRenderedPrompt: renderedPrompt)
  }

  /// Combine parser-relevant prompt-boundary state with caller-supplied
  /// generated prior text before passing it through
  /// `makeParser(..., priorOutput:)`.
  func combinedPriorOutput(
    fromPromptTokens tokens: [Int],
    tokenizer: any ResponseTokenizer,
    generatedPriorOutput: String?,
  ) throws -> String? {
    guard requiresRenderedPromptBoundaryPriorText else {
      return generatedPriorOutput
    }
    guard !tokens.isEmpty else { return generatedPriorOutput }
    let renderedPrompt = try tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
    return combinedPriorOutput(
      fromRenderedPrompt: renderedPrompt,
      generatedPriorOutput: generatedPriorOutput,
    )
  }
}
