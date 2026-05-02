// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import MLX
import MLXLMCommon

extension ResponseFormat {
  /// Derive parser boundary context from the prepared MLX prompt.
  ///
  /// The parser-domain decision lives in `LMResponseParser`; this bridge
  /// method only decodes MLX's prepared prompt tokens into the rendered text
  /// that core can inspect.
  func promptBoundaryPriorText(
    fromPreparedPrompt input: LMInput,
    tokenizer: MLXTokenizerAdapter,
  ) -> String? {
    guard requiresRenderedPromptBoundaryPriorText else { return nil }
    let tokenIds = input.text.tokens.asArray(Int.self)
    guard !tokenIds.isEmpty else { return nil }
    let renderedPrompt = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
    return promptBoundaryPriorText(fromRenderedPrompt: renderedPrompt)
  }

  /// Combine parser-relevant prompt-boundary state with caller-supplied
  /// generated prior text before passing it through
  /// `makeParser(..., priorOutput:)`.
  func combinedPriorOutput(
    fromPreparedPrompt input: LMInput,
    tokenizer: MLXTokenizerAdapter,
    generatedPriorOutput: String?,
  ) -> String? {
    guard requiresRenderedPromptBoundaryPriorText else {
      return generatedPriorOutput
    }
    let tokenIds = input.text.tokens.asArray(Int.self)
    guard !tokenIds.isEmpty else { return generatedPriorOutput }
    let renderedPrompt = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
    return combinedPriorOutput(
      fromRenderedPrompt: renderedPrompt,
      generatedPriorOutput: generatedPriorOutput,
    )
  }
}
