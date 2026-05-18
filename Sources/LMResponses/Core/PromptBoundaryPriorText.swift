// Copyright © Anthony DePasquale

import Foundation

package extension ResponseFormat {
  /// Whether this format may need rendered-prompt context to initialize a
  /// parser that starts at a generated suffix boundary.
  var requiresRenderedPromptBoundaryPriorText: Bool {
    promptBoundaryReasoningBoundary != nil
  }

  /// Derive parser boundary context from an already-rendered prompt.
  ///
  /// Parsers operate on generated suffix text, but some chat templates make
  /// that suffix begin inside a structural region opened by the prompt. This
  /// returns minimal prior text that should be passed through
  /// ``makeParser(tokenizer:tools:priorOutput:)`` so parser construction sees
  /// the same parser state it sees on continuation streams.
  func promptBoundaryPriorText(fromRenderedPrompt renderedPrompt: String) -> String? {
    guard let boundary = promptBoundaryReasoningBoundary,
          boundary.isOpen(in: renderedPrompt)
    else { return nil }
    return boundary.start
  }

  /// Combine prompt-boundary context with generated prior text before parser
  /// construction.
  func combinedPriorOutput(
    fromRenderedPrompt renderedPrompt: String?,
    generatedPriorOutput: String?,
  ) -> String? {
    guard let renderedPrompt,
          let promptPriorOutput = promptBoundaryPriorText(fromRenderedPrompt: renderedPrompt)
    else {
      return generatedPriorOutput
    }
    guard let generatedPriorOutput else {
      return promptPriorOutput
    }
    return promptPriorOutput + generatedPriorOutput
  }

  private var promptBoundaryReasoningBoundary: DelimitedReasoningBoundary? {
    switch self {
      case .qwen, .qwen3Xml:
        // Mirrors vLLM's Qwen3ReasoningParser.is_reasoning_end prompt scan:
        // a paired `<tool_call>...</tool_call>` can be a template example, so
        // only an unpaired prompt-side `<tool_call>` implicitly ends reasoning.
        DelimitedReasoningBoundary.think(unpairedImplicitEnds: [.toolCall])

      case .cohereCmd4:
        // cmd4 chat templates can pre-inject `<|START_THINKING|>` via
        // the `response_prefix` variable. The MLX bridge scans the
        // rendered prompt for an unclosed pair so the parser starts
        // inside the reasoning region.
        DelimitedReasoningBoundary(start: "<|START_THINKING|>", end: "<|END_THINKING|>")

      default:
        nil
    }
  }
}
