// Copyright © Anthony DePasquale

import Foundation

/// Why generation stopped, plus token-usage breakdown.
///
/// Internal data shape passed from a streaming driver to
/// ``ResponseStreamEmitter/finalize(info:)``. External consumers don't
/// construct this directly – they call ``ResponseStream/finalize(finishReason:inputTokens:cachedInputTokens:reasoningOutputTokens:)``,
/// which builds it for them.
package struct FinishInfo: Equatable {
  package var finishReason: FinishReason

  package var inputTokens: Int
  package var outputTokens: Int

  /// Tokens that were retrieved from the prompt cache. 0 when the
  /// consumer's engine has no prompt cache or did not report cache info.
  package var cachedInputTokens: Int

  /// Output tokens that were classified as reasoning content. 0 unless
  /// the consumer wants to break out reasoning vs message tokens.
  package var reasoningOutputTokens: Int

  package init(
    finishReason: FinishReason,
    inputTokens: Int,
    outputTokens: Int,
    cachedInputTokens: Int = 0,
    reasoningOutputTokens: Int = 0,
  ) {
    self.finishReason = finishReason
    self.inputTokens = inputTokens
    self.outputTokens = outputTokens
    self.cachedInputTokens = cachedInputTokens
    self.reasoningOutputTokens = reasoningOutputTokens
  }
}

/// Why generation stopped.
///
/// The emitter maps these to ``ResponseStatus`` on the terminal
/// `response.completed` event:
///
/// - ``stop`` → ``ResponseStatus/completed``
/// - ``length`` → ``ResponseStatus/incomplete`` plus
///   ``IncompleteDetails`` with ``IncompleteReason/maxOutputTokens``
/// - ``cancelled`` → ``ResponseStatus/cancelled``
public enum FinishReason: String, Sendable, Equatable, CaseIterable {
  /// Model emitted a stop token (EOS or equivalent).
  case stop
  /// Generation hit `max_tokens` / `max_output_tokens`.
  case length
  /// Consumer cancelled the generation.
  case cancelled
}
