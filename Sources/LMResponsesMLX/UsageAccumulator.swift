// Copyright © Anthony DePasquale

import Foundation
import LMResponses

/// Aggregates per-pass MLX token counts plus reasoning-token attribution
/// across multiple generation passes. The bridge constructs one
/// ``FinishInfo`` for the terminal response event from the accumulator's
/// totals.
///
/// **Pass-input summation is correct because** the bridge follows
/// `ChatSession.streamMap`'s pattern: after each `prepare`, `messages` is
/// cleared and only the tool result is appended before the next prepare.
/// Each pass's `promptTokenCount` is therefore the incremental contribution
/// the model actually processed; direct summation gives the right total.
///
/// **Output tokens use MLX's per-pass `generationTokenCount`** – not the
/// sum of chunk token counts – so stop tokens that the bridge drops
/// before reaching the parser are still counted as generated output.
///
/// **Reasoning attribution is coarse**: chunks are the granularity, so a
/// chunk containing both reasoning and non-reasoning content counts
/// entirely as reasoning. The "open at start *or* events contain reasoning
/// content" rule covers the chunk that opens a reasoning item (whose first
/// delta arrives in the same batch as `output_item.added`); a "check before
/// applying events" rule would miss it.
struct UsageAccumulator {
  private(set) var inputTokens: Int = 0
  private(set) var outputTokens: Int = 0
  private(set) var reasoningOutputTokens: Int = 0
  private var inReasoning: Bool = false

  init() {}

  mutating func addPassInput(_ promptTokenCount: Int) {
    inputTokens += promptTokenCount
  }

  mutating func addPassOutput(_ generationTokenCount: Int) {
    outputTokens += generationTokenCount
  }

  /// Update reasoning-token attribution from this chunk's events. Call
  /// after the parser has produced events for the chunk and before
  /// forwarding them.
  mutating func observe(events: [ResponseStreamingEvent], tokenCount: Int) {
    let wasInReasoning = inReasoning
    let chunkOpensOrContainsReasoning = events.contains { event in
      switch event {
        case let .outputItemAdded(e):
          if case .reasoning = e.item { return true }
          return false
        case .reasoningDelta:
          return true
        default:
          return false
      }
    }
    if wasInReasoning || chunkOpensOrContainsReasoning {
      reasoningOutputTokens += tokenCount
    }
    for event in events {
      switch event {
        case let .outputItemAdded(e):
          if case .reasoning = e.item { inReasoning = true }
        case let .outputItemDone(e):
          if case .reasoning = e.item { inReasoning = false }
        default:
          break
      }
    }
  }

  func finalInfo(finishReason: FinishReason) -> FinishInfo {
    FinishInfo(
      finishReason: finishReason,
      inputTokens: inputTokens,
      outputTokens: outputTokens,
      cachedInputTokens: 0,
      reasoningOutputTokens: reasoningOutputTokens,
    )
  }
}
