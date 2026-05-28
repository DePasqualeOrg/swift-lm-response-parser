// Copyright © Anthony DePasquale

import Foundation
import LMResponses

/// Aggregates per-pass token counts plus reasoning-token attribution across
/// multiple generation passes. The bridge constructs one ``FinishInfo`` for
/// the terminal response event from the accumulator's totals.
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
