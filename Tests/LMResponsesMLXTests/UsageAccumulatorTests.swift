// Copyright © Anthony DePasquale

import Foundation
import LMResponses
@testable import LMResponsesMLX
import Testing

@Suite("UsageAccumulator")
struct UsageAccumulatorTests {
  @Test
  func `Pass inputs and outputs sum directly`() {
    var usage = UsageAccumulator()
    usage.addPassInput(200)
    usage.addPassOutput(50)
    usage.addPassInput(30)
    usage.addPassOutput(25)
    usage.addPassInput(25)
    usage.addPassOutput(10)
    let info = usage.finalInfo(finishReason: .stop)
    #expect(info.inputTokens == 255)
    #expect(info.outputTokens == 85)
  }

  @Test
  func `Final info reports zero reasoning tokens when no reasoning items occur`() {
    var usage = UsageAccumulator()
    let messageItem = ResponseOutputItem.message(.init(id: "msg_x"))
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(item: messageItem, outputIndex: 0, sequenceNumber: 0)),
      .outputTextDelta(.init(
        itemId: "msg_x", outputIndex: 0, contentIndex: 0,
        delta: "hi", sequenceNumber: 1,
      )),
    ]
    usage.observe(events: events, tokenCount: 5)
    usage.addPassInput(10)
    usage.addPassOutput(5)
    let info = usage.finalInfo(finishReason: .stop)
    #expect(info.reasoningOutputTokens == 0)
  }

  @Test
  func `Chunk that opens a reasoning item is counted as reasoning`() {
    var usage = UsageAccumulator()
    let reasoningItem = ResponseOutputItem.reasoning(.init(id: "rs_a"))
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(item: reasoningItem, outputIndex: 0, sequenceNumber: 0)),
      .reasoningDelta(.init(
        itemId: "rs_a", outputIndex: 0, contentIndex: 0,
        delta: "thinking", sequenceNumber: 1,
      )),
    ]
    usage.observe(events: events, tokenCount: 7)
    let info = usage.finalInfo(finishReason: .stop)
    #expect(info.reasoningOutputTokens == 7)
  }

  @Test
  func `Chunks while reasoning item is open continue counting as reasoning`() {
    var usage = UsageAccumulator()
    let reasoningItem = ResponseOutputItem.reasoning(.init(id: "rs_a"))

    // Chunk 1: opens reasoning + some delta. 5 tokens.
    usage.observe(
      events: [
        .outputItemAdded(.init(item: reasoningItem, outputIndex: 0, sequenceNumber: 0)),
        .reasoningDelta(.init(
          itemId: "rs_a", outputIndex: 0, contentIndex: 0,
          delta: "first", sequenceNumber: 1,
        )),
      ],
      tokenCount: 5,
    )

    // Chunk 2: continues reasoning, no add events but still inside the
    // reasoning item. 3 tokens.
    usage.observe(
      events: [
        .reasoningDelta(.init(
          itemId: "rs_a", outputIndex: 0, contentIndex: 0,
          delta: "second", sequenceNumber: 2,
        )),
      ],
      tokenCount: 3,
    )

    // Chunk 3: closes reasoning + opens message. 2 tokens during the
    // closing chunk (which is still reasoning per our rule), then
    // chunk 4 will be normal text.
    usage.observe(
      events: [
        .reasoningDone(.init(
          itemId: "rs_a", outputIndex: 0, contentIndex: 0,
          text: "first second", sequenceNumber: 3,
        )),
        .outputItemDone(.init(
          item: reasoningItem, outputIndex: 0, sequenceNumber: 4,
        )),
      ],
      tokenCount: 2,
    )

    // Chunk 4: pure message content. 4 tokens, none counted as reasoning.
    let messageItem = ResponseOutputItem.message(.init(id: "msg_y"))
    usage.observe(
      events: [
        .outputItemAdded(.init(item: messageItem, outputIndex: 1, sequenceNumber: 5)),
        .outputTextDelta(.init(
          itemId: "msg_y", outputIndex: 1, contentIndex: 0,
          delta: "answer", sequenceNumber: 6,
        )),
      ],
      tokenCount: 4,
    )

    let info = usage.finalInfo(finishReason: .stop)
    // 5 + 3 + 2 reasoning tokens; 4 message tokens not counted.
    #expect(info.reasoningOutputTokens == 10)
  }

  @Test
  func `FinishReason flows through to the final info`() {
    var usage = UsageAccumulator()
    usage.addPassInput(1)
    usage.addPassOutput(1)
    #expect(usage.finalInfo(finishReason: .length).finishReason == .length)
    #expect(usage.finalInfo(finishReason: .stop).finishReason == .stop)
    #expect(usage.finalInfo(finishReason: .cancelled).finishReason == .cancelled)
  }

  @Test
  func `Cached input tokens stay 0 until mlx-swift-lm reports cache hits`() {
    var usage = UsageAccumulator()
    usage.addPassInput(100)
    usage.addPassOutput(50)
    let info = usage.finalInfo(finishReason: .stop)
    #expect(info.cachedInputTokens == 0)
  }
}
