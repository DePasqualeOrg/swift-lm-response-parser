// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
@testable import LMResponseParserMLX
import Testing

@Suite("ResponseTurnEnvelope — start/finalize")
struct EnvelopeStartFinalizeTests {
  @Test
  func `start emits responseCreated then responseInProgress with monotonic sequence`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    let events = envelope.start()
    #expect(events.count == 2)
    guard case let .responseCreated(created) = events[0] else {
      Issue.record("Expected responseCreated"); return
    }
    guard case let .responseInProgress(inProgress) = events[1] else {
      Issue.record("Expected responseInProgress"); return
    }
    #expect(created.sequenceNumber == 0)
    #expect(inProgress.sequenceNumber == 1)
    #expect(created.response.id == envelope.responseId)
    #expect(inProgress.response.id == envelope.responseId)
  }

  @Test
  func `finalize emits responseCompleted with derived status from finishReason`() {
    for (reason, expected) in [
      (FinishReason.stop, ResponseStatus.completed),
      (.length, .incomplete),
      (.cancelled, .cancelled),
    ] {
      let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
      _ = envelope.start()
      let events = envelope.finalize(info: FinishInfo(
        finishReason: reason, inputTokens: 1, outputTokens: 1,
      ))
      guard case let .responseCompleted(e) = events[0] else {
        Issue.record("Expected responseCompleted for \(reason)"); return
      }
      #expect(e.response.status == expected, "status for \(reason)")
    }
  }

  @Test
  func `incompleteDetails populated for length but not for stop`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()
    let events = envelope.finalize(info: FinishInfo(
      finishReason: .length, inputTokens: 0, outputTokens: 0,
    ))
    guard case let .responseCompleted(e) = events[0] else {
      Issue.record("Expected responseCompleted"); return
    }
    #expect(e.response.incompleteDetails?.reason == .maxOutputTokens)
  }

  @Test
  func `usage carries through inputTokens, outputTokens, total, and reasoning`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()
    let events = envelope.finalize(info: FinishInfo(
      finishReason: .stop,
      inputTokens: 100,
      outputTokens: 50,
      cachedInputTokens: 20,
      reasoningOutputTokens: 30,
    ))
    guard case let .responseCompleted(e) = events[0] else {
      Issue.record("Expected responseCompleted"); return
    }
    #expect(e.response.usage?.inputTokens == 100)
    #expect(e.response.usage?.outputTokens == 50)
    #expect(e.response.usage?.totalTokens == 150)
    #expect(e.response.usage?.inputTokensDetails.cachedTokens == 20)
    #expect(e.response.usage?.outputTokensDetails.reasoningTokens == 30)
  }
}

@Suite("ResponseTurnEnvelope — pass forwarding")
struct EnvelopeForwardTests {
  /// Build the events one parser pass would produce for a single
  /// message item at output_index 0.
  private func messagePassEvents(itemId: String, text: String) -> [ResponseStreamingEvent] {
    let item = ResponseOutputItem.message(.init(
      id: itemId,
      content: [.outputText(.init(text: text))],
      status: .completed,
    ))
    return [
      .outputItemAdded(.init(
        item: .message(.init(id: itemId)),
        outputIndex: 0,
        sequenceNumber: 0,
      )),
      .outputTextDelta(.init(
        itemId: itemId, outputIndex: 0, contentIndex: 0,
        delta: text, sequenceNumber: 1,
      )),
      .outputItemDone(.init(
        item: item,
        outputIndex: 0,
        sequenceNumber: 2,
      )),
    ]
  }

  @Test
  func `Sequence numbers are strictly monotonic across pass boundaries`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    var allEvents = envelope.start()

    let pass1 = envelope.beginPass()
    allEvents += pass1.forward(messagePassEvents(itemId: "msg_a", text: "hi"))
    pass1.end()

    let pass2 = envelope.beginPass()
    allEvents += pass2.forward(messagePassEvents(itemId: "msg_b", text: "bye"))
    pass2.end()

    allEvents += envelope.finalize(info: FinishInfo(
      finishReason: .stop, inputTokens: 1, outputTokens: 1,
    ))

    let sequences = allEvents.map(\.sequenceNumber)
    for (i, n) in sequences.enumerated() where i > 0 {
      #expect(n == sequences[i - 1] + 1, "sequence #\(i) not monotonic")
    }
  }

  @Test
  func `Output indices rebase per pass — second pass items don't collide with first`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()

    let pass1 = envelope.beginPass()
    let pass1Out = pass1.forward(messagePassEvents(itemId: "msg_a", text: "hi"))
    pass1.end()

    let pass2 = envelope.beginPass()
    let pass2Out = pass2.forward(messagePassEvents(itemId: "msg_b", text: "bye"))
    pass2.end()

    let pass1Indices = pass1Out.compactMap { event -> Int? in
      switch event {
        case let .outputItemAdded(e): return e.outputIndex
        case let .outputItemDone(e): return e.outputIndex
        case let .outputTextDelta(e): return e.outputIndex
        default: return nil
      }
    }
    let pass2Indices = pass2Out.compactMap { event -> Int? in
      switch event {
        case let .outputItemAdded(e): return e.outputIndex
        case let .outputItemDone(e): return e.outputIndex
        case let .outputTextDelta(e): return e.outputIndex
        default: return nil
      }
    }
    #expect(pass1Indices.allSatisfy { $0 == 0 })
    #expect(pass2Indices.allSatisfy { $0 == 1 })
  }

  @Test
  func `Tool-result item lands between passes at the next output_index`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()

    // Pass 1: function call at index 0.
    let pass1 = envelope.beginPass()
    let fc = ResponseFunctionToolCall(
      id: "fc_a", callId: "call_a", name: "tool", arguments: "{}", status: .completed,
    )
    let pass1Events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(
        item: .functionCall(.init(id: "fc_a", callId: "call_a", name: "tool", arguments: "")),
        outputIndex: 0, sequenceNumber: 0,
      )),
      .outputItemDone(.init(
        item: .functionCall(fc),
        outputIndex: 0, sequenceNumber: 1,
      )),
    ]
    _ = pass1.forward(pass1Events)
    pass1.end()

    // Tool result.
    let toolEvents = envelope.emitToolResult(.init(
      id: "fco_a", callId: "call_a", output: "ok",
    ))
    let toolIndex = toolEvents.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }.first
    #expect(toolIndex == 1)

    // Pass 2: message at local index 0 should land at turn-scoped 2.
    let pass2 = envelope.beginPass()
    let pass2Out = pass2.forward(messagePassEvents(itemId: "msg_b", text: "done"))
    pass2.end()

    let pass2AddedIndex = pass2Out.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }.first
    #expect(pass2AddedIndex == 2)
  }

  @Test
  func `Lifecycle envelope events from a per-pass parser are dropped by forward`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()

    let pass = envelope.beginPass()
    let strayResponse = Response(id: "wrong", createdAt: 0, model: "m")
    let stray: [ResponseStreamingEvent] = [
      .responseCreated(.init(response: strayResponse, sequenceNumber: 0)),
      .responseInProgress(.init(response: strayResponse, sequenceNumber: 1)),
      .outputItemAdded(.init(
        item: .message(.init(id: "msg_x")),
        outputIndex: 0, sequenceNumber: 2,
      )),
      .responseCompleted(.init(response: strayResponse, sequenceNumber: 3)),
    ]
    let out = pass.forward(stray)
    let kinds = out.map { event -> String in
      switch event {
        case .responseCreated: "responseCreated"
        case .responseInProgress: "responseInProgress"
        case .responseCompleted: "responseCompleted"
        case .outputItemAdded: "outputItemAdded"
        default: "other"
      }
    }
    #expect(kinds == ["outputItemAdded"])
  }

  @Test
  func `After end(), further forwards are no-ops`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()
    let pass = envelope.beginPass()
    pass.end()
    let out = pass.forward(messagePassEvents(itemId: "msg_a", text: "hi"))
    #expect(out.isEmpty)
  }
}

@Suite("ResponseTurnEnvelope — emitToolResult")
struct EmitToolResultTests {
  @Test
  func `Emits added then done with the same item`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()

    let events = envelope.emitToolResult(.init(
      id: "fco_x", callId: "call_x", output: "result",
    ))
    #expect(events.count == 2)
    guard case let .outputItemAdded(added) = events[0] else {
      Issue.record("Expected outputItemAdded"); return
    }
    guard case let .outputItemDone(done) = events[1] else {
      Issue.record("Expected outputItemDone"); return
    }
    #expect(added.outputIndex == done.outputIndex)
    guard case let .functionCallOutput(item) = added.item else {
      Issue.record("Expected functionCallOutput item"); return
    }
    #expect(item.callId == "call_x")
    #expect(item.output == .string("result"))
  }

  @Test
  func `Tool-result indices increment correctly when emitted in sequence`() {
    let envelope = ResponseTurnEnvelope(config: .init(model: "m", createdAt: 0))
    _ = envelope.start()

    let first = envelope.emitToolResult(.init(id: "fco_1", callId: "c1", output: "a"))
    let second = envelope.emitToolResult(.init(id: "fco_2", callId: "c2", output: "b"))

    let firstIndex = first.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }.first
    let secondIndex = second.compactMap { event -> Int? in
      if case let .outputItemAdded(e) = event { return e.outputIndex }
      return nil
    }.first
    #expect(firstIndex == 0)
    #expect(secondIndex == 1)
  }
}
