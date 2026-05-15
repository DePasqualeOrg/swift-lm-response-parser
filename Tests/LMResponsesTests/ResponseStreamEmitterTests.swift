// Copyright © Anthony DePasquale

@testable import LMResponses
import Testing

@Suite("ResponseStreamEmitter — lifecycle")
struct ResponseStreamEmitterLifecycleTests {
  @Test
  func `start() emits response.created at sequence 0 and response.in_progress at sequence 1`() {
    let emitter = makeEmitter(parser: NoOpParser())
    let events = emitter.start()

    #expect(events.count == 2)

    guard case let .responseCreated(created) = events[0] else {
      Issue.record("Expected responseCreated as first event")
      return
    }
    #expect(created.sequenceNumber == 0)
    #expect(created.response.id.hasPrefix("resp_"))
    #expect(created.response.status == .inProgress)
    #expect(created.response.model == "test-model")

    guard case let .responseInProgress(inProgress) = events[1] else {
      Issue.record("Expected responseInProgress as second event")
      return
    }
    #expect(inProgress.sequenceNumber == 1)
    #expect(inProgress.response.id == created.response.id, "responseId is shared across envelope events")
  }

  @Test
  func `finalize() emits response.completed as terminal event with sequence after envelope`() {
    let emitter = makeEmitter(parser: NoOpParser())
    _ = emitter.start()
    let final = emitter.finalize(info: FinishInfo(finishReason: .stop, inputTokens: 5, outputTokens: 7))

    #expect(final.count == 1)
    guard case let .responseCompleted(completed) = final[0] else {
      Issue.record("Expected responseCompleted")
      return
    }
    #expect(completed.sequenceNumber == 2, "Sequence number continues monotonically across the stream")
    #expect(completed.response.status == .completed)
    #expect(completed.response.usage?.inputTokens == 5)
    #expect(completed.response.usage?.outputTokens == 7)
    #expect(completed.response.usage?.totalTokens == 12)
    #expect(completed.response.incompleteDetails == nil)
  }

  @Test
  func `Length stop emits response incomplete with max_output_tokens`() {
    let emitter = makeEmitter(parser: NoOpParser())
    _ = emitter.start()
    let final = emitter.finalize(info: FinishInfo(finishReason: .length, inputTokens: 5, outputTokens: 100))

    guard case let .responseIncomplete(incomplete) = final[0] else {
      Issue.record("Expected responseIncomplete"); return
    }
    #expect(incomplete.response.status == .incomplete)
    #expect(incomplete.response.incompleteDetails?.reason == .maxOutputTokens)
  }

  @Test
  func `Cancellation maps to status=cancelled with no incompleteDetails`() {
    let emitter = makeEmitter(parser: NoOpParser())
    _ = emitter.start()
    let final = emitter.finalize(info: FinishInfo(finishReason: .cancelled, inputTokens: 5, outputTokens: 0))

    guard case let .responseCompleted(completed) = final[0] else {
      Issue.record("Expected responseCompleted"); return
    }
    #expect(completed.response.status == .cancelled)
    #expect(completed.response.incompleteDetails == nil)
  }

  @Test
  func `Sequence numbers increase monotonically across start, process, finalize`() {
    let parser = ScriptedParser(
      onProcess: { _ in
        [
          .outputItemAdded(.init(
            item: .message(.init(id: "msg_test")),
            outputIndex: 0,
            sequenceNumber: 0,
          )),
          .outputItemDone(.init(
            item: .message(.init(id: "msg_test", status: .completed)),
            outputIndex: 0,
            sequenceNumber: 1,
          )),
        ]
      },
    )
    let emitter = makeEmitter(parser: parser)
    var events = emitter.start()
    events += emitter.process(text: "hi")
    events += emitter.finalize(info: FinishInfo(finishReason: .stop, inputTokens: 1, outputTokens: 1))

    let sequences = events.map(\.sequenceNumber)
    #expect(sequences == Array(0 ..< sequences.count), "Sequence numbers should be 0..<n with no gaps")
  }

  @Test
  func `Process forwards items into Response.output on the terminal completion`() {
    let parser = ScriptedParser(
      onProcess: { _ in
        [
          .outputItemAdded(.init(
            item: .message(.init(id: "msg_x", status: .inProgress)),
            outputIndex: 0,
            sequenceNumber: 0,
          )),
          .outputItemDone(.init(
            item: .message(.init(
              id: "msg_x",
              content: [.outputText(.init(text: "hello"))],
              status: .completed,
            )),
            outputIndex: 0,
            sequenceNumber: 1,
          )),
        ]
      },
    )
    let emitter = makeEmitter(parser: parser)
    _ = emitter.start()
    _ = emitter.process(text: "hello")
    let final = emitter.finalize(info: FinishInfo(finishReason: .stop, inputTokens: 1, outputTokens: 1))

    guard case let .responseCompleted(completed) = final.last else {
      Issue.record("Expected responseCompleted as terminal event"); return
    }
    #expect(completed.response.output.count == 1)
    if case let .message(m) = completed.response.output[0] {
      #expect(m.status == .completed)
      #expect(m.content.count == 1)
    } else {
      Issue.record("Expected message item in Response.output")
    }
  }

  @Test
  func `Gap-indexed parser items are not fabricated into Response output`() {
    let parser = ScriptedParser(
      onProcess: { _ in
        [
          .outputItemAdded(.init(
            item: .message(.init(id: "msg_gap", status: .completed)),
            outputIndex: 2,
            sequenceNumber: 0,
          )),
        ]
      },
    )
    let emitter = makeEmitter(parser: parser)
    _ = emitter.start()
    _ = emitter.process(text: "gap")
    let final = emitter.finalize(info: FinishInfo(finishReason: .stop, inputTokens: 1, outputTokens: 1))

    guard case let .responseCompleted(completed) = final.last else {
      Issue.record("Expected responseCompleted as terminal event"); return
    }
    #expect(completed.response.output.isEmpty)
  }
}

// MARK: Test helpers

private func makeEmitter(parser: any ResponseFormatParser) -> ResponseStreamEmitter {
  ResponseStreamEmitter(
    parser: parser,
    config: ResponseStreamConfig(model: "test-model", createdAt: 1_700_000_000),
  )
}

/// Test parser that returns pre-recorded events for each call to `process`.
struct ScriptedParser: ResponseFormatParser {
  var onProcess: @Sendable (ParserInput) -> [ResponseStreamingEvent]
  var onFinalize: @Sendable () -> [ResponseStreamingEvent] = { [] }

  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent] {
    onProcess(chunk)
  }

  mutating func finalize() -> [ResponseStreamingEvent] {
    onFinalize()
  }
}
