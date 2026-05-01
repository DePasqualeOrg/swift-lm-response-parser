// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
@testable import LMResponseParserMLX
import Testing

@Suite("ResponseStreamHandle.finalResponse")
struct ResponseStreamHandleTests {
  @Test
  func `Defaults to nil when no finalResponse closure is supplied`() async {
    let (stream, _) = AsyncStream<ResponseStreamingEvent>.makeStream()
    let handle = ResponseStreamHandle(stream: stream, awaitCleanup: {})
    let response = await handle.finalResponse()
    #expect(response == nil)
  }

  @Test
  func `Returns whatever the supplied closure produces`() async {
    let (stream, _) = AsyncStream<ResponseStreamingEvent>.makeStream()
    let expected = Response(
      id: "resp_test",
      createdAt: 0,
      model: "test-model",
      output: [],
      status: .completed,
      usage: ResponseUsage(inputTokens: 7, outputTokens: 11, totalTokens: 18),
    )
    let handle = ResponseStreamHandle(
      stream: stream,
      awaitCleanup: {},
      finalResponse: { expected },
    )
    guard let actual = await handle.finalResponse() else {
      Issue.record("finalResponse() returned nil despite closure returning a value")
      return
    }
    #expect(actual.id == expected.id)
    #expect(actual.status == expected.status)
    #expect(actual.usage?.inputTokens == 7)
    #expect(actual.usage?.outputTokens == 11)
  }
}
