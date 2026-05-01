// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParserMLX
import Testing

@Suite("ResponseChatSessionError")
struct ResponseChatSessionErrorTests {
  @Test
  func `noCacheAvailable description references streamResponseEvents`() {
    let error = ResponseChatSessionError.noCacheAvailable
    let message = error.errorDescription ?? ""
    #expect(message.contains("KV cache"))
    #expect(message.contains("streamResponseEvents"))
    #expect(message.contains("saveCache"))
  }

  @Test
  func `passDidNotFinish description is actionable`() {
    let error = ResponseChatSessionError.passDidNotFinish
    let message = error.errorDescription ?? ""
    // Per swift-error-handling-ui.md, user-facing errors must be
    // actionable. Pin the recovery hint vocabulary so future
    // rewrites preserve it.
    #expect(message.contains("stop reason"))
    #expect(message.contains("Try the request again"))
    #expect(message.contains("bug report"))
  }
}

@Suite("ResponseChatSession.respond drain")
struct ResponseChatSessionRespondDrainTests {
  @Test
  func `drainTerminalResponse returns this stream's terminal response`() async throws {
    let expected = Response(
      id: "resp_current",
      createdAt: 0,
      model: "test-model",
      status: .completed,
    )
    let stream = AsyncThrowingStream<ResponseStreamingEvent, Error> { continuation in
      continuation.yield(.responseCompleted(.init(response: expected, sequenceNumber: 0)))
      continuation.finish()
    }

    let actual = try await ResponseChatSession.drainTerminalResponse(from: stream)
    #expect(actual.id == "resp_current")
    #expect(actual.status == .completed)
  }

  @Test
  func `drainTerminalResponse throws when the stream closes without responseCompleted`() async {
    let stream = AsyncThrowingStream<ResponseStreamingEvent, Error> { continuation in
      continuation.finish()
    }

    do {
      _ = try await ResponseChatSession.drainTerminalResponse(from: stream)
      Issue.record("Expected passDidNotFinish")
    } catch let error as ResponseChatSessionError {
      guard case .passDidNotFinish = error else {
        Issue.record("Expected passDidNotFinish, got \(error)")
        return
      }
    } catch {
      Issue.record("Expected ResponseChatSessionError, got \(error)")
    }
  }
}
