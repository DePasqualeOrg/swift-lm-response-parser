// Copyright © Anthony DePasquale

@testable import LMResponses
import Testing

@Suite("ResponseStream")
struct ResponseStreamTests {
  @Test
  func `start/process/finalize cover the full lifecycle envelope`() {
    let stream = makeStream(parser: NoOpParser())
    let start = stream.start()
    #expect(start.count == 2)
    guard case .responseCreated = start[0], case .responseInProgress = start[1] else {
      Issue.record("Expected created + in_progress at the head of the stream"); return
    }

    // Tokens 1, 2, 3 decode to "a", "ab", "abc" respectively, so each
    // process(tokenId:) yields a one-character chunk and the parser sees it.
    let mid = (1 ... 3).flatMap { stream.process(tokenId: $0) }
    // NoOpParser emits nothing per chunk.
    #expect(mid.isEmpty)

    let final = stream.finalize(finishReason: .stop, inputTokens: 7)
    #expect(final.count == 1)
    guard case let .responseCompleted(completed) = final[0] else {
      Issue.record("Expected response.completed terminal event"); return
    }
    #expect(completed.response.usage?.inputTokens == 7)
    #expect(completed.response.usage?.outputTokens == 3)
    #expect(completed.response.usage?.totalTokens == 10)
  }

  @Test
  func `length finalize emits response_incomplete and populates finalResponse`() {
    let stream = makeStream(parser: NoOpParser())
    _ = stream.start()

    let final = stream.finalize(finishReason: .length, inputTokens: 7)

    guard case let .responseIncomplete(incomplete) = final[0] else {
      Issue.record("Expected response.incomplete terminal event"); return
    }
    #expect(incomplete.response.status == .incomplete)
    #expect(incomplete.response.incompleteDetails?.reason == .maxOutputTokens)
    #expect(stream.finalResponse == incomplete.response)
  }

  @Test
  func `process forwards detokenized chunks into the parser with aligned token IDs`() {
    let recorder = CallRecorder()
    let parser = ScriptedParser(
      onProcess: { input in
        recorder.record(text: input.text, tokenIds: input.tokenIds)
        return []
      },
    )
    let stream = makeStream(parser: parser)
    _ = stream.start()
    _ = stream.process(tokenId: 1)
    _ = stream.process(tokenId: 2)
    _ = stream.process(tokenId: 3)
    _ = stream.finalize(finishReason: .stop, inputTokens: 0)

    let seen = recorder.calls
    #expect(seen.count == 3, "Each fully-decoded token should reach the parser exactly once")
    #expect(seen.map(\.text) == ["a", "b", "c"])
    // Buffered token IDs should reset each chunk.
    #expect(seen[0].tokenIds == [1])
    #expect(seen[1].tokenIds == [2])
    #expect(seen[2].tokenIds == [3])
  }

  @Test
  func `generatedTokens counts every process call regardless of withholding`() {
    // U+1F600 (😀) is two UTF-16 code units; its UTF-8 encoding is 4 bytes.
    // We emit it across two tokens to exercise the boundary-withholding
    // path: the first token decodes to U+FFFD and is held back.
    let tokenizer = SplitScalarTokenizer()
    let stream = ResponseStream(
      parser: NoOpParser(),
      config: ResponseStreamConfig(model: "test-model"),
      tokenizer: tokenizer,
      promptTokenIds: [],
    )
    _ = stream.start()
    _ = stream.process(tokenId: 1) // decodes to lone replacement char
    _ = stream.process(tokenId: 2) // completes the scalar
    #expect(stream.generatedTokens == 2)

    let final = stream.finalize(finishReason: .stop, inputTokens: 0)
    guard case let .responseCompleted(completed) = final[0] else {
      Issue.record("Expected response.completed"); return
    }
    #expect(completed.response.usage?.outputTokens == 2)
  }

  @Test
  func `responseId is stable across the entire stream`() {
    let stream = makeStream(parser: NoOpParser())
    let start = stream.start()
    _ = stream.process(tokenId: 1)
    let final = stream.finalize(finishReason: .stop, inputTokens: 0)
    let ids = (start + final).map(\.sequenceNumber)
    #expect(ids == Array(0 ..< ids.count), "Sequence numbers should run 0..<n")

    guard case let .responseCreated(created) = start[0],
          case let .responseCompleted(completed) = final[0]
    else {
      Issue.record("Unexpected envelope shape"); return
    }
    #expect(created.response.id == completed.response.id)
    #expect(created.response.id == stream.responseId)
  }

  @Test
  func `items property updates live as events arrive`() {
    let parser = ScriptedParser(
      onProcess: { input in
        guard input.text == "a" else { return [] }
        return [
          .outputItemAdded(.init(
            item: .message(.init(id: "msg_x", status: .inProgress)),
            outputIndex: 0,
            sequenceNumber: 0,
          )),
        ]
      },
      onFinalize: {
        [
          .outputItemDone(.init(
            item: .message(.init(
              id: "msg_x",
              content: [.outputText(.init(text: "hi"))],
              status: .completed,
            )),
            outputIndex: 0,
            sequenceNumber: 1,
          )),
        ]
      },
    )
    let stream = makeStream(parser: parser)
    _ = stream.start()
    #expect(stream.items.isEmpty, "items is empty before any chunks are seen")

    _ = stream.process(tokenId: 1)
    #expect(stream.items.count == 1)
    if case let .message(m) = stream.items[0] {
      #expect(m.status == .inProgress, "items reflect in-progress state mid-stream")
    } else {
      Issue.record("Expected a message item")
    }

    _ = stream.finalize(finishReason: .stop, inputTokens: 0)
    if case let .message(m) = stream.items[0] {
      #expect(m.status == .completed, "items roll forward to completed once output_item.done arrives")
      #expect(m.content.count == 1)
    } else {
      Issue.record("Expected a message item after finalize")
    }
  }

  @Test
  func `prefix-invariant violation resets the detokenizer and the retry chunk carries only the retry token`() {
    // The mock decodes [1] as "a" and [1, 2] as "DIFFERENT" — the
    // two-token decode does not start with "a", so the streaming
    // detokenizer throws on the second consume. The recovery resets
    // the detokenizer and retries `consume(2)` on a fresh stream,
    // where decode([2]) = "b" succeeds. The chunk that reaches the
    // parser must carry tokenIds = [2] only — the bytes for the
    // prior token were lost in the reset.
    let recorder = CallRecorder()
    let parser = ScriptedParser(
      onProcess: { input in
        recorder.record(text: input.text, tokenIds: input.tokenIds)
        return []
      },
    )
    let stream = ResponseStream(
      parser: parser,
      config: ResponseStreamConfig(model: "test-model"),
      tokenizer: NonMonotonicTokenizer(),
      promptTokenIds: [],
    )
    _ = stream.start()
    _ = stream.process(tokenId: 1)
    _ = stream.process(tokenId: 2)

    let calls = recorder.calls
    #expect(calls.count == 2)
    #expect(calls[0].text == "a")
    #expect(calls[0].tokenIds == [1])
    #expect(calls[1].text == "b")
    #expect(calls[1].tokenIds == [2])
  }

  @Test
  func `pass-through decode error drops only the failing token from the pending buffer`() {
    // Token 1 decodes to U+FFFD (mid-scalar withhold) and pending
    // becomes [1]. Token 99 makes decode throw mid-call; the
    // detokenizer's transactional rollback leaves its state intact,
    // and the wrapper drops just the failing token from pending so
    // the original mid-scalar token is preserved. Token 2 completes
    // the scalar and emits with tokenIds = [1, 2].
    let recorder = CallRecorder()
    let parser = ScriptedParser(
      onProcess: { input in
        recorder.record(text: input.text, tokenIds: input.tokenIds)
        return []
      },
    )
    let stream = ResponseStream(
      parser: parser,
      config: ResponseStreamConfig(model: "test-model"),
      tokenizer: SelectiveFailureTokenizer(),
      promptTokenIds: [],
    )
    _ = stream.start()
    _ = stream.process(tokenId: 1)
    _ = stream.process(tokenId: 99)
    _ = stream.process(tokenId: 2)

    #expect(recorder.calls.count == 1)
    #expect(recorder.calls[0].text == "🌍")
    #expect(recorder.calls[0].tokenIds == [1, 2])
  }
}

// MARK: Test helpers

private func makeStream(parser: any ResponseFormatParser) -> ResponseStream {
  ResponseStream(
    parser: parser,
    config: ResponseStreamConfig(model: "test-model", createdAt: 1_700_000_000),
    tokenizer: AlphabetTokenizer(),
    promptTokenIds: [],
  )
}

/// Reference holder for capturing parser-input calls from a `@Sendable`
/// closure without tripping data-race checking.
private final class CallRecorder: @unchecked Sendable {
  private var inner: [(text: String, tokenIds: [Int]?)] = []
  var calls: [(text: String, tokenIds: [Int]?)] {
    inner
  }

  func record(text: String, tokenIds: [Int]?) {
    inner.append((text, tokenIds))
  }
}

/// Token IDs 1, 2, 3 … decode cumulatively to "a", "ab", "abc", … –
/// one ASCII letter per token, so each `process(tokenId:)` produces a
/// one-character chunk.
private struct AlphabetTokenizer: ResponseTokenizer {
  func convertTokenToId(_: String) -> Int? {
    nil
  }

  func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
    []
  }

  func decode(tokenIds: [Int], skipSpecialTokens _: Bool) throws -> String {
    let alphabet = Array("abcdefghijklmnopqrstuvwxyz")
    return String(tokenIds.map { alphabet[($0 - 1) % alphabet.count] })
  }
}

/// Splits U+1F600 (😀) across two tokens. Token 1 alone decodes to a lone
/// U+FFFD (the byte sequence is incomplete); tokens 1+2 together decode to
/// the full emoji. Exercises `StreamingDetokenizer`'s mid-scalar
/// withholding path.
private struct SplitScalarTokenizer: ResponseTokenizer {
  func convertTokenToId(_: String) -> Int? {
    nil
  }

  func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
    []
  }

  func decode(tokenIds: [Int], skipSpecialTokens _: Bool) throws -> String {
    switch tokenIds {
      case [1]: "\u{FFFD}"
      case [1, 2]: "\u{1F600}"
      default: ""
    }
  }
}

/// Decodes [1] as "a" and [2] as "b", but [1, 2] as "DIFFERENT" —
/// breaks the byte-prefix-monotonic invariant the streaming
/// detokenizer requires, triggering its prefix-invariant error on
/// the second token. After a reset, decode([2]) alone succeeds, so
/// the wrapper's retry path produces a clean chunk.
private struct NonMonotonicTokenizer: ResponseTokenizer {
  func convertTokenToId(_: String) -> Int? {
    nil
  }

  func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
    []
  }

  func decode(tokenIds: [Int], skipSpecialTokens _: Bool) throws -> String {
    switch tokenIds {
      case [1]: "a"
      case [2]: "b"
      case [1, 2]: "DIFFERENT"
      default: ""
    }
  }
}

/// Splits 🌍 across tokens 1 and 2, but rejects token 99. Lets us
/// drive a sequence where (a) token 1 returns U+FFFD and is held,
/// (b) token 99 makes decode throw, and (c) token 2 completes the
/// scalar — exercising the wrapper's "drop just the failing token"
/// branch.
private struct SelectiveFailureTokenizer: ResponseTokenizer {
  struct Failure: Error {}

  func convertTokenToId(_: String) -> Int? {
    nil
  }

  func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
    []
  }

  func decode(tokenIds: [Int], skipSpecialTokens _: Bool) throws -> String {
    if tokenIds.contains(99) { throw Failure() }
    var bytes: [UInt8] = []
    for id in tokenIds {
      switch id {
        case 1: bytes.append(contentsOf: [0xF0, 0x9F]) // first half of 🌍
        case 2: bytes.append(contentsOf: [0x8C, 0x8D]) // second half of 🌍
        default: break
      }
    }
    return String(decoding: bytes, as: UTF8.self)
  }
}
