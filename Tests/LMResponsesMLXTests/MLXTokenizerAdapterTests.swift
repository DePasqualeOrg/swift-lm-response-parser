// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponsesMLX
import MLXLMCommon
import Testing

/// Trivial `MLXLMCommon.Tokenizer` stand-in for adapter tests. Records
/// every call so we can assert the adapter forwards verbatim.
private final class RecordingTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
  var encodeCalls: [(text: String, addSpecialTokens: Bool)] = []
  var decodeCalls: [(tokenIds: [Int], skipSpecialTokens: Bool)] = []
  var convertCalls: [String] = []

  let bosToken: String? = "<bos>"
  let eosToken: String? = "<eos>"
  let unknownToken: String? = "<unk>"

  func encode(text: String, addSpecialTokens: Bool) throws -> [Int] {
    encodeCalls.append((text, addSpecialTokens))
    return text.utf8.map { Int($0) }
  }

  func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
    decodeCalls.append((tokenIds, skipSpecialTokens))
    return tokenIds.map { String($0) }.joined(separator: ",")
  }

  func convertTokenToId(_ token: String) -> Int? {
    convertCalls.append(token)
    return token.count
  }

  func convertIdToToken(_ id: Int) -> String? {
    String(id)
  }

  func applyChatTemplate(
    messages _: [[String: any Sendable]],
    tools _: [[String: any Sendable]]?,
    additionalContext _: [String: any Sendable]?,
  ) throws -> [Int] {
    []
  }
}

@Suite("MLXTokenizerAdapter")
struct MLXTokenizerAdapterTests {
  @Test
  func `convertTokenToId forwards verbatim`() {
    let underlying = RecordingTokenizer()
    let adapter = MLXTokenizerAdapter(underlying)
    let id = adapter.convertTokenToId("hello")
    #expect(id == 5)
    #expect(underlying.convertCalls == ["hello"])
  }

  @Test
  func `encode forwards arguments verbatim`() throws {
    let underlying = RecordingTokenizer()
    let adapter = MLXTokenizerAdapter(underlying)
    _ = try adapter.encode(text: "abc", addSpecialTokens: false)
    _ = try adapter.encode(text: "abc", addSpecialTokens: true)
    #expect(underlying.encodeCalls.count == 2)
    #expect(underlying.encodeCalls[0].text == "abc")
    #expect(underlying.encodeCalls[0].addSpecialTokens == false)
    #expect(underlying.encodeCalls[1].addSpecialTokens == true)
  }

  @Test
  func `decode forwards arguments verbatim`() throws {
    let underlying = RecordingTokenizer()
    let adapter = MLXTokenizerAdapter(underlying)
    _ = try adapter.decode(tokenIds: [1, 2, 3], skipSpecialTokens: true)
    _ = try adapter.decode(tokenIds: [4, 5], skipSpecialTokens: false)
    #expect(underlying.decodeCalls.count == 2)
    #expect(underlying.decodeCalls[0].tokenIds == [1, 2, 3])
    #expect(underlying.decodeCalls[0].skipSpecialTokens == true)
    #expect(underlying.decodeCalls[1].skipSpecialTokens == false)
  }
}

@Suite("SessionCacheStore")
struct SessionCacheStoreTests {
  @Test
  func `Sequential update calls observe each previous mutation`() async {
    let store = SessionCacheStore<Int>(0)
    await store.update { value in
      value += 5
    }
    await store.read { value in
      #expect(value == 5)
    }
    await store.update { value in
      value *= 2
    }
    await store.read { value in
      #expect(value == 10)
    }
  }

  @Test
  func `Concurrent update calls serialize`() async {
    let store = SessionCacheStore<Int>(0)
    let counter = AsyncCounter()

    await withTaskGroup(of: Void.self) { group in
      for _ in 0 ..< 8 {
        group.addTask {
          await store.update { _ in
            await counter.observeEntry()
            try? await Task.sleep(nanoseconds: 1_000_000)
            await counter.observeExit()
          }
        }
      }
    }

    #expect(await counter.maxConcurrent == 1)
  }
}

private actor AsyncCounter {
  private(set) var maxConcurrent = 0
  private var current = 0

  func observeEntry() {
    current += 1
    if current > maxConcurrent { maxConcurrent = current }
  }

  func observeExit() {
    current -= 1
  }
}
