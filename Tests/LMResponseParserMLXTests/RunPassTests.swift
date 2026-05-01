// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
@testable import LMResponseParserMLX
import MLXLMCommon
import Testing

/// Mock tokenizer with a fixed string→ID dictionary, used by
/// ``effectiveStopTokenIds`` and ``validateStopTokenPolicy`` tests so we
/// don't need a real MLX model.
private final class DictionaryTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
  private let stringToId: [String: Int]
  let bosToken: String?
  let eosToken: String?
  let unknownToken: String?

  init(
    stringToId: [String: Int],
    bosToken: String? = nil,
    eosToken: String? = nil,
    unknownToken: String? = nil,
  ) {
    self.stringToId = stringToId
    self.bosToken = bosToken
    self.eosToken = eosToken
    self.unknownToken = unknownToken
  }

  func convertTokenToId(_ token: String) -> Int? {
    stringToId[token]
  }

  func convertIdToToken(_ id: Int) -> String? {
    stringToId.first(where: { $0.value == id })?.key
  }

  func encode(text _: String, addSpecialTokens _: Bool) -> [Int] {
    []
  }

  func decode(tokenIds _: [Int], skipSpecialTokens _: Bool) -> String {
    ""
  }

  func applyChatTemplate(
    messages _: [[String: any Sendable]],
    tools _: [[String: any Sendable]]?,
    additionalContext _: [String: any Sendable]?,
  ) throws -> [Int] {
    []
  }
}

@Suite("effectiveStopTokenIds")
struct EffectiveStopTokenIdsTests {
  @Test
  func `Includes eosTokenIds, tokenizer eos, tokenizer unknown, and resolved extra tokens`() {
    let tokenizer = DictionaryTokenizer(
      stringToId: [
        "<eos>": 1,
        "<unk>": 2,
        "<|call|>": 100,
        "<|return|>": 101,
      ],
      eosToken: "<eos>",
      unknownToken: "<unk>",
    )
    var configuration = ModelConfiguration(
      id: "test",
      extraEOSTokens: ["<|call|>", "<|return|>"],
    )
    configuration.eosTokenIds = [50]

    let stopSet = effectiveStopTokenIds(
      modelConfiguration: configuration,
      tokenizer: tokenizer,
    )
    #expect(stopSet == Set([1, 2, 50, 100, 101]))
  }

  @Test
  func `Empty extras yield only eos and unknown`() {
    let tokenizer = DictionaryTokenizer(
      stringToId: ["<eos>": 1, "<unk>": 2],
      eosToken: "<eos>",
      unknownToken: "<unk>",
    )
    let stopSet = effectiveStopTokenIds(
      modelConfiguration: ModelConfiguration(id: "test"),
      tokenizer: tokenizer,
    )
    #expect(stopSet == Set([1, 2]))
  }
}

@Suite("validateTokenizerVocabulary")
struct ValidateTokenizerVocabularyTests {
  @Test
  func `Policy with empty requiredExtraEOSTokens passes for any tokenizer`() throws {
    let tokenizer = DictionaryTokenizer(stringToId: [:])
    try validateTokenizerVocabulary(
      policy: .init(includedStopTokens: [], requiredExtraEOSTokens: []),
      format: .json,
      tokenizer: tokenizer,
    )
  }

  @Test
  func `Tokenizer that resolves every required token passes`() throws {
    let tokenizer = DictionaryTokenizer(
      stringToId: ["<|call|>": 100, "<|return|>": 101],
    )
    try validateTokenizerVocabulary(
      policy: ResponseFormat.harmony.stopTokenPolicy,
      format: .harmony,
      tokenizer: tokenizer,
    )
  }

  @Test
  func `Token unknown to tokenizer throws tokenizerMissingRequiredTokens`() {
    let tokenizer = DictionaryTokenizer(stringToId: [:])
    do {
      try validateTokenizerVocabulary(
        policy: ResponseFormat.harmony.stopTokenPolicy,
        format: .harmony,
        tokenizer: tokenizer,
      )
      Issue.record("Expected throw")
    } catch let error as BridgeError {
      guard case let .tokenizerMissingRequiredTokens(_, missing) = error else {
        Issue.record("Expected tokenizerMissingRequiredTokens"); return
      }
      #expect(Set(missing) == Set(["<|call|>", "<|return|>"]))
    } catch {
      Issue.record("Unexpected error: \(error)")
    }
  }

  @Test
  func `Partial vocabulary mismatch reports only the missing tokens`() {
    // Tokenizer knows <|return|> but not <|call|>.
    let tokenizer = DictionaryTokenizer(stringToId: ["<|return|>": 101])
    do {
      try validateTokenizerVocabulary(
        policy: ResponseFormat.harmony.stopTokenPolicy,
        format: .harmony,
        tokenizer: tokenizer,
      )
      Issue.record("Expected throw")
    } catch let error as BridgeError {
      guard case let .tokenizerMissingRequiredTokens(_, missing) = error else {
        Issue.record("Expected tokenizerMissingRequiredTokens"); return
      }
      #expect(Set(missing) == Set(["<|call|>"]))
    } catch {
      Issue.record("Unexpected error: \(error)")
    }
  }

  @Test
  func `Token in includedStopTokens but not requiredExtraEOSTokens is also validated`() {
    // Future-proofing: validation covers the union of both sets, so a
    // hypothetical format with parser-observation tokens disjoint from
    // its halt set still fails loudly when the tokenizer doesn't know
    // them. Without this, the unresolved string would be silently
    // dropped from `includedIds` and the parser would never see the halt.
    let tokenizer = DictionaryTokenizer(stringToId: [:])
    let policy = ResponseFormatStopTokenPolicy(
      includedStopTokens: ["<|observe|>"],
      requiredExtraEOSTokens: [],
    )
    do {
      try validateTokenizerVocabulary(
        policy: policy,
        format: .json,
        tokenizer: tokenizer,
      )
      Issue.record("Expected throw")
    } catch let error as BridgeError {
      guard case let .tokenizerMissingRequiredTokens(_, missing) = error else {
        Issue.record("Expected tokenizerMissingRequiredTokens"); return
      }
      #expect(Set(missing) == Set(["<|observe|>"]))
    } catch {
      Issue.record("Unexpected error: \(error)")
    }
  }

  @Test
  func `BridgeError describes the format and missing tokens`() {
    let error = BridgeError.tokenizerMissingRequiredTokens(
      format: .harmony,
      missing: ["<|call|>", "<|return|>"],
    )
    let message = error.errorDescription ?? ""
    #expect(message.contains("harmony"))
    #expect(message.contains("<|call|>"))
    #expect(message.contains("<|return|>"))
  }
}
