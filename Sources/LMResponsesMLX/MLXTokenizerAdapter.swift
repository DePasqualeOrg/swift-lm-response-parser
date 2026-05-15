// Copyright © Anthony DePasquale

import Foundation
import LMResponses
import MLXLMCommon

/// Bridges `any MLXLMCommon.Tokenizer` to `any ResponseTokenizer`.
///
/// `ResponseTokenizer`'s three method signatures match `MLXLMCommon.Tokenizer`'s
/// exactly, but Swift cannot pass `any MLXLMCommon.Tokenizer` where
/// `any ResponseTokenizer` is expected: protocol-existential-to-protocol-existential
/// conformance requires concrete-type conformance, not method-signature
/// equivalence. The adapter is a thin wrapper that forwards every call.
///
/// Internal to the bridge – consumers continue to deal in
/// `any MLXLMCommon.Tokenizer`.
struct MLXTokenizerAdapter: ResponseTokenizer {
  let underlying: any MLXLMCommon.Tokenizer

  init(_ underlying: any MLXLMCommon.Tokenizer) {
    self.underlying = underlying
  }

  func convertTokenToId(_ token: String) -> Int? {
    underlying.convertTokenToId(token)
  }

  func encode(text: String, addSpecialTokens: Bool) throws -> [Int] {
    try underlying.encode(text: text, addSpecialTokens: addSpecialTokens)
  }

  func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
    try underlying.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
  }
}
