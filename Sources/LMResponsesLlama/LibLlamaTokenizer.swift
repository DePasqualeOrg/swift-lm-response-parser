// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses

/// `ResponseTokenizer` over `LlamaModel`'s libllama-backed tokenize /
/// detokenize / vocab lookups. The GGUF is the sole source of vocab —
/// callers do not need an HF `tokenizer.json` or `tokenizer_config.json`
/// alongside the model file.
public struct LibLlamaTokenizer: ResponseTokenizer {
  public let model: LlamaModel

  public init(model: LlamaModel) {
    self.model = model
  }

  public func convertTokenToId(_ token: String) -> Int? {
    model.tokenID(for: token).map(Int.init)
  }

  public func encode(text: String, addSpecialTokens: Bool) throws -> [Int] {
    try model.tokenize(text, addSpecial: addSpecialTokens, parseSpecial: true)
      .map(Int.init)
  }

  public func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
    let tokens: [LlamaToken] = tokenIds.map { LlamaToken($0) }
    return try model.detokenize(tokens, skipSpecialTokens: skipSpecialTokens)
  }

  /// EOS token ID, or nil if the model has no canonical EOS.
  public var eosTokenId: Int? {
    let id = model.eosToken
    return id < 0 ? nil : Int(id)
  }

  /// EOT (end-of-turn) token ID, or nil when absent. Distinct from EOS for
  /// chat models with a separate turn terminator (Llama 3.x's `<|eot_id|>`).
  public var eotTokenId: Int? {
    let id = model.eotToken
    return id < 0 ? nil : Int(id)
  }
}
