// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses
import LMResponsesLlama
import Testing

/// End-to-end check that `LibLlamaTokenizer` + ``StreamingDetokenizer``
/// produces a chunk stream whose concatenation equals the batch
/// detokenize for a representative set of inputs, across vocab families.
///
/// This is the practical "streaming works" test: it exercises the
/// boundary-buffer fix-up that hides partial multi-byte sequences from
/// the parser layer.
@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct StreamingDetokenizerIntegrationTests {
  let fixtures = IntegrationFixtures()

  static let samples: [(label: String, text: String)] = [
    ("ascii", "Hello, world. This is a streaming test."),
    ("cjk", "日本語のストリーミングテスト。中文流式测试。"),
    ("emoji", "Streaming 🚀 with 🌈 emoji 🎊 in the middle."),
    ("mixed", "Mix: 1234 — café 北京 🦊 done."),
  ]

  @Test(arguments: [LlamaTestFixture.qwen3_0_6b])
  func `chunks concatenate to batch detokenize`(fixture: LlamaTestFixture) async throws {
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    let tokenizer = LibLlamaTokenizer(model: model)

    for (label, text) in Self.samples {
      let tokenIds = try tokenizer.encode(text: text, addSpecialTokens: false)
      let batchDecoded = try tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)

      let detokenizer = tokenizer.streamingDetokenizer()
      var streamed = ""
      for id in tokenIds {
        if let chunk = try detokenizer.consume(id) {
          streamed += chunk
        }
      }
      if let leftover = try detokenizer.flush() {
        streamed += leftover
      }

      #expect(
        streamed == batchDecoded,
        "\(fixture.modelName) streaming != batch for \(label):\n  stream: \(streamed.debugDescription)\n  batch:  \(batchDecoded.debugDescription)",
      )
      #expect(
        streamed == text,
        "\(fixture.modelName) streaming != input for \(label):\n  stream: \(streamed.debugDescription)\n  in:     \(text.debugDescription)",
      )
    }
  }

  /// The detokenizer must never emit a String that ends mid-UTF8-scalar
  /// (i.e. no incomplete surrogate halves leak to the parser). For an
  /// input that splits an emoji across two tokens, the chunk that
  /// contains the partial sequence should be held back until the
  /// completing token arrives — at which point both bytes are emitted
  /// together as a valid scalar.
  @Test(arguments: [LlamaTestFixture.qwen3_0_6b])
  func `chunk boundaries never split scalars`(fixture: LlamaTestFixture) async throws {
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    let tokenizer = LibLlamaTokenizer(model: model)

    // Heavy emoji + CJK content forces multi-token UTF-8 boundaries.
    let text = "🦊🌍🎉 日本語 中文 한국어 🚀✨🌈 — emoji-heavy stream."
    let tokenIds = try tokenizer.encode(text: text, addSpecialTokens: false)

    let detokenizer = tokenizer.streamingDetokenizer()
    for id in tokenIds {
      if let chunk = try detokenizer.consume(id) {
        // Each emitted chunk must be valid UTF-8 with no replacement
        // chars introduced mid-scalar. Re-decoding via String.utf8
        // round-trips the bytes; a replacement char would change them.
        let roundtripped = String(decoding: Array(chunk.utf8), as: UTF8.self)
        #expect(roundtripped == chunk, "chunk \(chunk.debugDescription) failed UTF-8 round-trip")
      }
    }
  }
}
