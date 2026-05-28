// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponsesLlama
import Testing

@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct KVReuseIntegrationTests {
  let fixtures = IntegrationFixtures()

  /// Runs the same two-turn conversation twice on identical sessions:
  /// once with prefix reuse (default), once with the cache discarded
  /// between turns. With a fixed seed both runs must produce identical
  /// output — proof that prefix reuse doesn't drift the sampler.
  @Test func `text prefix reuse matches no reuse`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)

    func runConversation(discardCacheBetween: Bool) async throws -> [String] {
      let model = try await LlamaModel.load(from: ggufURL)
      let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
      let session = ResponseChatSession(
        context: context,
        modelName: fixture.modelName,
        generateParameters: GenerateParameters(seed: 42, maxTokens: 40),
        additionalContext: ["enable_thinking": false],
      )

      var responses: [String] = []
      for (i, prompt) in ["My favorite number is 47.", "What number did I just mention?"].enumerated() {
        var text = ""
        for try await event in session.streamResponseEvents(prompt: prompt) {
          if case let .outputTextDelta(e) = event { text += e.delta }
        }
        responses.append(text.trimmingCharacters(in: .whitespacesAndNewlines))
        if discardCacheBetween, i < 1 {
          await session.discardCachedPrefix()
        }
      }
      return responses
    }

    let withReuse = try await runConversation(discardCacheBetween: false)
    let withoutReuse = try await runConversation(discardCacheBetween: true)

    print("--- With prefix reuse ---")
    for (i, r) in withReuse.enumerated() {
      print("Turn \(i + 1): \(r)")
    }
    print("--- Without prefix reuse ---")
    for (i, r) in withoutReuse.enumerated() {
      print("Turn \(i + 1): \(r)")
    }

    #expect(withReuse == withoutReuse)
  }

  /// Multimodal version. Two-turn vision chat: first turn includes an
  /// image, second is a follow-up. Run once with default (image chunk
  /// reused across turns) and once with the cache discarded between
  /// turns. Same seed → must give identical text.
  @Test func `multimodal chunk reuse matches no reuse`() async throws {
    let fixture = LlamaVLMTestFixture.gemma4_e2b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)

    let red = makeSolidColor(width: 96, height: 96, red: 220, green: 30, blue: 30)

    func runConversation(discardCacheBetween: Bool) async throws -> [String] {
      let model = try await LlamaModel.load(from: modelURL)
      let textContext = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
      let mtmd = try await LlamaMtmdContext.create(
        textContext: textContext,
        mmprojURL: mmprojURL,
        parameters: MtmdContextParameters(mediaMarker: "<|image|>", warmup: false),
      )
      let session = MultimodalResponseChatSession(
        context: mtmd,
        modelName: fixture.modelName,
        generateParameters: GenerateParameters(seed: 42, maxTokens: 32),
      )

      var responses: [String] = []
      let prompts: [(String, [VisionImage])] = [
        ("What color is this image? Answer briefly.", [red]),
        ("What color did you just say?", []),
      ]
      for (i, (prompt, images)) in prompts.enumerated() {
        var text = ""
        for try await event in session.streamResponseEvents(prompt: prompt, images: images) {
          if case let .outputTextDelta(e) = event { text += e.delta }
        }
        responses.append(text.trimmingCharacters(in: .whitespacesAndNewlines))
        if discardCacheBetween, i < prompts.count - 1 {
          await session.discardCachedPrefix()
        }
      }
      return responses
    }

    let withReuse = try await runConversation(discardCacheBetween: false)
    let withoutReuse = try await runConversation(discardCacheBetween: true)

    print("--- Multimodal with prefix reuse ---")
    for (i, r) in withReuse.enumerated() {
      print("Turn \(i + 1): \(r)")
    }
    print("--- Multimodal without prefix reuse ---")
    for (i, r) in withoutReuse.enumerated() {
      print("Turn \(i + 1): \(r)")
    }

    #expect(withReuse == withoutReuse)
  }

  /// Three-turn vision chat where each turn's text chunk extends the
  /// previous turn's (same image throughout). With intra-text-chunk
  /// reuse, each turn's post-image text chunk re-decodes only the new
  /// suffix tokens instead of the whole chunk. Bit-identical output
  /// proves the partial-decode path doesn't drift the sampler.
  @Test func `multimodal intra chunk reuse across three turns`() async throws {
    let fixture = LlamaVLMTestFixture.gemma4_e2b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)

    let red = makeSolidColor(width: 96, height: 96, red: 220, green: 30, blue: 30)
    let prompts: [(String, [VisionImage])] = [
      ("What color is this image?", [red]),
      ("Is it warm or cool?", []),
      ("Name one thing it reminds you of.", []),
    ]

    func runConversation(discardCacheBetween: Bool) async throws -> [String] {
      let model = try await LlamaModel.load(from: modelURL)
      let textContext = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
      let mtmd = try await LlamaMtmdContext.create(
        textContext: textContext,
        mmprojURL: mmprojURL,
        parameters: MtmdContextParameters(mediaMarker: "<|image|>", warmup: false),
      )
      let session = MultimodalResponseChatSession(
        context: mtmd,
        modelName: fixture.modelName,
        generateParameters: GenerateParameters(seed: 42, maxTokens: 24),
      )

      var responses: [String] = []
      for (i, (prompt, images)) in prompts.enumerated() {
        var text = ""
        for try await event in session.streamResponseEvents(prompt: prompt, images: images) {
          if case let .outputTextDelta(e) = event { text += e.delta }
        }
        responses.append(text.trimmingCharacters(in: .whitespacesAndNewlines))
        if discardCacheBetween, i < prompts.count - 1 {
          await session.discardCachedPrefix()
        }
      }
      return responses
    }

    let withReuse = try await runConversation(discardCacheBetween: false)
    let withoutReuse = try await runConversation(discardCacheBetween: true)

    print("--- 3-turn multimodal WITH intra-chunk reuse ---")
    for (i, r) in withReuse.enumerated() {
      print("Turn \(i + 1): \(r)")
    }
    print("--- 3-turn multimodal WITHOUT reuse ---")
    for (i, r) in withoutReuse.enumerated() {
      print("Turn \(i + 1): \(r)")
    }

    #expect(withReuse == withoutReuse)
  }
}

private func makeSolidColor(width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8) -> VisionImage {
  let pixelCount = width * height
  var bytes = [UInt8](repeating: 0, count: pixelCount * 3)
  for i in 0 ..< pixelCount {
    bytes[i * 3 + 0] = red
    bytes[i * 3 + 1] = green
    bytes[i * 3 + 2] = blue
  }
  return VisionImage(width: UInt32(width), height: UInt32(height), rgbData: Data(bytes))
}
