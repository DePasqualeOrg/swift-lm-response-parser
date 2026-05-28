// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponsesLlama
import Testing

@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct MultimodalSessionIntegrationTests {
  let fixtures = IntegrationFixtures()

  /// Two-turn multimodal chat. Turn 1: show a red square and ask its
  /// color. Turn 2 (no new image): ask Gemma to repeat what it said.
  /// Proves the session retained both the image and the conversation
  /// history across turns.
  @Test func `multi turn multimodal remembers image`() async throws {
    let fixture = LlamaVLMTestFixture.gemma4_e2b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)
    let model = try await LlamaModel.load(from: modelURL)
    let textContext = try model.makeContext(parameters: ContextParameters(contextLength: 4096))

    // Gemma 4's chat template renders {"type":"image"} as the literal
    // `<|image|>` marker token; configure mtmd to recognize it.
    let mtmd = try await LlamaMtmdContext.create(
      textContext: textContext,
      mmprojURL: mmprojURL,
      parameters: MtmdContextParameters(mediaMarker: "<|image|>", warmup: false),
    )
    #expect(mtmd.supportsVision)

    let session = MultimodalResponseChatSession(
      context: mtmd,
      modelName: fixture.modelName,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 48),
    )

    func runTurn(_ prompt: String, images: [VisionImage] = []) async throws -> String {
      var text = ""
      for try await event in session.streamResponseEvents(prompt: prompt, images: images) {
        if case let .outputTextDelta(e) = event {
          text += e.delta
        }
      }
      return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    let red = makeSolidColor(width: 96, height: 96, red: 220, green: 30, blue: 30)

    let firstAnswer = try await runTurn(
      "What color is this image? Answer in one short sentence.",
      images: [red],
    )
    print("--- Multimodal turn 1 ---")
    print(firstAnswer)

    let followUp = try await runTurn(
      "What color did you just say the image was?",
    )
    print("--- Multimodal turn 2 ---")
    print(followUp)

    #expect(!firstAnswer.isEmpty)
    #expect(!followUp.isEmpty)
    // Both turns should mention "red" — the second only if the session
    // remembered.
    #expect(firstAnswer.localizedCaseInsensitiveContains("red"))
    #expect(followUp.localizedCaseInsensitiveContains("red"))
  }

  /// Multimodal session created with pre-populated history (including a
  /// prior image entry) must replay both the text turn and the image
  /// when answering. Validates the multimodal `init(history:)` path:
  /// images survive re-hydration via `HistoryEntry.user(text:images:)`
  /// and the Jinja template renders the image-marker tokens at the
  /// correct positions on turn 1.
  @Test func `init with history retains image context`() async throws {
    let fixture = LlamaVLMTestFixture.gemma4_e2b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)
    let model = try await LlamaModel.load(from: modelURL)
    let textContext = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
    let mtmd = try await LlamaMtmdContext.create(
      textContext: textContext,
      mmprojURL: mmprojURL,
      parameters: MtmdContextParameters(mediaMarker: "<|image|>", warmup: false),
    )

    let red = makeSolidColor(width: 96, height: 96, red: 220, green: 30, blue: 30)
    let priorHistory: [MultimodalResponseChatSession.HistoryEntry] = [
      .user(text: "What color is this image? Answer briefly.", images: [red]),
      .assistant(text: "The image is red."),
    ]

    let session = MultimodalResponseChatSession(
      context: mtmd,
      modelName: fixture.modelName, history: priorHistory,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 32),
    )

    var text = ""
    for try await event in session.streamResponseEvents(prompt: "What color did you just say?") {
      if case let .outputTextDelta(e) = event { text += e.delta }
    }
    let answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Multimodal re-hydrated answer ---")
    print(answer)
    #expect(answer.localizedCaseInsensitiveContains("red"))
  }

  /// Round-trip a multimodal session through saveCache/init(cache:).
  /// Confirms both the KV state and the chunk-signature sidecar are
  /// restored, and the resumed session answers a follow-up that
  /// requires the prior image context.
  @Test func `save load multimodal cache resumes conversation`() async throws {
    let fixture = LlamaVLMTestFixture.gemma4_e2b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)

    let tempDir = FileManager.default.temporaryDirectory
    let cacheURL = tempDir.appendingPathComponent("mm-session-cache-\(UUID().uuidString).bin")
    defer {
      try? FileManager.default.removeItem(at: cacheURL)
      try? FileManager.default.removeItem(at: cacheURL.appendingPathExtension("sigs"))
    }

    let red = makeSolidColor(width: 96, height: 96, red: 220, green: 30, blue: 30)
    let priorHistory: [MultimodalResponseChatSession.HistoryEntry] = [
      .user(text: "What color is this image? Answer briefly.", images: [red]),
      .assistant(text: "The image is red."),
    ]

    // Phase 1: run the first turn and save cache.
    do {
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
      _ = try await session.respond(
        to: "What color is this image? Answer briefly.",
        images: [red],
      )
      try await session.saveCache(to: cacheURL)
    }

    // Phase 2: re-hydrate. History must mirror what phase 1 wrote so the
    // re-rendered prompt matches the saved-KV's positions.
    let model = try await LlamaModel.load(from: modelURL)
    let textContext = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
    let mtmd = try await LlamaMtmdContext.create(
      textContext: textContext,
      mmprojURL: mmprojURL,
      parameters: MtmdContextParameters(mediaMarker: "<|image|>", warmup: false),
    )
    let resumed = try await MultimodalResponseChatSession(
      context: mtmd,
      modelName: fixture.modelName, history: priorHistory,
      cache: cacheURL,
      generateParameters: GenerateParameters(seed: 42, maxTokens: 32),
    )

    var text = ""
    for try await event in resumed.streamResponseEvents(prompt: "What color did you just say?") {
      if case let .outputTextDelta(e) = event { text += e.delta }
    }
    let answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Multimodal resumed-from-cache answer ---")
    print(answer)
    #expect(answer.localizedCaseInsensitiveContains("red"))
  }

  /// Cancel a multimodal generation mid-stream and assert the session
  /// is still usable for a follow-up turn. The cancellation path
  /// resets KV+signatures to a known-clean state so the next turn
  /// re-encodes from scratch (no stale-position decode failures).
  @Test func `multimodal cancellation allows follow up turn`() async throws {
    let fixture = LlamaVLMTestFixture.gemma4_e2b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)
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
      generateParameters: GenerateParameters(seed: 42, maxTokens: 200),
    )

    let red = makeSolidColor(width: 96, height: 96, red: 220, green: 30, blue: 30)

    // Turn 1: start streaming, break after a few deltas. onTermination
    // cancels the generator task.
    var deltaCount = 0
    do {
      for try await event in session.streamResponseEvents(
        prompt: "Describe this image in great detail.",
        images: [red],
      ) {
        if case .outputTextDelta = event {
          deltaCount += 1
          if deltaCount >= 3 { break }
        }
      }
    }
    #expect(deltaCount >= 3)
    await session.synchronize()

    // Turn 2: fresh prompt with a new image. Cancellation should have
    // reset the KV+signatures so this turn decodes from scratch
    // without conflicting with stale positions.
    let green = makeSolidColor(width: 96, height: 96, red: 30, green: 200, blue: 30)
    var followUp = ""
    for try await event in session.streamResponseEvents(
      prompt: "What color is this image? Answer briefly.",
      images: [green],
    ) {
      if case let .outputTextDelta(e) = event { followUp += e.delta }
    }
    let trimmed = followUp.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Multimodal post-cancellation follow-up ---")
    print(trimmed)
    #expect(!trimmed.isEmpty)
    #expect(trimmed.localizedCaseInsensitiveContains("green"))
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
