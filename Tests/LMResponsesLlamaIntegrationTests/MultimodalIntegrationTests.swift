// Copyright © Anthony DePasquale

import CoreGraphics
import Foundation
import Llama
import LMResponsesLlama
import Testing
import Tokenizers

@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct MultimodalIntegrationTests {
  let fixtures = IntegrationFixtures()

  /// Loads Gemma 4 E2B + its mmproj, feeds a real PNG into the
  /// LMResponsesLlama multimodal bridge, and prints the assembled
  /// description text. The image is a CoreGraphics-generated solid red
  /// square — Gemma should at least name the dominant color.
  @Test func `describes red square through bridge`() async throws {
    let fixture = LlamaVLMTestFixture.gemma4_e2b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)

    let model = try await LlamaModel.load(from: modelURL)
    let textContext = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
    let mtmd = try await LlamaMtmdContext.create(
      textContext: textContext,
      mmprojURL: mmprojURL,
      parameters: MtmdContextParameters(warmup: false),
    )

    #expect(mtmd.supportsVision)

    let image = makeSolidColor(width: 96, height: 96, red: 220, green: 30, blue: 30)
    let marker = llamaMtmdDefaultMarker

    // Gemma 4 turn format with the mtmd default media marker. mtmd swaps
    // the marker for the model's image tokens during its own tokenize
    // pass; the surrounding text is what mtmd's internal text-chunk
    // tokenizer sees.
    let prompt = """
    <|turn>user
    \(marker)
    What color is this image? Answer in one short sentence.<turn|>
    <|turn>model

    """

    let input = MultimodalInput(
      prompt: prompt,
      media: [.image(image)],
      addSpecialTokens: true,
      parseSpecialTokens: true,
    )

    let handle = try LMResponsesLlama.streamResponseEvents(
      on: mtmd,
      input: input,
      parameters: GenerateParameters(seed: 42, maxTokens: 48),
      modelName: fixture.modelName,
      config: ResponseStreamConfig(model: fixture.modelName),
    )

    var collected = ""
    var sawCompleted = false
    for await event in handle {
      switch event {
        case let .outputTextDelta(e):
          collected += e.delta
        case .responseCompleted, .responseIncomplete:
          sawCompleted = true
        default:
          break
      }
    }

    let trimmed = collected.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Gemma 4 E2B multimodal response ---")
    print(trimmed)
    print("---------------------------------------")

    #expect(sawCompleted)
    #expect(!trimmed.isEmpty)

    if let usage = await handle.finalResponse()?.usage {
      print("Tokens — input: \(usage.inputTokens), output: \(usage.outputTokens)")
    }
  }

  /// Measurement-only test: prints the mtmd chunk count and per-chunk
  /// sizes for a typical single-image vision turn. Helpful to confirm
  /// we're not paying extra batches on turn 1 vs upstream `mtmd-cli`.
  @Test func `reports chunk count for typical vision turn`() async throws {
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
    let renderedPrompt = try ChatTemplate.render(
      template: model.defaultChatTemplate ?? "",
      messages: [
        ["role": "user", "content": [
          ["type": "image"],
          ["type": "text", "text": "What color is this image?"],
        ] as [any Sendable]],
      ],
      specialTokens: .init(
        bos: model.tokenText(for: model.bosToken),
        eos: model.tokenText(for: model.eosToken),
      ),
    )

    let input = MultimodalInput(
      prompt: renderedPrompt,
      media: [.image(red)],
      addSpecialTokens: false,
      parseSpecialTokens: true,
    )
    let prepared = try await mtmd.prepare(input: input)

    print("=== Chunk efficiency for typical Gemma 4 vision turn ===")
    print("Total chunks: \(prepared.signatures.count)")
    print("Total positions (nPos sum): \(prepared.totalPositions)")
    for (i, sig) in prepared.signatures.enumerated() {
      switch sig.kind {
        case let .text(tokens):
          print("  Chunk \(i): text, \(tokens.count) tokens, nPos=\(sig.nPos)")
        case let .image(id):
          print("  Chunk \(i): image \(id.prefix(8))…, nPos=\(sig.nPos)")
        case let .audio(id):
          print("  Chunk \(i): audio \(id.prefix(8))…, nPos=\(sig.nPos)")
      }
    }
    print("========================================================")

    #expect(prepared.signatures.count >= 2) // at minimum: text-before + image
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
