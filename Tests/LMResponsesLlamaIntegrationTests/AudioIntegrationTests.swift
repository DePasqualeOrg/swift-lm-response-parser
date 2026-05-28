// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponsesLlama
import Testing

@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct AudioIntegrationTests {
  let fixtures = IntegrationFixtures()

  /// End-to-end audio: load Voxtral, hand it a synthesized PCM-F32
  /// sine wave, run the prepared-input pipeline. The transcription
  /// won't say anything meaningful for a pure tone, but the test
  /// validates that:
  ///
  /// 1. `LlamaMtmdContext.supportsAudio` reports true with the audio
  ///    mmproj loaded.
  /// 2. `audioSampleRate` returns a sane value (16 kHz for Whisper-
  ///    style encoders).
  /// 3. `MultimodalInput(media: [.audio(...)])` flows through
  ///    `prepare(input:)` and yields signatures including an `.audio`
  ///    chunk.
  /// 4. `evaluate` + sampling produce at least one output token
  ///    without crashing.
  ///
  /// Either passes (audio path works end-to-end) or surfaces concrete
  /// bugs to fix.
  @Test func `voxtral processes synthetic audio`() async throws {
    let fixture = LlamaVLMTestFixture.voxtral_mini_3b
    let (modelURL, mmprojURL) = try await fixtures.vlmGGUFURLs(for: fixture)
    let model = try await LlamaModel.load(from: modelURL)
    let textContext = try model.makeContext(parameters: ContextParameters(contextLength: 4096))
    let mtmd = try await LlamaMtmdContext.create(
      textContext: textContext,
      mmprojURL: mmprojURL,
      parameters: MtmdContextParameters(warmup: false),
    )

    #expect(mtmd.supportsAudio)
    let sampleRate = mtmd.audioSampleRate
    print("Voxtral audio sample rate: \(sampleRate) Hz")
    #expect(sampleRate > 0)

    // Synthesize ~2 s of a 440 Hz sine wave at the mmproj's expected
    // sample rate. Real callers would feed a WAV-decoded PCM buffer
    // (AVAudioPCMBuffer.floatChannelData, etc.).
    let durationSeconds = 2.0
    let frameCount = Int(Double(sampleRate) * durationSeconds)
    let frequency: Float = 440
    let twoPiOverRate = Float(2 * Double.pi) / Float(sampleRate)
    var samples = [Float](repeating: 0, count: frameCount)
    for i in 0 ..< frameCount {
      samples[i] = sin(twoPiOverRate * frequency * Float(i)) * 0.5
    }
    let clip = AudioClip(samples: samples)

    // Use the default mtmd marker — mtmd substitutes embeddings at
    // its position. Wrap the marker in Voxtral's [INST] format
    // (Mistral chat template). The exact format may not match what
    // Voxtral was trained with for audio; the goal is to confirm
    // mtmd accepts our audio path, not get a useful transcription
    // from a pure tone.
    let marker = llamaMtmdDefaultMarker
    let prompt = "[INST]\(marker)\nTranscribe this audio.[/INST]"

    let input = MultimodalInput(
      prompt: prompt,
      media: [.audio(clip)],
      addSpecialTokens: true,
      parseSpecialTokens: true,
    )

    // Confirm prepare produces an audio chunk with non-zero positions.
    let prepared = try await mtmd.prepare(input: input)
    print("Audio prepared chunks: \(prepared.signatures.count), total nPos: \(prepared.totalPositions)")
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
    #expect(prepared.signatures.contains { if case .audio = $0.kind { true } else { false } })

    // Drive a small generation through the bridge — proves
    // mtmd_helper_eval_chunks handles audio and the sampler loop
    // runs to completion without throwing.
    let handle = try LMResponsesLlama.streamResponseEvents(
      on: mtmd,
      input: input,
      parameters: GenerateParameters(seed: 42, maxTokens: 16),
      modelName: fixture.modelName,
      config: ResponseStreamConfig(model: fixture.modelName),
    )

    var collected = ""
    for await event in handle {
      if case let .outputTextDelta(e) = event {
        collected += e.delta
      }
    }
    print("--- Voxtral output for synthetic sine ---")
    print(collected.trimmingCharacters(in: .whitespacesAndNewlines))
    print("-----------------------------------------")
    // Don't assert on content — a 440 Hz sine wave is nonsense to a
    // speech model. We only care that some text came out without
    // crashing.
    #expect(!collected.isEmpty)
  }
}
