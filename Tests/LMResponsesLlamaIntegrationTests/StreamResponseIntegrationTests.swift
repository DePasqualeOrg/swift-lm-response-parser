// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponsesLlama
import Testing
import Tokenizers

@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct StreamResponseIntegrationTests {
  let fixtures = IntegrationFixtures()

  /// Loads Qwen3-0.6B, applies its chat template to a one-shot prompt,
  /// streams response events through `LMResponsesLlama`, and asserts the
  /// model produced visible text. Prints the assembled answer so a manual
  /// run can sanity-check what the model said.
  @Test func `generates and decodes qwen 3 response`() async throws {
    let fixture = LlamaTestFixture.qwen3_0_6b
    let ggufURL = try await fixtures.ggufURL(for: fixture)

    let model = try await LlamaModel.load(from: ggufURL)
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 2048))

    let messages: [[String: any Sendable]] = [
      ["role": "user", "content": "Say hello in one short sentence."],
    ]
    // Disable Qwen3's reasoning so the 64-token budget actually contains
    // the final answer instead of being consumed by a `<think>` block.
    let rendered = try ChatTemplate.render(
      template: model.defaultChatTemplate ?? "",
      messages: messages,
      additionalContext: ["enable_thinking": false],
      specialTokens: .init(
        bos: model.tokenText(for: model.bosToken),
        eos: model.tokenText(for: model.eosToken),
      ),
    )
    let prompt = try model.tokenize(rendered, addSpecial: false, parseSpecial: true)

    let handle = try LMResponsesLlama.streamResponseEvents(
      on: context,
      prompt: prompt,
      parameters: GenerateParameters(seed: 42, maxTokens: 64),
      modelName: fixture.modelName,
      config: ResponseStreamConfig(model: fixture.modelName),
    )

    var collected = ""
    var reasoning = ""
    var sawCompleted = false
    var eventTypeCounts: [String: Int] = [:]
    for await event in handle {
      let typeName = String(describing: event).split(separator: "(", maxSplits: 1).first.map(String.init) ?? "unknown"
      eventTypeCounts[typeName, default: 0] += 1
      switch event {
        case let .outputTextDelta(e):
          collected += e.delta
        case let .reasoningDelta(e):
          reasoning += e.delta
        case .responseCompleted, .responseIncomplete:
          sawCompleted = true
        default:
          break
      }
    }
    print("Event counts: \(eventTypeCounts)")
    if !reasoning.isEmpty {
      print("--- reasoning (first 500 chars) ---")
      print(String(reasoning.prefix(500)))
    }

    let trimmed = collected.trimmingCharacters(in: .whitespacesAndNewlines)
    print("--- Qwen3-0.6B response ---")
    print(trimmed)
    print("---------------------------")

    #expect(sawCompleted)
    #expect(!trimmed.isEmpty)

    let response = await handle.finalResponse()
    #expect(response != nil)
    if let usage = response?.usage {
      print("Tokens — input: \(usage.inputTokens), output: \(usage.outputTokens)")
    }
  }
}
