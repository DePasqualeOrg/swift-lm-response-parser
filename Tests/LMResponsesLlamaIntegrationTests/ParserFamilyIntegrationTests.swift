// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses
import LMResponsesLlama
import Testing

// Parser-family correctness gate. `LMResponses`' parsers match byte-level
// patterns in detokenized strings; llama.cpp's detokenizer can produce
// subtly different whitespace or special-token handling than MLX's, so each
// parser family needs at least one end-to-end pass against a real GGUF
// driven through the `LMResponsesLlama` bridge. The likely failure mode is
// whitespace around special tokens — fixable parser-side, but only caught
// by running the parser against actual llama.cpp output.
//
// Each test pins `format:` explicitly rather than relying on name/arch
// inference, so a regression in the parser itself (not the inference table)
// is what these assertions isolate. `ResponseFormat.infer` is covered
// separately in the unit tests.
//
// One representative checkpoint per family the bridge ships today:
//   - `.qwen`       Qwen3-0.6B (thinking off) / Qwen3-4B (thinking on)
//   - `.llama3`     Llama-3.2-1B-Instruct
//   - `.gemma4`     gemma-4-E2B-it
//   - `.deepseekR1` DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-family stand-in;
//                   see the fixture note on the missing small V3 checkpoint)
//   - `.harmony`    gpt-oss-20b — disabled by default (~11.5 GB, the smallest
//                   Harmony checkpoint that exists; needs 24 GB of RAM or
//                   more — see the test trait)
//
// Gated behind `LMRESPONSES_INTEGRATION_TESTS=1` and `.serialized` so the
// GGUFs load one at a time and generation passes don't contend for GPU
// memory. First run downloads each checkpoint into the standard Hugging Face
// cache; set `HF_OFFLINE=1` to run against already-cached snapshots.

@Suite(.serialized, .enabled(if: integrationTestsEnabled))
struct ParserFamilyIntegrationTests {
  let fixtures = IntegrationFixtures()

  // MARK: Non-reasoning families — clean message + clean stop

  @Test func `llama3 parser yields a clean message`() async throws {
    let response = try await runTurn(
      fixture: .llama3_2_1b,
      format: .llama3,
      instructions: "Be brief.",
      prompt: "Reply with the single word: ok.",
      maxTokens: 64,
    )
    expectCleanMessage(response)
  }

  @Test func `gemma4 parser yields a clean message`() async throws {
    let response = try await runTurn(
      fixture: .gemma4_e2b,
      format: .gemma4,
      instructions: "Be brief.",
      prompt: "Reply with the single word: ok.",
      maxTokens: 64,
    )
    expectCleanMessage(response)
  }

  @Test func `qwen parser with thinking off yields a message and no reasoning`() async throws {
    let response = try await runTurn(
      fixture: .qwen3_0_6b,
      format: .qwen,
      instructions: "Be brief.",
      prompt: "Reply with the single word: ok.",
      thinking: false,
      maxTokens: 64,
    )
    expectCleanMessage(response)
    #expect(
      !response.output.contains { if case .reasoning = $0 { true } else { false } },
      "Thinking-off turn should not produce a reasoning item; got \(kinds(of: response))",
    )
  }

  // MARK: Reasoning families — reasoning item precedes the message

  @Test func `qwen parser with thinking on yields reasoning before the message`() async throws {
    let response = try await runTurn(
      fixture: .qwen3_4b,
      format: .qwen,
      instructions: "Think step by step, then answer.",
      prompt: "What is 7 times 8?",
      thinking: true,
      // Budget enough for 4B to close `<think>` and emit the message. (The
      // 0.6B model reasons past any reasonable budget here, hence the 4B.)
      maxTokens: 1024,
    )
    expectReasoningPrecedesMessage(response)
  }

  @Test func `deepseekR1 parser yields reasoning before the message`() async throws {
    let response = try await runTurn(
      fixture: .deepseekR1_distill_qwen_1_5b,
      format: .deepseekR1,
      prompt: "What is 7 times 8? Then give the final answer.",
      maxTokens: 800,
    )
    expectReasoningPrecedesMessage(response)
  }

  // Disabled by default: gpt-oss-20b (~11.5 GB) is the smallest Harmony
  // checkpoint that exists, and running it requires 24 GB of RAM or more.
  // Remove the trait on a machine with enough memory to exercise the
  // Harmony path.
  @Test(.disabled("Running gpt-oss-20b requires 24 GB of RAM or more"))
  func `harmony parser yields reasoning before the message`() async throws {
    // gpt-oss emits an `analysis` channel (reasoning) before the `final`
    // channel (message). This also exercises the Harmony-specific
    // included-stop-token path: `<|call|>` / `<|return|>` must reach the
    // parser as tokens rather than being dropped after halting (see
    // RunPass.swift / ResponseFormatStopTokenPolicy).
    let response = try await runTurn(
      fixture: .gptOss20b,
      format: .harmony,
      prompt: "What is 7 times 8? Then give the final answer.",
      maxTokens: 800,
    )
    expectReasoningPrecedesMessage(response)
  }

  // MARK: Helpers

  /// Load the fixture's GGUF, build a fresh single-turn session with the
  /// parser format pinned, and run one turn to completion.
  private func runTurn(
    fixture: LlamaTestFixture,
    format: ResponseFormat,
    instructions: String? = nil,
    prompt: String,
    thinking: Bool? = nil,
    maxTokens: Int,
  ) async throws -> Response {
    let ggufURL = try await fixtures.ggufURL(for: fixture)
    let model = try await LlamaModel.load(from: ggufURL)
    let context = try model.makeContext(parameters: ContextParameters(contextLength: 4096))

    var additionalContext: [String: any Sendable]?
    if let thinking {
      additionalContext = ["enable_thinking": thinking]
    }

    let session = ResponseChatSession(
      context: context,
      modelName: fixture.modelName,
      instructions: instructions,
      // Greedy (temperature 0) + fixed seed for a deterministic, reproducible
      // pass. The seed only matters for tie-breaks under greedy sampling.
      generateParameters: GenerateParameters(temperature: 0, seed: 42, maxTokens: maxTokens),
      additionalContext: additionalContext,
      format: format,
    )

    let response = try await session.respond(to: prompt)
    print("--- \(fixture.modelName) [\(format)] ---")
    print("status: \(String(describing: response.status))")
    print("items: \(kinds(of: response))")
    print("text: \(response.outputText)")
    return response
  }

  /// Assert the turn stopped cleanly and produced a message with text.
  private func expectCleanMessage(
    _ response: Response,
    sourceLocation: SourceLocation = #_sourceLocation,
  ) {
    #expect(
      response.status == .completed,
      "Expected status=.completed; got \(String(describing: response.status))",
      sourceLocation: sourceLocation,
    )
    let hasMessage = response.output.contains {
      if case .message = $0 { true } else { false }
    }
    #expect(hasMessage, "Expected a message item; got \(kinds(of: response))", sourceLocation: sourceLocation)
    #expect(
      !response.outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      "Expected non-empty assistant text; got items \(kinds(of: response))",
      sourceLocation: sourceLocation,
    )
  }

  /// Assert the turn produced a reasoning item followed by a message with
  /// non-empty text — the full reasoning → message sequence.
  ///
  /// `status == .completed` is intentionally not pinned: the model may run to
  /// the token budget mid-message (`.incomplete`) yet still have emitted a
  /// valid reasoning → message sequence, which is what this gate checks. The
  /// chosen fixtures (DeepSeek-R1-1.5B, Qwen3-4B) reliably close their
  /// reasoning and start a message within budget on a trivial prompt.
  private func expectReasoningPrecedesMessage(
    _ response: Response,
    sourceLocation: SourceLocation = #_sourceLocation,
  ) {
    let itemKinds = kinds(of: response)
    #expect(
      itemKinds.contains("reasoning"),
      "Expected a reasoning item; got \(itemKinds)",
      sourceLocation: sourceLocation,
    )
    if let firstReasoning = itemKinds.firstIndex(of: "reasoning"),
       let firstMessage = itemKinds.firstIndex(of: "message")
    {
      #expect(
        firstReasoning < firstMessage,
        "Reasoning should precede the message; got \(itemKinds)",
        sourceLocation: sourceLocation,
      )
    }
    #expect(
      !response.outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      "Expected non-empty message text after reasoning; got items \(itemKinds)",
      sourceLocation: sourceLocation,
    )
  }

  /// Item-kind labels in output order, for readable assertion messages.
  private func kinds(of response: Response) -> [String] {
    response.output.map { item in
      switch item {
        case .message: "message"
        case .functionCall: "functionCall"
        case .reasoning: "reasoning"
        case .functionCallOutput: "functionCallOutput"
      }
    }
  }
}
