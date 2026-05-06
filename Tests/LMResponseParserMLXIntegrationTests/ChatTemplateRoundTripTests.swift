// Copyright © Anthony DePasquale

import Foundation
import LMResponseParser
import Testing
@preconcurrency import Tokenizers

/// Round-trip tests that render a synthesized assistant→tool history
/// through the model's actual HuggingFace chat template via
/// `swift-tokenizers`. The shape of the test:
///
/// 1. Mint a `callId` matching the format the parser would produce
///    (wire-extracted for Kimi K2, Mistral-strict for Mistral, generic
///    IDFactory for everything else).
/// 2. Build messages mirroring what the bridge will hand the chat
///    template once mlx-swift-lm gains the tool-call ID transport
///    surface (PR #246 or successor).
/// 3. Assert that `applyChatTemplate` succeeds without raising and that
///    the rendered prompt contains the call ID at the expected position.
@Suite(.serialized)
struct ChatTemplateRoundTripTests {
  static let fixtures = IntegrationTokenizers()

  // MARK: - Mistral

  @Test
  func `Mistral 7B v0.3 accepts a 9-char alphanumeric callId`() async throws {
    let tokenizer = try await Self.fixtures.tokenizer(.mistral7BInstructV0_3)
    // Drive the Mistral parser end-to-end so that a regression in
    // `MistralParser.emitToolCall` (e.g. someone reverting to
    // `IDFactory.make(.callId)`) fails this test, not just the
    // factory-only assertion in `MistralParserTests`.
    let callId = try mistralParserCallId()
    try expectChatTemplateAccepts(
      tokenizer: tokenizer,
      messages: openAIShapedMessages(callId: callId),
      expectedIDInPrompt: callId,
    )
  }

  @Test
  func `Ministral-8B accepts a 9-char alphanumeric callId`() async throws {
    let tokenizer = try await Self.fixtures.tokenizer(.ministral8B)
    let callId = try mistralParserCallId()
    try expectChatTemplateAccepts(
      tokenizer: tokenizer,
      messages: openAIShapedMessages(callId: callId),
      expectedIDInPrompt: callId,
    )
  }

  @Test
  func `Mistral 7B v0.3 rejects the prefixed IDFactory shape`() async throws {
    let tokenizer = try await Self.fixtures.tokenizer(.mistral7BInstructV0_3)
    let badCallId = IDFactory.make(.callId)
    let messages = openAIShapedMessages(callId: badCallId)
    do {
      _ = try tokenizer.applyChatTemplate(
        messages: messages,
        chatTemplate: nil,
        addGenerationPrompt: false,
        truncation: false,
        maxLength: nil,
        tools: nil,
        additionalContext: nil,
      )
      Issue.record("Expected applyChatTemplate to reject \(badCallId.count)-char callId, but it succeeded")
    } catch {
      // Assert on the literal `raise_exception` argument from the
      // upstream Mistral template ("Tool call IDs should be
      // alphanumeric strings with length 9!"). Catching the specific
      // failure mode avoids a false-negative pass when something
      // unrelated throws (config-load failure, downstream tokenization
      // error, etc.). If swift-tokenizers ever stops surfacing the
      // Jinja message, that itself is a regression we want to know
      // about — ergo the substring check rather than a typed-error
      // match.
      let description = "\(error)"
      let mentionsContract =
        description.contains("length 9") ||
        description.lowercased().contains("alphanumeric")
      #expect(
        mentionsContract,
        "Expected Jinja length/alphanumeric error from upstream Mistral template, got: \(description)",
      )
    }
  }

  // MARK: - Kimi K2

  // Kimi K2 ships `tiktoken.model` rather than `tokenizer.json`, so
  // `swift-tokenizers`' `AutoTokenizer.from(directory:)` cannot
  // construct a tokenizer for it. A swift-jinja-direct path is feasible
  // (the chat_template.jinja is bundled in the repo and swift-jinja is
  // already in the dependency graph via swift-tokenizers), but
  // reproducing swift-tokenizers' template-selection wiring —
  // generation-prompt insertion, system-message defaults, tool-spec
  // serialization — for a single test isn't justified before item (5)
  // of `tool-call-id-roundtrip.md` lands. The wire-ID preservation
  // logic this round-trip would otherwise validate is exercised
  // end-to-end by the unit tests in `KimiK2ParserTests` (`callId
  // preserves wire-format functions.NAME:INDEX header` and friends),
  // which assert on the parser's `callId` output without needing a
  // tokenizer.

  // MARK: - Hermes / Llama (ID is correlation-only, never round-tripped)

  @Test
  func `Hermes 3 chat template accepts any callId shape`() async throws {
    let tokenizer = try await Self.fixtures.tokenizer(.hermes3Llama31)
    let callId = IDFactory.make(.callId)
    // The Hermes `tool_use` template never references `tool_call.id`,
    // so the prefixed IDFactory shape is fine. We only assert that the
    // render does not raise; the ID won't appear in the prompt.
    let messages = openAIShapedMessages(callId: callId)
    _ = try tokenizer.applyChatTemplate(
      messages: messages,
      chatTemplate: ChatTemplateArgument.name("tool_use"),
      addGenerationPrompt: false,
      truncation: false,
      maxLength: nil,
      tools: openAIShapedTools(),
      additionalContext: nil,
    )
  }

  @Test
  func `Llama 3.2 chat template accepts any callId shape`() async throws {
    let tokenizer = try await Self.fixtures.tokenizer(.llama3_2_1bInstruct)
    let callId = IDFactory.make(.callId)
    // The Llama 3.2 template likewise never references the ID. Only
    // assert that the render does not raise.
    let messages = openAIShapedMessages(callId: callId)
    _ = try tokenizer.applyChatTemplate(
      messages: messages,
      chatTemplate: nil,
      addGenerationPrompt: false,
      truncation: false,
      maxLength: nil,
      tools: openAIShapedTools(),
      additionalContext: nil,
    )
  }

  // MARK: - Helpers

  /// Drive `MistralParser` end-to-end on a minimal JSON-array tool-call
  /// payload and return the `callId` it emits. Using parser output
  /// rather than `IDFactory.makeMistralStrict()` directly means a
  /// regression in `MistralParser.emitToolCall` (e.g. someone reverting
  /// to `IDFactory.make(.callId)`) fails this test, not just the
  /// factory-only assertion in `MistralParserTests`.
  private func mistralParserCallId() throws -> String {
    let items = parseResponse(
      #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]"#,
      format: .mistral,
      tokenizer: NullParserTokenizer(),
    )
    for item in items {
      if case let .functionCall(f) = item {
        return f.callId
      }
    }
    Issue.record("MistralParser did not emit a function call")
    return ""
  }

  /// Construct an OpenAI-chat-completions-shaped message history with
  /// one assistant tool call carrying `callId` and one tool result that
  /// echoes it. This is exactly the shape mlx-swift-lm's `Chat.Message`
  /// will render once PR #246 (or its successor) lands.
  private func openAIShapedMessages(callId: String) -> [[String: any Sendable]] {
    [
      [
        "role": "user",
        "content": "What's the weather in Paris?",
      ],
      [
        "role": "assistant",
        "content": "",
        "tool_calls": [
          [
            "id": callId,
            "type": "function",
            "function": [
              "name": "get_weather",
              "arguments": #"{"city": "Paris"}"#,
            ] as [String: any Sendable],
          ] as [String: any Sendable],
        ] as [[String: any Sendable]],
      ] as [String: any Sendable],
      [
        "role": "tool",
        "tool_call_id": callId,
        "content": #"{"tempC": 18}"#,
      ],
    ]
  }

  private func openAIShapedTools() -> [[String: any Sendable]] {
    [
      [
        "type": "function",
        "function": [
          "name": "get_weather",
          "description": "Look up the current weather in a city.",
          "parameters": [
            "type": "object",
            "properties": [
              "city": [
                "type": "string",
                "description": "The city to look up.",
              ] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["city"],
          ] as [String: any Sendable],
        ] as [String: any Sendable],
      ],
    ]
  }

  private func expectChatTemplateAccepts(
    tokenizer: any Tokenizers.Tokenizer,
    messages: [[String: any Sendable]],
    expectedIDInPrompt: String,
    sourceLocation: SourceLocation = #_sourceLocation,
  ) throws {
    // We ask for a string render rather than the encoded token IDs so
    // we can assert on the ID's position in the prompt. swift-tokenizers
    // doesn't expose the string render publicly, so re-derive it by
    // decoding the encoded result.
    let tokens = try tokenizer.applyChatTemplate(
      messages: messages,
      chatTemplate: nil,
      addGenerationPrompt: false,
      truncation: false,
      maxLength: nil,
      tools: nil,
      additionalContext: nil,
    )
    let rendered = try tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
    #expect(
      rendered.contains(expectedIDInPrompt),
      "Rendered prompt does not contain expected call ID \(expectedIDInPrompt). Rendered: \(rendered)",
      sourceLocation: sourceLocation,
    )
  }
}

/// Tokenizer satisfying `ParserTokenizer`'s minimal surface for parsers
/// that match on detokenized text. The Mistral parser path doesn't read
/// any of these methods (it operates on the raw input string), so the
/// returns can be trivial.
private struct NullParserTokenizer: ParserTokenizer {
  func convertTokenToId(_: String) -> Int? {
    nil
  }

  func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
    []
  }

  func decode(tokenIds _: [Int], skipSpecialTokens _: Bool) throws -> String {
    ""
  }
}
