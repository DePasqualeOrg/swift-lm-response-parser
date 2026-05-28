// Copyright © Anthony DePasquale

import Foundation
import Llama
import LMResponses

/// Multimodal counterpart to ``runOnePass``. Drives one generation pass
/// from a pre-prepared mtmd input through the parser/envelope pipeline.
/// Each chunk's events are forwarded through the pass scope (rebased onto
/// the turn-scoped envelope) and ingested into the turn-scoped
/// `itemsBox` so the caller can read pending tool calls from
/// `itemsBox.snapshot` after this call returns.
func runOneMtmdPass<Element: Sendable>(
  on context: LlamaMtmdContext,
  prepared: MtmdPreparedInput,
  startingAtChunk: Int,
  nPast: LlamaPosition,
  promptText: String,
  tokenizer: LibLlamaTokenizer,
  generateParameters: GenerateParameters,
  modelName: String,
  tools: [ToolSpec],
  envelope: ResponseTurnEnvelope,
  transform: @Sendable ([ResponseStreamingEvent]) -> [Element],
  continuation: AsyncThrowingStream<Element, Error>.Continuation,
  usageBox: UsageBox,
  itemsBox: TurnItemsBox,
  format: ResponseFormat?,
  extraEOSTokens: Set<String>,
) async throws -> PassFinishInfo {
  let model = context.textContext.model
  let resolvedFormat = format
    ?? ResponseFormat.infer(
      modelName: modelName,
      modelType: model.architecture ?? "",
      modelConfig: ["vocab_size": Int(model.vocabSize)],
    )
    ?? .json

  let promptTextIds: [Int] = (try? tokenizer.encode(text: promptText, addSpecialTokens: false)) ?? []
  let effectivePriorOutput = try resolvedFormat.combinedPriorOutput(
    fromPromptTokens: promptTextIds,
    tokenizer: tokenizer,
    generatedPriorOutput: nil,
  )
  var parser = resolvedFormat.makeParser(
    tokenizer: tokenizer,
    tools: tools,
    priorOutput: effectivePriorOutput,
  )
  let promptSeedIds = Array(promptTextIds.suffix(7))

  let pass = try runPass(
    on: context,
    prepared: prepared,
    startingAtChunk: startingAtChunk,
    nPast: nPast,
    promptSeedIds: promptSeedIds,
    parameters: generateParameters,
    format: resolvedFormat,
    tokenizer: tokenizer,
    extraEOSTokens: extraEOSTokens,
  )

  return try await drivePassLoop(
    pass: pass,
    parser: &parser,
    envelope: envelope,
    transform: transform,
    continuation: continuation,
    usageBox: usageBox,
    itemsBox: itemsBox,
  )
}
