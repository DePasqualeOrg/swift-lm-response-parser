// Copyright © Anthony DePasquale

@testable import LMResponses
import Testing

@Suite("ResponseFormat.normalize")
struct NormalizeTests {
  @Test
  func `HF repo IDs are split and lowercased`() {
    #expect(ResponseFormat.normalize("Qwen/Qwen3-Coder-30B") == "qwen3-coder-30b")
    #expect(ResponseFormat.normalize("meta-llama/Llama-3.2-1B-Instruct") == "llama-3.2-1b-instruct")
  }

  @Test
  func `Local-directory paths are split and lowercased`() {
    #expect(ResponseFormat.normalize("models/Qwen3-Coder-30B") == "qwen3-coder-30b")
  }

  @Test
  func `Bare names are lowercased`() {
    #expect(ResponseFormat.normalize("DeepSeek-V3.2") == "deepseek-v3.2")
  }

  @Test
  func `Empty string round-trips`() {
    #expect(ResponseFormat.normalize("") == "")
  }
}

@Suite("ResponseFormat.infer — name table")
struct NamePrefixDispatchTests {
  @Test
  func `Qwen 3 Coder routes to qwen3Xml`() {
    let f = ResponseFormat.infer(modelName: "Qwen/Qwen3-Coder-30B", modelType: "qwen3", modelConfig: [:])
    #expect(f == .qwen3Xml)
  }

  @Test
  func `Qwen 3.5 routes to qwen3Xml`() {
    let f = ResponseFormat.infer(modelName: "Qwen/Qwen3.5-7B", modelType: "qwen3", modelConfig: [:])
    #expect(f == .qwen3Xml)
  }

  @Test
  func `Qwen 3.6 routes to qwen3Xml by name`() {
    let f = ResponseFormat.infer(modelName: "Qwen/Qwen3.6-35B-A3B-FP8", modelType: "", modelConfig: [:])
    #expect(f == .qwen3Xml)
  }

  @Test
  func `Qwen 3 base (no special suffix) routes to qwen via type fallback`() {
    let f = ResponseFormat.infer(modelName: "Qwen/Qwen3-7B", modelType: "qwen3", modelConfig: [:])
    #expect(f == .qwen)
  }

  @Test
  func `Llama 4 routes to pythonic`() {
    let f = ResponseFormat.infer(modelName: "meta-llama/Llama-4-17B-128E", modelType: "llama4", modelConfig: [:])
    #expect(f == .pythonic)
  }

  @Test
  func `Llama 3.2 routes to llama3`() {
    let f = ResponseFormat.infer(modelName: "meta-llama/Llama-3.2-1B-Instruct", modelType: "llama", modelConfig: ["vocab_size": 128_256])
    #expect(f == .llama3)
  }

  @Test
  func `DeepSeek V3.2-Exp routes to deepseekV31 — longest-prefix tiebreak`() {
    let f = ResponseFormat.infer(
      modelName: "deepseek-ai/DeepSeek-V3.2-Exp",
      modelType: "deepseek_v3",
      modelConfig: [:],
    )
    #expect(f == .deepseekV31, "V3.2-Exp shares V3.1 wire format; longest-prefix wins over deepseek-v3.2")
  }

  @Test
  func `DeepSeek V3.2 base routes to deepseekV32`() {
    let f = ResponseFormat.infer(
      modelName: "deepseek-ai/DeepSeek-V3.2",
      modelType: "deepseek_v3",
      modelConfig: [:],
    )
    #expect(f == .deepseekV32)
  }

  @Test
  func `DeepSeek V3.1 routes to deepseekV31`() {
    let f = ResponseFormat.infer(
      modelName: "deepseek-ai/DeepSeek-V3.1",
      modelType: "deepseek_v3",
      modelConfig: [:],
    )
    #expect(f == .deepseekV31)
  }

  @Test
  func `DeepSeek V3 base routes to deepseekV3`() {
    let f = ResponseFormat.infer(
      modelName: "deepseek-ai/DeepSeek-V3",
      modelType: "deepseek_v3",
      modelConfig: [:],
    )
    #expect(f == .deepseekV3)
  }

  @Test
  func `DeepSeek R1 routes to deepseekR1`() {
    let f = ResponseFormat.infer(modelName: "deepseek-ai/DeepSeek-R1", modelType: "deepseek_v3", modelConfig: [:])
    #expect(f == .deepseekR1)
  }

  @Test
  func `Mistral routes to mistral`() {
    let f = ResponseFormat.infer(modelName: "mistralai/Mistral-7B", modelType: "mistral", modelConfig: [:])
    #expect(f == .mistral)
  }

  @Test
  func `Mixtral routes to mistral`() {
    let f = ResponseFormat.infer(modelName: "mistralai/Mixtral-8x7B", modelType: "mistral", modelConfig: [:])
    #expect(f == .mistral)
  }

  @Test
  func `GPT-OSS routes to harmony`() {
    let f = ResponseFormat.infer(modelName: "openai/gpt-oss-20b", modelType: "gpt_oss", modelConfig: [:])
    #expect(f == .harmony)
  }

  @Test
  func `Gemma 4 routes to gemma4`() {
    let f = ResponseFormat.infer(modelName: "google/Gemma-4-9B", modelType: "gemma4", modelConfig: [:])
    #expect(f == .gemma4)
  }

  @Test
  func `LFM2 routes to lfm2 by name`() {
    let f = ResponseFormat.infer(modelName: "LiquidAI/LFM2-1.2B", modelType: "lfm2", modelConfig: [:])
    #expect(f == .lfm2)
  }

  @Test
  func `LFM2-MoE routes to lfm2`() {
    let f = ResponseFormat.infer(modelName: "LiquidAI/LFM2-8B-A1B-preview", modelType: "lfm2_moe", modelConfig: [:])
    #expect(f == .lfm2)
  }

  @Test
  func `LFM2-VL routes to lfm2`() {
    let f = ResponseFormat.infer(modelName: "LiquidAI/LFM2-VL-3B", modelType: "lfm2_vl", modelConfig: [:])
    #expect(f == .lfm2)
  }

  @Test
  func `OLMo 3 Instruct routes to olmo3`() {
    let f = ResponseFormat.infer(
      modelName: "allenai/Olmo-3-7B-Instruct",
      modelType: "olmo3",
      modelConfig: [:],
    )
    #expect(f == .olmo3)
  }

  @Test
  func `OLMo 3 Think routes to olmo3Thinking`() {
    let f = ResponseFormat.infer(
      modelName: "allenai/Olmo-3-32B-Think",
      modelType: "olmo3",
      modelConfig: [:],
    )
    #expect(f == .olmo3Thinking)
  }

  @Test
  func `Phi-4-mini-instruct routes to phi4Mini`() {
    let f = ResponseFormat.infer(
      modelName: "microsoft/Phi-4-mini-instruct",
      modelType: "phi3",
      modelConfig: [:],
    )
    #expect(f == .phi4Mini)
  }

  @Test
  func `Step-3.5-Flash routes to step3p5 by name`() {
    let f = ResponseFormat.infer(
      modelName: "stepfun-ai/Step-3.5-Flash",
      modelType: "",
      modelConfig: [:],
    )
    #expect(f == .step3p5)
  }

  @Test
  func `Trinity routes to qwen by name`() {
    let f = ResponseFormat.infer(
      modelName: "arcee-ai/Trinity-Mini",
      modelType: "qwen2",
      modelConfig: [:],
    )
    #expect(f == .qwen)
  }

  @Test
  func `Phi-4 base does NOT route to phi4Mini`() {
    let f = ResponseFormat.infer(
      modelName: "microsoft/phi-4",
      modelType: "phi3",
      modelConfig: [:],
    )
    #expect(f == nil, "Phi-4 base has no native tool-call format")
  }
}

@Suite("ResponseFormat.infer — model_type fallback")
struct TypeFallbackDispatchTests {
  @Test
  func `qwen3_5 model_type routes to qwen3Xml even without name signal`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen3_5", modelConfig: [:])
    #expect(f == .qwen3Xml)
  }

  @Test
  func `qwen3_5_moe model_type routes to qwen3Xml for Qwen 3.6 MoE`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen3_5_moe", modelConfig: [:])
    #expect(f == .qwen3Xml)
  }

  @Test
  func `qwen3_next model_type routes to qwen3Xml`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen3_next", modelConfig: [:])
    #expect(f == .qwen3Xml)
  }

  @Test
  func `qwen3_moe model_type routes to qwen`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen3_moe", modelConfig: [:])
    #expect(f == .qwen)
  }

  @Test
  func `qwen2_5_vl model_type routes to qwen (Qwen2.5-VL)`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen2_5_vl", modelConfig: [:])
    #expect(f == .qwen)
  }

  @Test
  func `qwen2_vl model_type routes to qwen (Qwen2-VL)`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen2_vl", modelConfig: [:])
    #expect(f == .qwen)
  }

  @Test
  func `qwen3_vl model_type routes to qwen (Qwen3-VL)`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen3_vl", modelConfig: [:])
    #expect(f == .qwen)
  }

  @Test
  func `qwen2_5 (text) model_type routes to qwen`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen2_5", modelConfig: [:])
    #expect(f == .qwen)
  }

  @Test
  func `gemma4_text model_type routes to gemma4`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "gemma4_text", modelConfig: [:])
    #expect(f == .gemma4)
  }

  @Test
  func `Plain gemma model_type routes to gemmaFunctionCall (Gemma 1/2)`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "gemma", modelConfig: [:])
    #expect(f == .gemmaFunctionCall)
  }

  @Test
  func `gemma2 model_type routes to gemmaFunctionCall (Gemma 2 fine-tunes)`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "gemma2", modelConfig: [:])
    #expect(f == .gemmaFunctionCall)
  }

  @Test
  func `FunctionGemma routes to gemmaFunctionCall by name`() {
    let f = ResponseFormat.infer(
      modelName: "google/functiongemma-270m-it",
      modelType: "gemma",
      modelConfig: [:],
    )
    #expect(f == .gemmaFunctionCall)
  }

  @Test
  func `mistral3 variants route to mistral`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "mistral3_text", modelConfig: [:])
    #expect(f == .mistral)
  }

  @Test
  func `lfm2 family model_types route to lfm2 without name signal`() {
    #expect(ResponseFormat.infer(modelName: "", modelType: "lfm2", modelConfig: [:]) == .lfm2)
    #expect(ResponseFormat.infer(modelName: "", modelType: "lfm2_moe", modelConfig: [:]) == .lfm2)
    #expect(ResponseFormat.infer(modelName: "", modelType: "lfm2_5", modelConfig: [:]) == .lfm2)
    #expect(ResponseFormat.infer(modelName: "", modelType: "lfm2_vl", modelConfig: [:]) == .lfm2)
  }

  @Test
  func `glm4 family model_types route to glm4 without name signal`() {
    #expect(ResponseFormat.infer(modelName: "", modelType: "glm4", modelConfig: [:]) == .glm4)
    #expect(ResponseFormat.infer(modelName: "", modelType: "glm4_moe", modelConfig: [:]) == .glm4)
    #expect(ResponseFormat.infer(modelName: "", modelType: "glm4_moe_lite", modelConfig: [:]) == .glm4)
  }

  @Test
  func `olmo3 model_type routes to olmo3 without thinking signal`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "olmo3", modelConfig: [:])
    #expect(f == .olmo3)
  }

  @Test
  func `Llama 3 by vocab_size disambiguation`() {
    #expect(
      ResponseFormat.infer(modelName: "", modelType: "llama", modelConfig: ["vocab_size": 128_256]) == .llama3,
    )
  }

  @Test
  func `Llama 2 by vocab_size disambiguation falls through to nil`() {
    #expect(
      ResponseFormat.infer(modelName: "", modelType: "llama", modelConfig: ["vocab_size": 32000]) == nil,
      "Llama 2 has 32k vocab and is not in scope; falls through so caller defaults to .json",
    )
  }

  @Test
  func `Llama with no vocab_size falls through to nil (assumed Llama 2)`() {
    #expect(ResponseFormat.infer(modelName: "", modelType: "llama", modelConfig: [:]) == nil)
  }

  @Test
  func `gpt_oss model_type routes to harmony`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "gpt_oss", modelConfig: [:])
    #expect(f == .harmony)
  }

  @Test
  func `Unknown model_type returns nil`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "unknown_arch", modelConfig: [:])
    #expect(f == nil)
  }

  @Test
  func `Empty modelName falls through cleanly to type lookup`() {
    let f = ResponseFormat.infer(modelName: "", modelType: "qwen3", modelConfig: [:])
    #expect(f == .qwen)
  }
}

@Suite("ResponseFormat.infer — name takes priority over type")
struct NameWinsTests {
  @Test
  func `Name match wins even when model_type would resolve differently`() {
    // Qwen 3 Coder model_type is `qwen3` (which would resolve to .qwen);
    // the name signal pulls it to .qwen3Xml.
    let f = ResponseFormat.infer(
      modelName: "Qwen/Qwen3-Coder-30B",
      modelType: "qwen3",
      modelConfig: [:],
    )
    #expect(f == .qwen3Xml)
  }

  @Test
  func `Type fallback used when name is unknown`() {
    let f = ResponseFormat.infer(modelName: "myorg/MyCustomQwen-7B", modelType: "qwen3", modelConfig: [:])
    #expect(f == .qwen)
  }
}
