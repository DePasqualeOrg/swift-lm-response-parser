// Copyright © Anthony DePasquale

import Foundation

/// One of the per-format response shapes the parser supports.
///
/// Cases are added as their parsers land. The `.json` fallback is what
/// callers default to when neither name nor type inference resolves to a
/// known shape.
public enum ResponseFormat: Sendable, Equatable, CaseIterable {
  /// Hermes-style XML `<tool_call>…</tool_call>` with JSON arguments.
  case hermes
  /// Qwen 2.5 / Qwen 3 base – Hermes-style JSON tool calls.
  case qwen
  /// Qwen 3 Coder / Qwen 3.5+ – XML function format.
  case qwen3Xml
  /// DeepSeek-R1-style `<think>` reasoning + tool-call interleave.
  case deepseekR1
  /// DeepSeek V3 base tool-call format.
  case deepseekV3
  /// DeepSeek V3.1 tool-call format. Also the dispatch target for
  /// V3.2-Exp, which shares V3.1's wire format.
  case deepseekV31
  /// DeepSeek V3.2 base – DSML format (distinct from V3.1).
  case deepseekV32
  /// Mistral function-call format.
  case mistral
  /// Llama 3 / 3.1 / 3.2 inline JSON format.
  case llama3
  /// Pythonic literal-syntax tool calls (Llama 4 and similar).
  case pythonic
  /// LFM2 (Liquid AI) – Pythonic literal syntax wrapped in
  /// `<|tool_call_start|>` / `<|tool_call_end|>` tokens.
  case lfm2
  /// Phi-4-mini-instruct – `functools[{json}, ...]` JSON-array tool-call
  /// format. Distinct from Phi-4 reasoning variants.
  case phi4Mini
  /// Phi-4 reasoning variants (`Phi-4-reasoning`, `-reasoning-plus`,
  /// `Phi-4-mini-reasoning`) – `<think>...</think>` reasoning blocks
  /// only, no tool calls.
  case phiReasoning
  /// Gemma 1/2 and Google's `functiongemma` model – function calls in
  /// `<start_function_call>call:NAME{key:<escape>val<escape>}<end_function_call>`
  /// shape. Distinct from `gemma4` (multi-token markers).
  case gemmaFunctionCall
  /// GPT-OSS / Harmony reserved-token protocol.
  case harmony
  /// Gemma 4 multi-token thinking-marker format.
  case gemma4
  /// Kimi K2 reserved-token tool-call format (Instruct variant – no
  /// implicit reasoning preamble).
  case kimiK2
  /// Kimi K2 Thinking variant – same tool-call wire shape as
  /// ``kimiK2``, but the model output begins inside an implicit
  /// reasoning preamble that ends at the first `</think>` or at
  /// `<|tool_calls_section_begin|>` (whichever comes first). The
  /// `<think>` opener is optional.
  case kimiK2Thinking
  /// MiniMax M2 – XML invoke/parameter shape with implicit-reasoning
  /// preamble (everything before `</think>` is reasoning).
  case miniMaxM2
  /// MiniMax-Text-01 / M1 (the non-M2 line) – `<tool_calls>` envelope
  /// containing NDJSON (one `{name, arguments}` JSON object per line,
  /// not a JSON array). Tool calls inside `<think>...</think>` are
  /// filtered out. Distinct from ``miniMaxM2``.
  case miniMax
  /// GLM 4.x – XML invoke with `<arg_key>`/`<arg_value>` parameter pairs.
  case glm4
  /// GLM 4.5+ thinking variants – same tool-call wire shape as
  /// ``glm4``, with an implicit `<think>` reasoning preamble that ends at
  /// `</think>` or at `<tool_call>`.
  case glm4Thinking
  /// Meituan LongCat-Flash – Hermes-shaped JSON tool call wrapped in
  /// `<longcat_tool_call>` / `</longcat_tool_call>` envelope tokens.
  case longcat
  /// IBM Granite-20B-FunctionCalling – `<function_call>` marker (plain
  /// string) followed by a single JSON object `{name, arguments}`. Calls
  /// are delimited by the next marker or end of stream; no closing tag.
  /// Distinct from ``granite`` (3.x), which uses a JSON array, and from
  /// ``granite4`` (Hermes envelope).
  case granite20bFc
  /// IBM Granite 3.x – optional `<|tool_call|>` (3.0) or `<tool_call>`
  /// (3.1) marker followed by a JSON array of `{name, arguments}` objects.
  case granite
  /// IBM Granite 4 – Hermes-shaped `<tool_call>{...}</tool_call>` envelope
  /// where the `arguments` field may be either an object or a JSON-encoded
  /// string. Distinct from ``granite`` (3.x), which uses a JSON array.
  case granite4
  /// Shanghai AI Lab InternLM 2.x and Intern-S1 – `<|action_start|><|plugin|>`
  /// + JSON object + `<|action_end|>` envelope. Both spaced
  /// (`<|action_start|> <|plugin|>`) and unspaced variants are accepted.
  case internlm
  /// AI21 Jamba 1.5 / 1.7 – `<tool_calls>` envelope wrapping a JSON array
  /// of `{name, arguments}` objects.
  case jamba
  /// Tencent Hunyuan A13B – `<think>...</think>` reasoning preamble plus
  /// optional `<answer>...</answer>` envelope, with `<tool_calls>` JSON
  /// array tool-call envelopes inside the answer block. Tool calls
  /// inside the reasoning block are filtered out.
  case hunyuanA13B
  /// Magistral – Mistral tool-call format plus a `[THINK]...[/THINK]`
  /// reasoning preamble. Distinct from ``mistral`` (base Mistral and
  /// Mixtral don't emit the reasoning markers).
  case magistral
  /// Allen AI OLMo 3 – Pythonic calls separated by newlines (not
  /// commas) and wrapped in `<function_calls>...</function_calls>` XML.
  /// Argument values may use either Python literals (`True` / `False`
  /// / `None`) or JSON literals (`true` / `false` / `null`). Distinct
  /// routing target from ``pythonic`` because the inner shape and
  /// wrapper differ.
  case olmo3
  /// Allen AI OLMo 3 Think variants – same tool-call syntax as
  /// ``olmo3``, plus an implicit `<think>` reasoning preamble that
  /// ends at `</think>`.
  case olmo3Thinking
  /// Salesforce xLAM – JSON array of `{name, arguments}` objects,
  /// optionally wrapped in any of four envelopes: `<tool_call>` …
  /// `</tool_call>`, `[TOOL_CALLS]`, a fenced JSON code block, or bare.
  /// Distinct from ``hermes`` (which wraps a single object).
  case xlam
  /// ByteDance Seed-OSS – Qwen 3 Coder XML body wrapped in a
  /// `<seed:tool_call>` / `</seed:tool_call>` envelope, plus
  /// `<seed:think>` / `</seed:think>` reasoning markers. Mirrors
  /// vLLM's `SeedOssToolParser`, which itself is a Qwen 3 Coder XML
  /// parser with renamed envelope tokens.
  case seedOss
  /// StepFun Step-3.5-Flash – Qwen 3 Coder XML tool-call shape with a
  /// reasoning quirk: the model habitually emits a stray `\n`
  /// immediately before and/or after `</think>`. Mirrors vLLM's
  /// `Step3p5ReasoningParser`. Distinct routing target from
  /// ``qwen3Xml`` because of the newline trim.
  case step3p5
  /// Baidu ERNIE 4.5 – Hermes-shaped `<tool_call>{json}</tool_call>`
  /// envelopes plus an ERNIE-specific `</think>` reasoning closer
  /// (opener typically injected into the prompt) and an optional
  /// `<response>...</response>` content envelope. Reasoning is opt-in
  /// via direct construction with `acceptThink: true`.
  case ernie
  /// Baidu ERNIE 4.5 Thinking variants – same tool-call wire shape as
  /// ``ernie``, with an implicit reasoning preamble that ends at
  /// `</think>`.
  case ernieThinking
  /// Best-effort top-level JSON tool-call detection. Used as a fallback
  /// when neither the name nor model_type tables resolve.
  case json
}

extension ResponseFormat {
  /// Normalize a model name for prefix matching. Splits on `/` and takes
  /// the last component, then lowercases. Handles HF repo IDs
  /// (`Qwen/Qwen3-Coder-30B`), local-directory paths
  /// (`models/Qwen3-Coder-30B`), and bare names uniformly – all normalize
  /// to `qwen3-coder-30b` and match the same `qwen3-coder` prefix in
  /// ``namePrefixes``.
  package static func normalize(_ modelName: String) -> String {
    (modelName.split(separator: "/").last.map(String.init) ?? modelName)
      .lowercased()
  }

  /// Name-prefix table. Match is `String.hasPrefix` on the
  /// ``normalize(_:)``-d model name; among matching entries, longest
  /// prefix wins (so `deepseek-v3.2-exp` defeats `deepseek-v3` and routes
  /// to ``deepseekV31`` because V3.2-Exp shares V3.1's wire format).
  ///
  /// Order in the array is presentation, not match precedence – the
  /// longest-prefix tiebreak picks the winner regardless of order. Kept
  /// loosely sorted by family for readability.
  package static let namePrefixes: [(prefix: String, format: ResponseFormat)] = [
    // Qwen 3 Coder / Qwen 3.5+ – XML function format.
    ("qwen3-coder", .qwen3Xml),
    ("qwen3.6", .qwen3Xml),
    ("qwen3.5", .qwen3Xml),

    // StepFun Step-3.5-Flash. SGLang aliases its `step3p5` registry
    // key to `Qwen3CoderDetector`; for well-formed output the
    // tool-call format is identical to Qwen 3 Coder XML. Reasoning
    // adds a newline-trim quirk around `</think>`, hence the
    // dedicated `.step3p5` case rather than falling through to
    // `.qwen3Xml`. The lenient malformation repair logic in vLLM's
    // `step3p5_tool_parser.py` is not ported.
    ("step-3.5", .step3p5),
    ("stepfun-ai-step-3.5", .step3p5),

    // Llama variants.
    ("llama-4", .pythonic),
    ("meta-llama-4", .pythonic),
    ("llama-3", .llama3),
    ("meta-llama-3", .llama3),

    // DeepSeek – V3.2-Exp wins over V3.2 wins over V3.1 wins over V3.
    ("deepseek-v3.2-exp", .deepseekV31),
    ("deepseek-v3.2", .deepseekV32),
    ("deepseek-v3.1", .deepseekV31),
    ("deepseek-v3", .deepseekV3),
    ("deepseek-r1", .deepseekR1),

    // Mistral. Magistral wins via longer-prefix tiebreak so the
    // reasoning preamble path is selected for Magistral checkpoints.
    ("magistral", .magistral),
    ("mistralai-magistral", .magistral),
    ("mistral", .mistral),
    ("mixtral", .mistral),

    // GPT-OSS / Harmony.
    ("gpt-oss", .harmony),

    // Gemma 4.
    ("gemma-4", .gemma4),

    // Arcee Trinity – Qwen 2.5 / Qwen 3 base wire format with tool
    // calls that may appear inside `<think>...</think>`. The Qwen
    // parser handles `<tool_call>` mid-reasoning as an implicit
    // reasoning end, so the same `.qwen` route works.
    ("trinity", .qwen),
    ("arcee-trinity", .qwen),
    ("arceeai-trinity", .qwen),

    // Kimi K2. Longest-prefix tiebreak picks `kimi-k2-thinking`
    // before falling back to `kimi-k2` for the Instruct variant.
    ("kimi-k2-thinking", .kimiK2Thinking),
    ("moonshotai-kimi-k2-thinking", .kimiK2Thinking),
    ("kimi-k2", .kimiK2),
    ("moonshotai-kimi-k2", .kimiK2),

    // MiniMax M2 (longest-prefix wins; comes before the generic
    // `minimax` / `minimax-m1` prefixes for the older Text-01 / M1).
    ("minimax-m2", .miniMaxM2),
    ("minimaxai-minimax-m2", .miniMaxM2),

    // MiniMax-Text-01 / M1 – older non-M2 line.
    ("minimax-m1", .miniMax),
    ("minimaxai-minimax-m1", .miniMax),
    ("minimax-text-01", .miniMax),
    ("minimaxai-minimax-text-01", .miniMax),

    // GLM 4.x. Thinking variants win via longer-prefix tiebreak before
    // falling back to the non-thinking base GLM 4 route.
    ("glm-4.5", .glm4Thinking),
    ("glm-4.6", .glm4Thinking),
    ("glm-4.7", .glm4Thinking),
    ("glm-5", .glm4Thinking),
    ("glm45", .glm4Thinking),
    ("glm5", .glm4Thinking),
    ("glm-4", .glm4),
    ("zhipuai-glm-4", .glm4),

    // Meituan LongCat-Flash. Hermes-shaped with renamed envelope.
    ("longcat", .longcat),
    ("meituan-longcat", .longcat),

    // IBM Granite-20B-FunctionCalling – distinct single-checkpoint
    // wire format. Routed via prefix because the model name varies.
    ("granite-20b-functioncalling", .granite20bFc),
    ("ibm-granite-granite-20b-functioncalling", .granite20bFc),

    // IBM Granite 3.x. Granite 4 has its own Hermes-shaped wire format
    // and is not routed here.
    ("granite-3", .granite),
    ("ibm-granite-granite-3", .granite),

    // IBM Granite 4 – Hermes-shaped wire format with optional
    // string-encoded `arguments`. Distinct routing target from
    // `.granite` because the parser variant differs.
    ("granite-4", .granite4),
    ("ibm-granite-granite-4", .granite4),

    // Shanghai AI Lab InternLM 2.x and Intern-S1.
    ("internlm2", .internlm),
    ("internlm-2", .internlm),
    ("internlm-s1", .internlm),
    ("intern-s1", .internlm),
    ("internlm-internlm2", .internlm),
    ("internlm-internlm-2", .internlm),
    ("internlm-intern-s1", .internlm),

    // AI21 Jamba 1.5 / 1.7. Mistral-tokenizer Jambas are explicitly
    // rejected upstream, but we don't have a way to distinguish them
    // here; callers should override the format for those checkpoints.
    ("jamba", .jamba),
    ("ai21-jamba", .jamba),
    ("ai21labs-jamba", .jamba),

    // Tencent Hunyuan A13B. Distinct from Hunyuan V3 (different
    // wire format, not yet covered).
    ("hunyuan-a13b", .hunyuanA13B),
    ("tencent-hunyuan-a13b", .hunyuanA13B),

    // LFM2 (Liquid AI).
    ("lfm2", .lfm2),
    ("liquidai-lfm2", .lfm2),

    // Allen AI OLMo 3 – pythonic calls inside a `<function_calls>`
    // wrapper, newline-separated. Distinct from `.pythonic` (Llama 4)
    // because of the wrapper and inner separator.
    ("olmo-3", .olmo3),
    ("olmo3", .olmo3),
    ("allenai-olmo-3", .olmo3),
    ("allenai-olmo3", .olmo3),

    // Phi-4-mini-instruct. Specific prefix to avoid catching the base
    // `phi-4` model (no native tool-call format) or the
    // `phi-4-mini-reasoning` / `phi-4-reasoning` variants.
    ("phi-4-mini-instruct", .phi4Mini),
    ("microsoft-phi-4-mini-instruct", .phi4Mini),

    // Phi-4 reasoning variants. The reasoning prefixes are listed
    // longer-first so longest-prefix tiebreak picks the right entry –
    // `phi-4-reasoning-plus` is matched by the more specific entry
    // before falling back to `phi-4-reasoning`. `phi-4-mini-reasoning`
    // shares the `<think>` format and routes here too.
    ("phi-4-reasoning-plus", .phiReasoning),
    ("phi-4-reasoning", .phiReasoning),
    ("phi-4-mini-reasoning", .phiReasoning),
    ("microsoft-phi-4-reasoning-plus", .phiReasoning),
    ("microsoft-phi-4-reasoning", .phiReasoning),
    ("microsoft-phi-4-mini-reasoning", .phiReasoning),

    // Google FunctionGemma standalone model.
    ("functiongemma", .gemmaFunctionCall),
    ("google-functiongemma", .gemmaFunctionCall),

    // Baidu ERNIE 4.5 family.
    ("ernie-4.5", .ernie),
    ("ernie-4_5", .ernie),
    ("baidu-ernie-4.5", .ernie),
    ("baidu-ernie-4_5", .ernie),

    // Salesforce xLAM family. Multiple base archs (Llama, Qwen) all
    // share the xLAM tool-call wire format.
    ("xlam", .xlam),
    ("salesforce-xlam", .xlam),
    ("salesforce-llama-xlam", .xlam),
    ("salesforce-qwen-xlam", .xlam),
    ("llama-xlam", .xlam),
    ("qwen-xlam", .xlam),

    // ByteDance Seed-OSS family.
    ("seed-oss", .seedOss),
    ("seed_oss", .seedOss),
    ("bytedance-seed-seed-oss", .seedOss),
    ("bytedance-seed-oss", .seedOss),
  ]

  /// Type-prefix table. Match is `String.hasPrefix` on the lowercased
  /// `model_type` value; among matching entries, longest prefix wins so
  /// variant families (`qwen3_5`, `qwen3_moe`, `granitemoehybrid`)
  /// dispatch to their own format ahead of the shorter base-family
  /// prefix (`qwen3`, `granite`).
  ///
  /// Order in the array is presentation, not match precedence – kept
  /// loosely sorted by family for readability. Llama is the one entry
  /// missing here: `llama` can be either Llama 2 or Llama 3 depending
  /// on `vocab_size`, so it's resolved with a config-aware special case
  /// in ``resolveByType(_:config:)``.
  package static let typePrefixes: [(prefix: String, format: ResponseFormat)] = [
    // Qwen variant families. `qwen3_5` / `qwen3_next` are Qwen 3 Coder /
    // 3.5; the `qwen3_vl`, `qwen2_5*`, `qwen2_vl` families share the
    // base Hermes-style shape and route to `.qwen`.
    ("qwen3_5", .qwen3Xml),
    ("qwen3_next", .qwen3Xml),
    ("qwen3_moe", .qwen),
    ("qwen3_vl", .qwen),
    ("qwen2_5", .qwen),
    ("qwen2_vl", .qwen),
    ("qwen3", .qwen),
    ("qwen2", .qwen),

    // Gemma 4 multi-token thinking-marker family.
    ("gemma4", .gemma4),

    // Gemma 1 / 2 base have no native tool-call format; route to
    // FunctionGemma's shape, which Gemma 1/2 fine-tunes adopt when
    // opting into tool calling.
    ("gemma2", .gemmaFunctionCall),
    ("gemma", .gemmaFunctionCall),

    // Mistral / Mistral 3 share a target.
    ("mistral3", .mistral),
    ("mistral", .mistral),

    // LFM2 family – `lfm2`, `lfm2_moe`, `lfm2_5`, `lfm2_vl`.
    ("lfm2", .lfm2),

    // Allen AI OLMo 3 – mlx-swift-lm registers the arch as `olmo3`.
    ("olmo3", .olmo3),

    // Baidu ERNIE 4.5.
    ("ernie4_5", .ernie),
    ("ernie4.5", .ernie),

    // ByteDance Seed-OSS.
    ("seed_oss", .seedOss),

    // GLM 4 family. GLM 4.5 / 4.6 / 4.7 all report `glm4_moe`.
    ("glm4", .glm4),

    // IBM Granite. Granite 4.0 H (hybrid Mamba) reports
    // `granitemoehybrid` and uses the Hermes-style
    // `<tool_call>{...}</tool_call>` envelope. Granite 3.x base / MoE
    // report `granite` / `granitemoe` and use the older
    // `<|tool_call|>` / `<tool_call>` + JSON-array format.
    ("granitemoehybrid", .granite4),
    ("granitemoe", .granite),
    ("granite", .granite),

    // DeepSeek base variants. The successor families (`deepseek_v3.1`,
    // `deepseek_v3.2`, `_exp`) are routed by name-prefix.
    ("deepseek_v3", .deepseekV3),
    ("deepseek_r1", .deepseekR1),

    // GPT-OSS / Harmony reserved-token protocol.
    ("gpt_oss", .harmony),
  ]

  /// Resolve a response format from `model_type` plus optional secondary
  /// signals from `config.json`. Returns nil when the model_type does not
  /// uniquely identify one of the supported formats; the caller falls
  /// through to ``ResponseFormat/json`` in that case.
  ///
  /// `llama` is handled outside the prefix table because Llama 2 and
  /// Llama 3 share the same `model_type` and need a `vocab_size` lookup
  /// to disambiguate.
  package static func resolveByType(
    _ modelType: String,
    config: [String: any Sendable],
  ) -> ResponseFormat? {
    let type = modelType.lowercased()
    if type.isEmpty { return nil }

    if type == "llama" {
      // Llama 3 vs Llama 2 – same model_type, different vocab.
      // Llama 3's vocab is 128k+ tokens; Llama 2's is 32k.
      let vocabSize = (config["vocab_size"] as? Int) ?? 0
      return vocabSize >= 128_000 ? .llama3 : nil
    }

    return typePrefixes
      .filter { type.hasPrefix($0.prefix) }
      .max(by: { $0.prefix.count < $1.prefix.count })?
      .format
  }

  /// Three-signal inference. Tries the package-level name-prefix table
  /// (with longest-prefix tiebreak) first, then falls back to type-based
  /// resolution. Returns nil when neither resolves; ``ResponseFormat/json``
  /// is used as the default by parser-construction call sites.
  public static func infer(
    modelName: String,
    modelType: String,
    modelConfig: [String: any Sendable],
  ) -> ResponseFormat? {
    if !modelName.isEmpty {
      let name = normalize(modelName)
      if name.hasPrefix("ernie-4.5") || name.hasPrefix("ernie-4_5"),
         name.contains("thinking")
      {
        return .ernieThinking
      }
      let isOlmo3Name =
        name.hasPrefix("olmo-3") || name.hasPrefix("olmo3")
          || name.hasPrefix("allenai-olmo-3") || name.hasPrefix("allenai-olmo3")
      if isOlmo3Name, name.contains("think") {
        return .olmo3Thinking
      }
      let nameMatch = namePrefixes
        .filter { name.hasPrefix($0.prefix) }
        .max(by: { $0.prefix.count < $1.prefix.count })
      if let nameMatch {
        return nameMatch.format
      }
    }
    return resolveByType(modelType, config: modelConfig)
  }
}

extension ResponseFormat {
  /// Construct a parser for this format.
  ///
  /// The `tokenizer` argument is held by the protocol for API symmetry with
  /// `swift-tokenizers.Tokenizer` and to give parsers a place to look up
  /// special-token IDs at construction. In practice every per-format parser
  /// matches markers on detokenized text – the same approach SGLang's
  /// reference parsers take. The detokenized form of every marker we care
  /// about is unique and unambiguous (Hermes' `<think>`, Gemma 4's
  /// `<|channel>thought` / `<channel|>`, Harmony's seven `<|...|>` reserved
  /// tokens, the CJK-bracketed DeepSeek tokens), so text matching is
  /// equivalent to token-ID matching for these formats. The `tokenizer`
  /// parameter remains in the API in case a future format genuinely
  /// requires token-ID introspection (e.g., a model whose marker text
  /// overlaps with regular content the way Harmony's design rationale
  /// envisions).
  ///
  /// `tools` is held for the parsers that need it for argument-type
  /// coercion: Qwen3-Xml, MiniMax M2, and GLM 4 read parameter types out of
  /// the schema to coerce raw string values; every other parser ignores it.
  ///
  /// Pass `priorOutput` when parsing starts after already-rendered text,
  /// such as a continuation chunk or a generated suffix whose rendered
  /// prompt opened a parser region. The factory scans that preceding text
  /// for an unclosed reasoning marker and sets the parser's initial state
  /// accordingly. Delimited marker families use
  /// ``DelimitedReasoningBoundary``; families with implicit reasoning
  /// preambles use ``ImplicitReasoningPreamble``. Mid-tool-call
  /// continuation is deliberately not supported.
  package func makeParser(
    tokenizer _: any ParserTokenizer,
    tools: [ToolSpec] = [],
    priorOutput: String? = nil,
  ) -> any ResponseFormatParser {
    switch self {
      case .hermes:
        HermesParser()
      case .qwen:
        QwenParser(
          initialState: Self.thinkEndedByToolCallBoundary.isOpen(in: priorOutput)
            ? .reasoning
            : .normal,
        )
      case .qwen3Xml:
        Qwen3XmlParser(
          initialState: Self.thinkEndedByToolCallBoundary.isOpen(in: priorOutput)
            ? .reasoning
            : .normal,
          tools: tools,
        )
      case .deepseekR1:
        DeepSeekR1Parser(
          initialState: Self.thinkPreamble.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
        )
      case .deepseekV3:
        DeepSeekV3Parser()
      case .deepseekV31:
        DeepSeekV31Parser()
      case .kimiK2:
        KimiK2Parser()
      case .kimiK2Thinking:
        KimiK2Parser(
          initialState: Self.kimiK2ThinkingPreamble.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
        )
      case .harmony:
        HarmonyParser(initialState: harmonyHasUnclosedReasoning(priorOutput) ? .inReasoning : .idle)
      case .gemma4:
        Gemma4Parser(
          initialState: Self.gemma4ThoughtBoundary.isOpen(in: priorOutput) ? .reasoning : .normal,
        )
      case .miniMaxM2:
        MiniMaxM2Parser(
          initialState: Self.thinkPreamble.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
          tools: tools,
        )
      case .miniMax:
        MiniMaxParser()
      case .glm4:
        Glm4Parser(tools: tools)
      case .glm4Thinking:
        Glm4Parser(
          tools: tools,
          acceptThink: true,
          initialState: Self.thinkPreambleEndedByToolCall.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
        )
      case .longcat:
        HermesParser(toolCallStart: "<longcat_tool_call>", toolCallEnd: "</longcat_tool_call>")
      case .granite:
        GraniteParser(initialState: graniteInitialState(priorOutput: priorOutput))
      case .granite20bFc:
        Granite20bFcParser()
      case .granite4:
        HermesParser(argumentsMayBeJSONString: true)
      case .internlm:
        InternlmParser()
      case .jamba:
        JambaParser()
      case .hunyuanA13B:
        HunyuanA13BParser(
          initialState: Self.thinkBoundary.isOpen(in: priorOutput) ? .reasoning : .normal,
        )
      case .magistral:
        MistralParser(
          acceptThink: true,
          initialState: Self.magistralThinkBoundary.isOpen(in: priorOutput) ? .reasoning : .normal,
        )
      case .deepseekV32:
        DeepSeekV32Parser()
      case .mistral:
        MistralParser()
      case .llama3:
        Llama3Parser()
      case .pythonic:
        PythonicParser()
      case .lfm2:
        PythonicParser(
          startTag: "<|tool_call_start|>",
          endTag: "<|tool_call_end|>",
          acceptJSON: true,
          requiresWrapper: true,
          acceptBarePythonicCall: true,
        )
      case .olmo3:
        PythonicParser(
          startTag: "<function_calls>",
          endTag: "</function_calls>",
          newlineSeparated: true,
        )
      case .olmo3Thinking:
        PythonicParser(
          startTag: "<function_calls>",
          endTag: "</function_calls>",
          newlineSeparated: true,
          acceptThink: true,
          initialState: Self.thinkPreamble.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
        )
      case .phi4Mini:
        Phi4MiniParser()
      case .phiReasoning:
        PhiReasoningParser(
          initialState: Self.thinkBoundary.isOpen(in: priorOutput) ? .reasoning : .normal,
        )
      case .gemmaFunctionCall:
        GemmaFunctionCallParser()
      case .xlam:
        XlamParser()
      case .seedOss:
        Qwen3XmlParser(
          initialState: Self.seedOssPreamble.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
          tools: tools,
          thinkStart: "<seed:think>",
          thinkEnd: "</seed:think>",
          toolCallStart: "<seed:tool_call>",
          toolCallEnd: "</seed:tool_call>",
        )
      case .step3p5:
        Qwen3XmlParser(
          initialState: Self.thinkPreamble.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
          tools: tools,
          trimNewlineAroundThinkEnd: true,
        )
      case .ernie:
        ErnieParser()
      case .ernieThinking:
        ErnieParser(
          acceptThink: true,
          initialState: Self.thinkPreamble.startsInReasoning(after: priorOutput)
            ? .reasoning
            : .normal,
        )
      case .json:
        JSONFallbackParser()
    }
  }

  private static let thinkBoundary = DelimitedReasoningBoundary.think()
  private static let thinkEndedByToolCallBoundary =
    DelimitedReasoningBoundary.think(implicitEndTokens: ["<tool_call>"])
  private static let gemma4ThoughtBoundary =
    DelimitedReasoningBoundary(start: "<|channel>thought", end: "<channel|>")
  private static let magistralThinkBoundary =
    DelimitedReasoningBoundary(start: "[THINK]", end: "[/THINK]")
  private static let thinkPreamble = ImplicitReasoningPreamble.think()
  private static let thinkPreambleEndedByToolCall =
    ImplicitReasoningPreamble.think(implicitEndTokens: ["<tool_call>"])
  private static let kimiK2ThinkingPreamble =
    ImplicitReasoningPreamble.think(implicitEndTokens: ["<|tool_calls_section_begin|>"])
  private static let seedOssPreamble =
    ImplicitReasoningPreamble(endTokens: ["</seed:think>"])

  /// Returns true when `priorOutput` ended inside an unclosed Harmony
  /// `analysis` block – the only Harmony state we resume from per
  /// decision #7's "no mid-tool-call continuation" stance. Finds the last
  /// `<|channel|>` opener, checks that its header begins with `analysis`
  /// (so this isn't a `commentary` or `final` block), confirms the block
  /// has reached `<|message|>` (meaning we're past the header into
  /// content), and returns true only if no terminator (`<|end|>`,
  /// `<|call|>`, `<|return|>`, `<|start|>`) appears after that
  /// `<|message|>`.
  private func harmonyHasUnclosedReasoning(_ priorOutput: String?) -> Bool {
    guard let priorOutput, !priorOutput.isEmpty else { return false }
    guard let lastChannelRange = priorOutput.range(of: "<|channel|>", options: .backwards) else {
      return false
    }
    let afterChannel = priorOutput[lastChannelRange.upperBound...]
    let trimmed = afterChannel.drop(while: { $0.isWhitespace })
    guard trimmed.lowercased().hasPrefix("analysis") else { return false }
    guard let messageRange = priorOutput.range(
      of: "<|message|>",
      range: lastChannelRange.upperBound ..< priorOutput.endIndex,
    ) else {
      return false
    }
    let afterMessage = priorOutput[messageRange.upperBound...]
    for terminator in ["<|end|>", "<|call|>", "<|return|>", "<|start|>"] {
      if afterMessage.range(of: terminator) != nil { return false }
    }
    return true
  }

  /// Granite reasoning is opt-in by the model output: a fresh response
  /// always starts in `preReasoning` (auto-detect from prefix). For
  /// continuation requests, prior output that contains
  /// `Here is my response:` has already exited reasoning, so the new
  /// parser starts in normal phase. Prior output that contains
  /// `Here is my thought process:` (or `Here's my thought process:`)
  /// without a matching response-start marker means we're mid-reasoning;
  /// the parser starts in reasoning phase. Otherwise default to normal.
  private func graniteInitialState(priorOutput: String?) -> GraniteParser.InitialState {
    guard let priorOutput, !priorOutput.isEmpty else { return .normal }
    let thinkStarts = ["Here is my thought process:", "Here's my thought process:"]
    let responseStarts = ["Here is my response:", "Here's my response:"]
    let hasThink = thinkStarts.contains { priorOutput.range(of: $0) != nil }
    let hasResp = responseStarts.contains { priorOutput.range(of: $0) != nil }
    if hasThink, !hasResp { return .reasoning }
    return .normal
  }
}

// MARK: Stop-token policy

/// Per-format declaration of how stop tokens reach the parser. Read by
/// integration layers (such as the MLX bridge) so format-specific token
/// lists aren't hardcoded at the call site.
///
/// **The parser-library contract.** ``includedStopTokens`` lists the halt
/// tokens that the parser expects to see in the token stream yielded by
/// the inference engine. For the parser to actually receive them, the
/// integration must (a) configure its inference engine to halt on those
/// tokens, (b) include the halt token in the yielded stream rather than
/// dropping it after halting, and (c) decode it as text rather than
/// stripping it as a special. The MLX bridge handles all three for
/// shipped formats; an alternative integration (llama.cpp, etc.) reads
/// ``includedStopTokens`` to drive the same setup.
public struct ResponseFormatStopTokenPolicy: Sendable, Equatable {
  /// Stop tokens that the inference engine must include in the token
  /// stream it yields to the parser, rather than dropping them after
  /// halting. For Harmony, `<|call|>` and `<|return|>`. Empty for formats
  /// whose parsers only operate on text that arrived before the stop
  /// token.
  public var includedStopTokens: Set<String>

  /// Tokens the inference engine must halt on. Independent of
  /// ``includedStopTokens`` in principle — a format could halt on a token
  /// the parser doesn't observe, or observe a token that some other
  /// engine source already halts on — but for shipped formats the two
  /// sets coincide. Consumed by the bridge to auto-inject into
  /// `ModelConfiguration.extraEOSTokens`. Mirrors the `entrypoints/`-internal
  /// placement of `get_stop_tokens_for_assistant_actions` in vLLM and
  /// SGLang — engine-side plumbing, not a parser-library contract
  /// callers need to read.
  package var requiredExtraEOSTokens: Set<String>

  /// Whether the inference engine must include any of its stop tokens in
  /// the yielded stream. Derived: true iff ``includedStopTokens`` is
  /// non-empty.
  public var includeStopToken: Bool {
    !includedStopTokens.isEmpty
  }

  package init(includedStopTokens: Set<String>, requiredExtraEOSTokens: Set<String>) {
    self.includedStopTokens = includedStopTokens
    self.requiredExtraEOSTokens = requiredExtraEOSTokens
  }
}

public extension ResponseFormat {
  /// Per-format stop-token policy. Integration layers read this to
  /// configure the inference engine – they should not hardcode
  /// format-specific token lists.
  ///
  /// Mid-message Harmony tokens (`<|start|>`, `<|channel|>`, `<|message|>`,
  /// `<|constrain|>`, `<|end|>`) are not stop tokens – they flow through
  /// the normal token stream regardless and don't appear here. This is
  /// only about *stop* tokens that need an include/halt policy.
  var stopTokenPolicy: ResponseFormatStopTokenPolicy {
    switch self {
      case .harmony:
        ResponseFormatStopTokenPolicy(
          includedStopTokens: ["<|call|>", "<|return|>"],
          requiredExtraEOSTokens: ["<|call|>", "<|return|>"],
        )
      case .hermes, .qwen, .qwen3Xml, .deepseekR1, .deepseekV3, .deepseekV31,
           .deepseekV32, .mistral, .llama3, .pythonic, .lfm2, .olmo3,
           .olmo3Thinking, .phi4Mini,
           .phiReasoning, .gemmaFunctionCall, .gemma4, .kimiK2, .kimiK2Thinking,
           .miniMaxM2, .miniMax, .glm4, .glm4Thinking, .longcat, .granite, .granite20bFc, .granite4, .internlm, .jamba,
           .hunyuanA13B, .magistral, .xlam, .seedOss, .step3p5, .ernie, .ernieThinking, .json:
        ResponseFormatStopTokenPolicy(
          includedStopTokens: [],
          requiredExtraEOSTokens: [],
        )
    }
  }
}
