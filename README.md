# Swift LM Response Parser

[![Swift Version Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FDePasqualeOrg%2Fswift-lm-response-parser%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/DePasqualeOrg/swift-lm-response-parser) [![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FDePasqualeOrg%2Fswift-lm-response-parser%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/DePasqualeOrg/swift-lm-response-parser)

> Note: This library is in early development. Expect breaking changes.

Swift LM Response Parser turns language model output into [Open Responses](https://www.openresponses.org/) streaming events.

- [Documentation](https://swiftpackageindex.com/DePasqualeOrg/swift-lm-response-parser/main/documentation/lmresponseparser)
- [Getting Started](https://swiftpackageindex.com/DePasqualeOrg/swift-lm-response-parser/main/documentation/lmresponseparser/getting-started)
- [Streaming](https://swiftpackageindex.com/DePasqualeOrg/swift-lm-response-parser/main/documentation/lmresponseparser/streaming)
- [MLX Integration](https://swiftpackageindex.com/DePasqualeOrg/swift-lm-response-parser/main/documentation/lmresponseparsermlx/mlx-session)

## Installation

Add the package as a dependency in your `Package.swift`:

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-lm-response-parser", from: "0.1.1")
```

Then add the library you need to your target. Parser-only consumers depend on `LMResponseParser`:

```swift
.target(name: "MyApp", dependencies: [
    .product(name: "LMResponseParser", package: "swift-lm-response-parser"),
])
```

MLX consumers depend on `LMResponseParserMLX`:

```swift
.target(name: "MyApp", dependencies: [
    .product(name: "LMResponseParserMLX", package: "swift-lm-response-parser"),
])
```

## Supported formats

| Model family | Models |
|-------------|--------|
| Hermes | Nous Hermes, Hermes-2 |
| Qwen base | Qwen 2.5, Qwen 3 base, Qwen 3 MoE, Qwen2.5-VL, Qwen3-VL |
| Qwen 3 Coder/3.5 | Qwen 3 Coder, Qwen 3.5 |
| DeepSeek R1 | DeepSeek-R1 family |
| Mistral/Mixtral | Mistral, Mixtral, Mistral 3 |
| Llama 3 | Llama 3/3.1/3.2 |
| Llama 4 | Llama 4 |
| OLMo 3 | Allen AI OLMo 3, OLMo 3 Think |
| Kimi K2 | Kimi-K2-Instruct, Kimi-K2-Thinking |
| GPT-OSS/Harmony | GPT-OSS |
| Gemma 4 | Gemma 4 |
| DeepSeek V3 | DeepSeek V3 base, V3.1, V3.2-Exp, V3.2 |
| MiniMax | MiniMax M2, MiniMax-Text-01, MiniMax-M1 |
| GLM 4.x/5 | GLM 4, GLM 4.5+, GLM 5 |
| LongCat | Meituan LongCat-Flash |
| Granite | IBM Granite-20B-FunctionCalling, Granite 3.0, 3.1, 3.2, Granite 4.0 |
| InternLM | Shanghai AI Lab InternLM 2.x, Intern-S1 |
| Jamba | AI21 Jamba 1.5, 1.7 |
| Hunyuan A13B | Tencent Hunyuan A13B Instruct/Pretrain |
| Magistral | Magistral-Small 2506/1.1/2509 |
| LFM2 | LiquidAI LFM2, LFM2-MoE, LFM2-VL |
| Phi | Phi-4-mini-instruct, Phi-4-reasoning, Phi-4-reasoning-plus, Phi-4-mini-reasoning |
| FunctionGemma | Gemma 1/2, FunctionGemma |
| xLAM | Salesforce xLAM 1B/2/8B/32B/70B |
| Seed-OSS | Seed-OSS-36B-Instruct |
| Step-3.5-Flash | Step-3.5-Flash |
| ERNIE 4.5 | ERNIE-4.5, ERNIE-4.5-VL, ERNIE-4.5 Thinking |
| JSON fallback | Any model emitting bare JSON tool calls or without a native tool-call format |

## Planned improvements

The library is in early development. Planned changes include:

- **Data-driven parser routing registry.** The current routing logic in `ResponseFormat.infer(...)` is hardcoded. A future revision will expose the routing rules as data so consumers can augment or override routing without forking, and so a deployed app can pick up support for newly released models without rebuilding.
- **Custom parser registration.** The set of supported parsers is closed today – adding a new wire format requires modifying the library. A future revision will let consumers register their own parser implementations against the routing layer, so applications can support proprietary or experimental formats without forking.
- **Token-ID-aware marker hardening.** The library was designed from the beginning with optional token-ID plumbing: `ParserInput` can carry aligned token IDs, `ResponseStream` and the MLX bridge already preserve them, and `ParserTokenizer` exposes marker-ID lookup. Current parsers intentionally use SGLang-style text matching, but a future revision can use token IDs for targeted formats where reserved marker strings need to be distinguished from ordinary text that decodes the same way.

## Acknowledgements

The parsers and tests are based on prior art in [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), and [OpenAI's Harmony parser](https://github.com/openai/harmony). The events-to-items accumulation drew on [OpenAI's TypeScript SDK](https://github.com/openai/openai-node).
