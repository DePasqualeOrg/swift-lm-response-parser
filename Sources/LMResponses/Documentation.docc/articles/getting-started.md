# Getting Started

Configure a tokenizer, choose a response format, and start parsing.

## Overview

`LMResponses` parses detokenized model text, or consumes token IDs through ``ResponseStream`` and detokenizes them before parsing. It produces [Open Responses](https://www.openresponses.org/) streaming events plus an accumulated `[ResponseOutputItem]` snapshot. The parser is engine-agnostic; token-driven streaming only needs a tokenizer that conforms to ``ResponseTokenizer``.

For MLX Swift LM, the [`LMResponsesMLX`](/documentation/lmresponsesmlx) module wires this engine to `ModelContainer` and `ModelContext` so callers don't manage tokenization manually.

## Tokenizer conformance

`ResponseTokenizer` declares three methods (`convertTokenToId`, `encode`, `decode`) whose signatures match `MLXLMCommon.Tokenizer` exactly. Any conforming concrete type already satisfies `ResponseTokenizer`'s requirements; conform with a single empty extension:

```swift
import LMResponses

extension MyTokenizer: ResponseTokenizer {}
```

## Output items

``ResponseOutputItem`` is a structured view of an assistant turn that's stable to render in a UI, pattern-match on for tool dispatch, or persist for replay. The enum has four cases:

- ``ResponseOutputItem/message(_:)``: assistant text with optional refusal parts. Use `.text` for the joined text of all `outputText` parts.
- ``ResponseOutputItem/functionCall(_:)``: a tool invocation. Carries `name`, `callId`, and `arguments` as a JSON string. Use ``ResponseFunctionToolCall/decodedArguments(as:decoder:)`` to decode into a `Decodable` type.
- ``ResponseOutputItem/reasoning(_:)``: chain of thought from reasoning models. Use `.text` for the joined reasoning text.
- ``ResponseOutputItem/functionCallOutput(_:)``: the result of a tool call, paired to its `functionCall` by `callId`. The output can be a string or typed text/image/file content via ``ResponseFunctionCallOutput/Output``.

Each item carries an ``ItemStatus`` (`.inProgress`, `.completed`, `.incomplete`) so consumers can filter out partial items.

```swift
for item in items {
    switch item {
        case .message(let m):
            print(m.text)
        case .functionCall(let f):
            let args = try f.decodedArguments(as: WeatherArgs.self)
            // dispatch...
        case .reasoning(let r):
            print(r.text)
        case .functionCallOutput(let o):
            if case .string(let text) = o.output {
                print(text)
            }
    }
}
```

## Validation and recovery

Parsing and dispatch have separate responsibilities. Parsers make a best-effort attempt to turn model text into structured items. They may repair or normalize model-specific wire formats enough to produce valid JSON argument strings, but they do not decide whether a tool call is allowed to run.

Dispatch code should validate completed tool calls before executing them:

```swift
func dispatch(_ call: ResponseFunctionToolCall) async -> ResponseFunctionCallOutput.Output {
    guard call.status == .completed else {
        return .string("Error: tool call was incomplete; please call the tool again with complete arguments.")
    }

    guard call.name == "get_weather" else {
        return .string("Error: unknown tool '\(call.name)'. Available tools: get_weather.")
    }

    do {
        let args = try call.decodedArguments(as: WeatherArgs.self)
        return .string(try await getWeather(city: args.city))
    } catch {
        return .string("Error: invalid arguments for get_weather: \(error)")
    }
}
```

Returning an error string as tool output gives the model feedback it can use on the next pass. Return JSON structured results as ``ResponseFunctionCallOutput/Output/string(_:)`` containing JSON text. Use ``ResponseFunctionCallOutput/Output/content(_:)`` for typed text, image, or file outputs. Throwing is still appropriate for failures that should stop the response entirely, such as cancellation, lost network access, or an internal application invariant failure.

## One-shot parsing

For non-streaming use cases, the ``parseResponse(_:format:tokenizer:tools:)`` helper returns the accumulated `[ResponseOutputItem]`:

```swift
import LMResponses

let items = parseResponse(
    "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
    format: .hermes,
    tokenizer: tokenizer
)
// items contains one .functionCall with name "get_weather"
```

## Next steps

- <doc:streaming>: drive a ``ResponseStream`` from a token loop and forward items or events.
- [MLX integration](/documentation/lmresponsesmlx): high-level chat sessions on top of MLX Swift LM.
