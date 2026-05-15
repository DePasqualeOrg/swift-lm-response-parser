# MLX Integration

Stream Open Responses from an MLX Swift LM model.

## Overview

`LMResponsesMLX` offers two layers of integration:

- ``ResponseChatSession``: multi-turn session with built-in tool dispatch, mirrors `MLXLMCommon.ChatSession` but yields ``/LMResponses/ResponseOutputItem`` snapshots or ``/LMResponses/ResponseStreamingEvent`` values instead of plain strings.
- ``streamResponseEvents(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)`` and ``streamResponseItems(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``: lower-level helpers for callers that own cache lifecycle directly.

## ResponseChatSession

One session call corresponds to one assistant turn and produces one ``/LMResponses/Response``, regardless of how many internal generation passes happen for tool dispatch.

```swift
import LMResponsesMLX
import MLXLMCommon

let modelContainer = try await loadModelContainer(
    from: downloader,
    using: tokenizerLoader,
    configuration: ModelConfiguration(id: "mlx-community/gpt-oss-20b-MXFP4-Q8")
)

let session = ResponseChatSession(
    modelContainer,
    modelType: "gpt_oss",
    modelConfig: configDict,
    tools: tools,
    toolDispatch: { call in
        let args = try call.decodedArguments(as: WeatherArgs.self)
        return .string(try await getWeather(city: args.city))
    }
)

for try await items in session.streamResponseItems(prompt: "What's the weather in Paris?") {
    updateUI(items)
}
```

`streamResponseEvents(prompt:)` is also available for the SSE-proxy/[Open Responses](https://www.openresponses.org/) case, with the same lifecycle.

## Tool validation and recovery

``ResponseChatSession`` owns the restart loop for tool use. When the model emits a completed tool call and `toolDispatch` is set, the session runs the callback, appends the returned ``/LMResponses/ResponseFunctionCallOutput/Output`` as a function-call output, and starts the next internal generation pass.

The callback is the validation boundary. Check the function name, decode and validate arguments, enforce permissions, and handle runtime preconditions before executing the tool:

```swift
let session = ResponseChatSession(
    modelContainer,
    modelType: "gpt_oss",
    modelConfig: configDict,
    tools: tools,
    toolDispatch: { call in
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
)
```

Returning an error string gives the model feedback it can use to correct a bad tool call on the next pass. Return JSON structured results as ``/LMResponses/ResponseFunctionCallOutput/Output/string(_:)`` containing JSON text. Return typed text, image, or file results as ``/LMResponses/ResponseFunctionCallOutput/Output/content(_:)``.

The emitted Responses events preserve typed content outputs. For the next local MLX generation pass, ``ResponseChatSession`` renders typed text/image/file parts into the tool message text and attaches `input_image.image_url` values as `UserInput.Image.url` when they can be represented as URLs. Hosted file IDs and file contents require application-specific handling. Throwing from `toolDispatch` fails the response stream and should be reserved for failures where continuing the turn would be wrong, such as cancellation or an internal application error.

## Low-level helpers

The streaming helpers come in two forms:

- **`ModelContainer` extension**: stateless one-shot, no cache control. The container's `perform { … }` boundary and the non-`Sendable` `ModelContext` are managed for you.
- **Free function**: takes a bare `ModelContext` and `LMInput`. Use this from inside `perform { … }` when you want to pass a `cache:` and manage its lifecycle.

```swift
import LMResponsesMLX

let input = try await modelContainer.prepare(input: userInput)
let stream = try await modelContainer.streamResponseItems(
    input: input,
    parameters: generateParameters,
    modelType: "qwen3",
    modelConfig: configDict,
    config: ResponseStreamConfig(
        model: await modelContainer.configuration.name,
        tools: tools
    )
)

for await items in stream {
    updateUI(items)
}

if let response = await stream.finalResponse() {
    print("input \(response.usage?.inputTokens ?? 0), output \(response.usage?.outputTokens ?? 0)")
}
```

Use ``streamResponseEvents(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)`` instead when you want each ``/LMResponses/ResponseStreamingEvent`` directly – useful for SSE serialization or proxying.
