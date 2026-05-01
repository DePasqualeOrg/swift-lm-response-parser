# Streaming

Drive a ``ResponseStream`` from a token loop and forward items or events.

## Overview

``ResponseStream`` is the engine-agnostic streaming entry point. Configure it with a ``ResponseFormat``, a ``ParserTokenizer``, and the tools the model has been told about, then feed it tokens one at a time. After each token it exposes both the latest accumulated items and the streaming events the token produced.

For MLX, the higher-level conveniences in the [`LMResponseParserMLX`](/documentation/lmresponseparsermlx) module wrap this loop so callers don't manage tokens themselves.

## Setup

```swift
import LMResponseParser

let format = ResponseFormat.infer(
    modelName: "Qwen/Qwen3-Coder-30B",
    modelType: "qwen3_5",
    modelConfig: configDict
) ?? .json

let stream = ResponseStream(
    format: format,
    config: ResponseStreamConfig(model: "Qwen/Qwen3-Coder-30B"),
    tokenizer: tokenizer,
    tools: tools
)
```

## Forwarding items

``ResponseStream/items`` is the current `[ResponseOutputItem]` snapshot, updated as tokens are processed. Use this when you want to render assistant state directly:

```swift
stream.start()

while let token = await model.nextToken() {
    stream.process(tokenId: token)
    updateUI(stream.items)
}

stream.finalize(finishReason: .stop, inputTokens: promptTokenCount)
```

Function-call items in ``ResponseStream/items`` are candidates for dispatch, not proof that a tool should run. Wait for ``ItemStatus/completed`` before dispatching a call, then validate the name, decoded arguments, permissions, and runtime preconditions in your own dispatch layer. Treat ``ItemStatus/incomplete`` as model output that could not be completed, not as a call to execute.

The `tools` passed to ``ResponseStream`` help parsers understand the model's wire format. Some formats emit untyped key/value pairs, so the parser may consult JSON-schema parameter types to produce a valid JSON `arguments` string. That coercion is not semantic validation; application policy still belongs outside the parser.

## Forwarding events

To proxy to an [Open Responses](https://www.openresponses.org/) consumer or serialize to SSE, iterate the events returned by each method:

```swift
for event in stream.start() {
    forward(event)
}

while let token = await model.nextToken() {
    for event in stream.process(tokenId: token) {
        forward(event)
    }
}

for event in stream.finalize(finishReason: .stop, inputTokens: promptTokenCount) {
    forward(event)
}
```

## Terminal Response

After ``ResponseStream/finalize(finishReason:inputTokens:cachedInputTokens:reasoningOutputTokens:)`` returns, ``ResponseStream/finalResponse`` carries the terminal ``Response`` envelope with usage, status, and incomplete details. The same data is also available on the `responseCompleted` event for event-forwarding consumers.
