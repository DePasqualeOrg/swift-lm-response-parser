# ``LMResponses``

Turn language model output into Open Responses streaming events.

## Overview

`LMResponses` is the engine-agnostic core of the package. It parses detokenized model text directly, or consumes token IDs through ``ResponseStream`` and detokenizes them before handing text plus aligned optional token metadata to a per-format detector (Hermes, Harmony, Qwen, DeepSeek, and many others). It emits the streaming events and accumulated output items defined by the [Open Responses](https://www.openresponses.org/) shape.

The companion module [`LMResponsesMLX`](/documentation/lmresponsesmlx) wires this engine to MLX Swift LM, exposing a multi-turn chat session that yields Responses-shaped values directly.

## Parser and validation contract

Parsers perform best-effort structural extraction from model text. They recognize model-specific delimiters, stream messages, reasoning, and tool-call arguments, and close truncated structures with ``ItemStatus/incomplete`` where the format gives enough information to preserve the partial item. They also normalize wire-format details, such as converting a model's untyped parameter text into a JSON argument string when a format requires schema-aware coercion.

Semantic validation is intentionally outside the parser core. A parser does not decide whether a tool name is registered, whether arguments are acceptable for your application, whether the user may run the tool, or whether a tool should execute. Treat completed ``ResponseOutputItem/functionCall(_:)`` items as candidate calls, then validate names, decoded arguments, permissions, and runtime preconditions in your dispatch layer.

When the model makes a recoverable tool-call mistake, prefer returning an error as tool output so the next generation pass can see it and recover. Use thrown errors for control-plane failures that should abort the response stream.

## Topics

### Guides

- <doc:getting-started>
- <doc:streaming>

### MLX integration

- [`LMResponsesMLX`](/documentation/lmresponsesmlx)

### Streaming

- ``ResponseStream``
- ``ResponseStreamConfig``

### One-shot parsing

- ``parseResponse(_:format:tokenizer:tools:)``
- ``accumulateItems(from:)``

### Output items

- ``ResponseOutputItem``
- ``ResponseOutputMessage``
- ``ResponseFunctionToolCall``
- ``ResponseReasoningItem``
- ``ResponseFunctionCallOutput``
- ``ItemStatus``
- ``ResponseOutputText``
- ``ResponseOutputRefusal``
- ``ResponseContentPart``
- ``ReasoningTextContent``

### Response

- ``Response``
- ``ResponseStatus``
- ``ResponseUsage``
- ``IncompleteDetails``
- ``IncompleteReason``
- ``FinishReason``

### Streaming events

- ``ResponseStreamingEvent``
- ``ResponseCreatedEvent``
- ``ResponseInProgressEvent``
- ``ResponseCompletedEvent``
- ``ResponseIncompleteEvent``
- ``ResponseOutputItemAddedEvent``
- ``ResponseOutputItemDoneEvent``
- ``ResponseContentPartAddedEvent``
- ``ResponseContentPartDoneEvent``
- ``ResponseTextDeltaEvent``
- ``ResponseTextDoneEvent``
- ``ResponseReasoningDeltaEvent``
- ``ResponseReasoningDoneEvent``
- ``ResponseFunctionCallArgumentsDeltaEvent``
- ``ResponseFunctionCallArgumentsDoneEvent``

### Formats

- ``ResponseFormat``
- ``ResponseFormatStopTokenPolicy``
- ``ToolSpec``

### Tokenizer

- ``ResponseTokenizer``
