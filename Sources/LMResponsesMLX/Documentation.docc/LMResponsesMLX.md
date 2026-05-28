# ``LMResponsesMLX``

Bridge `LMResponses` to MLX Swift LM.

## Overview

`LMResponsesMLX` wires the engine-agnostic core in [`LMResponses`](/documentation/lmresponses) to `MLXLMCommon.ModelContainer` and `ModelContext`. It exposes a multi-turn chat session that mirrors `MLXLMCommon.ChatSession` (yielding `[ResponseOutputItem]` snapshots or `ResponseStreamingEvent` values instead of plain strings) plus low-level streaming helpers for callers that want to manage cache lifecycle directly.

``ResponseChatSession`` owns the multi-pass tool loop, but the host application still owns tool validation. Its `toolDispatch` callback receives completed parser output, validates the tool name and arguments, runs application policy, and returns the tool output that the next generation pass will see. Return recoverable validation failures as ``/LMResponses/ResponseFunctionCallOutput/Output/string(_:)`` values; throw only to abort the turn.

## Topics

### Guides

- <doc:mlx-session>

### Chat sessions

- ``ResponseChatSession``

### Streaming helpers

- ``ResponseStreamHandle``
- ``streamText(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``
- ``streamResponseEvents(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``
- ``streamResponseItems(input:cache:parameters:context:modelType:modelConfig:format:config:priorOutput:wiredMemoryTicket:)``

### Errors

- ``ResponseChatSessionError``
- ``BridgeError``

### Parser core

- [`LMResponses`](/documentation/lmresponses)
