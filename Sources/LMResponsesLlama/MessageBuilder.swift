// Copyright © Anthony DePasquale

import Foundation
import LMResponses
import Tokenizers

/// Chat-template `Message` construction shared by ``ResponseChatSession``
/// and ``MultimodalResponseChatSession``. Both sessions fold the
/// engine's `ResponseOutputItem` stream back into an assistant message
/// (so the next pass's chat template renders the prior turn correctly)
/// and wrap tool-dispatch results into `tool` messages.
enum MessageBuilder {
  /// Build the assistant `Message` to append to history for chat-template
  /// rendering in the next pass. Combines all output_text content into a
  /// single content string and gathers function-call entries into
  /// `tool_calls`. Mirrors the OpenAI Chat Completions assistant message
  /// shape that most modern chat templates accept.
  ///
  /// `.reasoningText` is dropped — reasoning lives in its own history
  /// slot for templates that re-render it, and none of the shipped chat
  /// templates round-trip reasoning content in a way the model expects.
  static func assistantMessage(
    from items: [ResponseOutputItem],
    alreadyDispatched: Set<String>,
  ) -> Tokenizers.Message {
    var content = ""
    var toolCalls: [[String: any Sendable]] = []

    for item in items {
      switch item {
        case let .message(msg):
          for part in msg.content {
            if case let .outputText(text) = part {
              content += text.text
            }
          }
        case let .functionCall(call):
          guard !alreadyDispatched.contains(call.callId) else { continue }
          toolCalls.append([
            "id": call.callId,
            "type": "function",
            "function": [
              "name": call.name,
              "arguments": call.arguments,
            ] as [String: any Sendable],
          ])
        case .reasoning, .functionCallOutput:
          break
      }
    }

    var message: Tokenizers.Message = ["role": "assistant", "content": content]
    if !toolCalls.isEmpty {
      message["tool_calls"] = toolCalls as [any Sendable]
    }
    return message
  }

  /// Build the `tool` message that carries a dispatched tool's result
  /// back into the next chat-template render. Most modern templates key
  /// off `tool_call_id` to associate the result with the original
  /// `tool_calls[].id`; `name` is included for templates that surface
  /// the tool name in the rendered transcript.
  static func toolMessage(
    for call: ResponseFunctionToolCall,
    result: ResponseFunctionCallOutput.Output,
  ) -> Tokenizers.Message {
    [
      "role": "tool",
      "tool_call_id": call.callId,
      "name": call.name,
      "content": result.toolMessageText,
    ]
  }
}
