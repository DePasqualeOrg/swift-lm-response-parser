// Copyright © Anthony DePasquale

import Foundation
import LMResponses
@testable import LMResponsesLlama
import Testing

@Suite("MessageBuilder")
struct MessageBuilderTests {
  // MARK: assistantMessage

  @Test func `message with only text produces plain assistant content`() {
    let items: [ResponseOutputItem] = [
      .message(.init(
        id: "msg_1",
        content: [.outputText(.init(text: "Hello, world."))],
      )),
    ]
    let result = MessageBuilder.assistantMessage(from: items, alreadyDispatched: [])

    #expect(result["role"] as? String == "assistant")
    #expect(result["content"] as? String == "Hello, world.")
    // No tool_calls when nothing was called.
    #expect(result["tool_calls"] == nil)
  }

  @Test func `multiple text parts are concatenated`() {
    let items: [ResponseOutputItem] = [
      .message(.init(
        id: "msg_1",
        content: [
          .outputText(.init(text: "Part one. ")),
          .outputText(.init(text: "Part two.")),
        ],
      )),
    ]
    let result = MessageBuilder.assistantMessage(from: items, alreadyDispatched: [])
    #expect(result["content"] as? String == "Part one. Part two.")
  }

  @Test func `function call adds tool calls entry`() {
    let items: [ResponseOutputItem] = [
      .message(.init(
        id: "msg_1",
        content: [.outputText(.init(text: "Calling now."))],
      )),
      .functionCall(.init(
        id: "fc_1",
        callId: "call_abc",
        name: "get_weather",
        arguments: "{\"city\":\"Paris\"}",
        status: .completed,
      )),
    ]
    let result = MessageBuilder.assistantMessage(from: items, alreadyDispatched: [])

    #expect(result["content"] as? String == "Calling now.")
    let toolCalls = result["tool_calls"] as? [[String: any Sendable]]
    #expect(toolCalls?.count == 1)
    #expect(toolCalls?.first?["id"] as? String == "call_abc")
    #expect(toolCalls?.first?["type"] as? String == "function")
    let function = toolCalls?.first?["function"] as? [String: any Sendable]
    #expect(function?["name"] as? String == "get_weather")
    #expect(function?["arguments"] as? String == "{\"city\":\"Paris\"}")
  }

  @Test func `already dispatched function calls are filtered out`() {
    let items: [ResponseOutputItem] = [
      .functionCall(.init(
        id: "fc_1",
        callId: "call_old",
        name: "fn_a",
        arguments: "{}",
        status: .completed,
      )),
      .functionCall(.init(
        id: "fc_2",
        callId: "call_new",
        name: "fn_b",
        arguments: "{}",
        status: .completed,
      )),
    ]
    let result = MessageBuilder.assistantMessage(
      from: items,
      alreadyDispatched: ["call_old"],
    )
    let toolCalls = result["tool_calls"] as? [[String: any Sendable]]
    #expect(toolCalls?.count == 1)
    #expect(toolCalls?.first?["id"] as? String == "call_new")
  }

  @Test func `reasoning and function call output items are dropped`() {
    let items: [ResponseOutputItem] = [
      .reasoning(.init(id: "rsn_1")),
      .message(.init(
        id: "msg_1",
        content: [.outputText(.init(text: "Body."))],
      )),
      .functionCallOutput(.init(
        id: "fco_1",
        callId: "call_x",
        output: .string("result"),
      )),
    ]
    let result = MessageBuilder.assistantMessage(from: items, alreadyDispatched: [])
    #expect(result["content"] as? String == "Body.")
    #expect(result["tool_calls"] == nil)
  }

  // MARK: toolMessage

  @Test func `tool message carries string result`() {
    let call = ResponseFunctionToolCall(
      id: "fc_1",
      callId: "call_42",
      name: "get_time",
      arguments: "{}",
      status: .completed,
    )
    let result = MessageBuilder.toolMessage(for: call, result: .string("12:00"))

    #expect(result["role"] as? String == "tool")
    #expect(result["tool_call_id"] as? String == "call_42")
    #expect(result["name"] as? String == "get_time")
    #expect(result["content"] as? String == "12:00")
  }

  @Test func `tool message handles content parts result`() {
    let call = ResponseFunctionToolCall(
      id: "fc_1",
      callId: "call_42",
      name: "lookup",
      arguments: "{}",
      status: .completed,
    )
    let result = MessageBuilder.toolMessage(
      for: call,
      result: .content([.inputText(.init(text: "found it"))]),
    )
    #expect(result["content"] as? String == "found it")
  }
}
