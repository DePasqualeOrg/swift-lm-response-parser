// Copyright © Anthony DePasquale

import Foundation
import LMResponses
import Testing

@Suite("Codable event wire format")
struct CodableEventsTests {
  @Test
  func `public event graph conforms to Codable`() {
    assertCodable(ResponseStreamingEvent.self)
    assertCodable(ResponseCreatedEvent.self)
    assertCodable(ResponseInProgressEvent.self)
    assertCodable(ResponseCompletedEvent.self)
    assertCodable(ResponseIncompleteEvent.self)
    assertCodable(ResponseOutputItemAddedEvent.self)
    assertCodable(ResponseOutputItemDoneEvent.self)
    assertCodable(ResponseContentPartAddedEvent.self)
    assertCodable(ResponseContentPartDoneEvent.self)
    assertCodable(ResponseTextDeltaEvent.self)
    assertCodable(ResponseTextDoneEvent.self)
    assertCodable(ResponseFunctionCallArgumentsDeltaEvent.self)
    assertCodable(ResponseFunctionCallArgumentsDoneEvent.self)
    assertCodable(ResponseReasoningDeltaEvent.self)
    assertCodable(ResponseReasoningDoneEvent.self)

    assertCodable(Response.self)
    assertCodable(ResponseOutputItem.self)
    assertCodable(ResponseContentPart.self)
    assertCodable(ResponseFunctionCallOutput.Output.self)
    assertCodable(ResponseFunctionCallOutput.Content.self)
  }

  @Test
  func `all streaming event cases round-trip through JSON`() throws {
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    for event in sampleEvents() {
      let data = try encoder.encode(event)
      let decoded = try decoder.decode(ResponseStreamingEvent.self, from: data)
      #expect(decoded == event)
    }
  }

  @Test
  func `direct event payload structs round-trip through JSON`() throws {
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let payload = ResponseFunctionCallArgumentsDoneEvent(
      itemId: "fc_1",
      outputIndex: 2,
      arguments: #"{"query":"swift"}"#,
      sequenceNumber: 9,
    )
    let data = try encoder.encode(payload)
    let decoded = try decoder.decode(ResponseFunctionCallArgumentsDoneEvent.self, from: data)

    #expect(decoded == payload)

    let object = try dictionary(from: data)
    #expect(object["name"] == nil)

    let vllmPayload = Data(
      #"{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":2,"name":"lookup","arguments":"{\"query\":\"swift\"}","sequence_number":9}"#
        .utf8,
    )
    #expect(try decoder.decode(ResponseFunctionCallArgumentsDoneEvent.self, from: vllmPayload) == payload)
  }

  @Test
  func `reasoning events decode vLLM and SGLang legacy type discriminators`() throws {
    let decoder = JSONDecoder()
    let encoder = JSONEncoder()

    let legacyDelta = Data(
      #"{"type":"response.reasoning_text.delta","item_id":"rs_1","output_index":2,"content_index":0,"delta":"think","sequence_number":43}"#
        .utf8,
    )
    let decodedEvent = try decoder.decode(ResponseStreamingEvent.self, from: legacyDelta)
    #expect(decodedEvent == .reasoningDelta(.init(
      itemId: "rs_1",
      outputIndex: 2,
      contentIndex: 0,
      delta: "think",
      sequenceNumber: 43,
    )))

    let decodedPayload = try decoder.decode(ResponseReasoningDeltaEvent.self, from: legacyDelta)
    #expect(decodedPayload.delta == "think")

    let reencoded = try dictionary(from: encoder.encode(decodedEvent))
    #expect(reencoded["type"] as? String == "response.reasoning.delta")
  }

  @Test
  func `image detail cases match Open Responses image detail enum`() {
    #expect(ResponseFunctionCallOutput.InputImage.Detail.allCases.map(\.rawValue) == [
      "low",
      "high",
      "auto",
    ])
  }

  @Test
  func `encoded events use spec type discriminators and snake case keys`() throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]

    let event = ResponseStreamingEvent.responseIncomplete(.init(
      response: sampleResponse(status: .incomplete),
      sequenceNumber: 42,
    ))
    let object = try dictionary(from: encoder.encode(event))

    #expect(object["type"] as? String == "response.incomplete")
    #expect(object["sequence_number"] as? Int == 42)
    #expect(object["sequenceNumber"] == nil)

    let response = try #require(object["response"] as? [String: Any])
    #expect(response["object"] as? String == "response")
    #expect(response["created_at"] as? Int == 1_700_000_000)
    #expect(response["createdAt"] == nil)
    #expect(response["completed_at"] is NSNull)
    #expect(response["status"] as? String == "incomplete")
    #expect(response["previous_response_id"] is NSNull)
    #expect(response["instructions"] as? String == "be concise")
    #expect(response["error"] is NSNull)
    #expect((response["tools"] as? [Any])?.isEmpty == true)
    #expect(response["tool_choice"] as? String == "auto")
    #expect(response["truncation"] as? String == "disabled")
    #expect(response["parallel_tool_calls"] as? Bool == true)
    let text = try #require(response["text"] as? [String: Any])
    let textFormat = try #require(text["format"] as? [String: Any])
    #expect(textFormat["type"] as? String == "text")
    #expect(response["top_p"] as? Double == 0.95)
    #expect(response["presence_penalty"] as? Int == 0)
    #expect(response["frequency_penalty"] as? Int == 0)
    #expect(response["top_logprobs"] as? Int == 0)
    #expect(response["temperature"] as? Double == 0.7)
    let reasoning = try #require(response["reasoning"] as? [String: Any])
    #expect(reasoning["effort"] is NSNull)
    #expect(reasoning["summary"] is NSNull)
    #expect(response["max_output_tokens"] as? Int == 512)
    #expect(response["max_tool_calls"] is NSNull)
    #expect(response["store"] as? Bool == true)
    #expect(response["background"] as? Bool == false)
    #expect(response["service_tier"] as? String == "default")
    #expect((response["metadata"] as? [String: Any])?.isEmpty == true)
    #expect(response["safety_identifier"] is NSNull)
    #expect(response["prompt_cache_key"] is NSNull)
    #expect(response["incomplete_details"] as? [String: Any] != nil)

    let usage = try #require(response["usage"] as? [String: Any])
    #expect(usage["input_tokens"] as? Int == 11)
    #expect(usage["output_tokens"] as? Int == 7)
    #expect(usage["total_tokens"] as? Int == 18)
    let inputDetails = try #require(usage["input_tokens_details"] as? [String: Any])
    let outputDetails = try #require(usage["output_tokens_details"] as? [String: Any])
    #expect(inputDetails["cached_tokens"] as? Int == 3)
    #expect(outputDetails["reasoning_tokens"] as? Int == 2)

    let output = try #require(response["output"] as? [[String: Any]])
    #expect(output.count == 5)
    #expect(output[0]["type"] as? String == "message")
    #expect(output[1]["type"] as? String == "function_call")
    #expect(output[1]["call_id"] as? String == "call_1")
    #expect(output[2]["type"] as? String == "reasoning")
    let reasoningSummary = try #require(output[2]["summary"] as? [[String: Any]])
    #expect(reasoningSummary[0]["type"] as? String == "summary_text")
    #expect(output[3]["type"] as? String == "function_call_output")

    let messageContent = try #require(output[0]["content"] as? [[String: Any]])
    #expect(messageContent[0]["type"] as? String == "output_text")
    #expect(messageContent[1]["type"] as? String == "refusal")

    let annotations = try #require(messageContent[0]["annotations"] as? [[String: Any]])
    #expect(annotations[0]["type"] as? String == "url_citation")
    #expect(annotations[0]["start_index"] as? Int == 2)
    #expect(annotations[0]["end_index"] as? Int == 9)

    let functionOutputContent = try #require(output[4]["output"] as? [[String: Any]])
    #expect(functionOutputContent[0]["type"] as? String == "input_text")
    #expect(functionOutputContent[1]["type"] as? String == "input_image")
    #expect(functionOutputContent[1]["image_url"] as? String == "file:///tmp/chart.png")
    #expect(functionOutputContent[1]["file_id"] as? String == "file_img")
    #expect(functionOutputContent[2]["type"] as? String == "input_file")
    #expect(functionOutputContent[2]["file_data"] as? String == "Zm9v")

    let reasoningDelta = ResponseStreamingEvent.reasoningDelta(.init(
      itemId: "rs_1",
      outputIndex: 2,
      contentIndex: 0,
      delta: "think",
      sequenceNumber: 43,
    ))
    let reasoningDeltaObject = try dictionary(from: encoder.encode(reasoningDelta))
    #expect(reasoningDeltaObject["type"] as? String == "response.reasoning.delta")
  }
}

private func assertCodable(_: (some Codable).Type) {}

private func sampleEvents() -> [ResponseStreamingEvent] {
  let response = sampleResponse(status: .completed)
  let message = sampleMessage()
  let functionCall = sampleFunctionCall()
  let textPart = ResponseContentPart.outputText(sampleOutputText())
  let reasoningPart = ResponseContentPart.reasoningText(.init(text: "thinking"))

  return [
    .responseCreated(.init(response: response, sequenceNumber: 0)),
    .responseInProgress(.init(response: response, sequenceNumber: 1)),
    .responseCompleted(.init(response: response, sequenceNumber: 2)),
    .responseIncomplete(.init(response: sampleResponse(status: .incomplete), sequenceNumber: 3)),
    .outputItemAdded(.init(item: .message(message), outputIndex: 0, sequenceNumber: 4)),
    .outputItemDone(.init(item: .functionCall(functionCall), outputIndex: 1, sequenceNumber: 5)),
    .contentPartAdded(.init(
      itemId: message.id,
      outputIndex: 0,
      contentIndex: 0,
      part: textPart,
      sequenceNumber: 6,
    )),
    .contentPartDone(.init(
      itemId: "rs_1",
      outputIndex: 2,
      contentIndex: 0,
      part: reasoningPart,
      sequenceNumber: 7,
    )),
    .outputTextDelta(.init(
      itemId: message.id,
      outputIndex: 0,
      contentIndex: 0,
      delta: "hel",
      sequenceNumber: 8,
    )),
    .outputTextDone(.init(
      itemId: message.id,
      outputIndex: 0,
      contentIndex: 0,
      text: "hello",
      sequenceNumber: 9,
    )),
    .functionCallArgumentsDelta(.init(
      itemId: functionCall.id,
      outputIndex: 1,
      delta: #"{"query":"#,
      sequenceNumber: 10,
    )),
    .functionCallArgumentsDone(.init(
      itemId: functionCall.id,
      outputIndex: 1,
      arguments: functionCall.arguments,
      sequenceNumber: 11,
    )),
    .reasoningDelta(.init(
      itemId: "rs_1",
      outputIndex: 2,
      contentIndex: 0,
      delta: "think",
      sequenceNumber: 12,
    )),
    .reasoningDone(.init(
      itemId: "rs_1",
      outputIndex: 2,
      contentIndex: 0,
      text: "thinking",
      sequenceNumber: 13,
    )),
  ]
}

private func sampleResponse(status: ResponseStatus) -> Response {
  Response(
    id: "resp_1",
    createdAt: 1_700_000_000,
    model: "test-model",
    output: [
      .message(sampleMessage()),
      .functionCall(sampleFunctionCall()),
      .reasoning(sampleReasoning()),
      .functionCallOutput(.init(id: "fco_1", callId: "call_1", output: "ok")),
      .functionCallOutput(sampleContentFunctionOutput()),
    ],
    status: status,
    incompleteDetails: status == .incomplete ? .init(reason: .maxOutputTokens) : nil,
    usage: .init(
      inputTokens: 11,
      outputTokens: 7,
      totalTokens: 18,
      inputTokensDetails: .init(cachedTokens: 3),
      outputTokensDetails: .init(reasoningTokens: 2),
    ),
    instructions: "be concise",
    temperature: 0.7,
    topP: 0.95,
    maxOutputTokens: 512,
  )
}

private func sampleMessage() -> ResponseOutputMessage {
  ResponseOutputMessage(
    id: "msg_1",
    content: [
      .outputText(sampleOutputText()),
      .refusal(.init(refusal: "no")),
    ],
    status: .completed,
    phase: .finalAnswer,
  )
}

private func sampleOutputText() -> ResponseOutputText {
  ResponseOutputText(
    text: "hello",
    annotations: [
      .urlCitation(url: "https://example.com", title: "Example", startIndex: 2, endIndex: 9),
    ],
  )
}

private func sampleFunctionCall() -> ResponseFunctionToolCall {
  ResponseFunctionToolCall(
    id: "fc_1",
    callId: "call_1",
    name: "lookup",
    arguments: #"{"query":"swift"}"#,
    status: .completed,
  )
}

private func sampleReasoning() -> ResponseReasoningItem {
  ResponseReasoningItem(
    id: "rs_1",
    content: [.reasoningText(.init(text: "thinking"))],
    summary: [.init(text: "summary")],
    encryptedContent: "encrypted",
    status: .completed,
  )
}

private func sampleContentFunctionOutput() -> ResponseFunctionCallOutput {
  ResponseFunctionCallOutput(
    id: "fco_2",
    callId: "call_2",
    output: .content([
      .inputText(.init(text: "text result")),
      .inputImage(.init(
        imageURL: "file:///tmp/chart.png",
        fileId: "file_img",
        detail: .auto,
      )),
      .inputFile(.init(
        fileId: "file_doc",
        filename: "report.txt",
        fileData: "Zm9v",
        fileURL: "file:///tmp/report.txt",
      )),
    ]),
  )
}

private func dictionary(from data: Data) throws -> [String: Any] {
  try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}
