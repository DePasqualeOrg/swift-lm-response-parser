// Copyright © Anthony DePasquale

import Foundation

// MARK: Shared coding helpers

private enum TypeCodingKeys: String, CodingKey {
  case type
}

private func validateType(
  _ expected: String,
  aliases: Set<String> = [],
  from decoder: Decoder,
) throws {
  let container = try decoder.container(keyedBy: TypeCodingKeys.self)
  let actual = try container.decode(String.self, forKey: .type)
  guard actual == expected || aliases.contains(actual) else {
    throw DecodingError.dataCorruptedError(
      forKey: .type,
      in: container,
      debugDescription: "Expected type '\(expected)', found '\(actual)'",
    )
  }
}

private struct EmptyObject: Encodable {}

private struct OpenResponsesTextField: Encodable {
  struct Format: Encodable {
    let type = "text"
  }

  let format = Format()
}

private struct OpenResponsesReasoningField: Encodable {
  private enum CodingKeys: String, CodingKey {
    case effort
    case summary
  }

  func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encodeNil(forKey: .effort)
    try container.encodeNil(forKey: .summary)
  }
}

// MARK: Response envelope

extension ResponseStatus: Codable {}
extension IncompleteReason: Codable {}

extension IncompleteDetails: Codable {
  private enum CodingKeys: String, CodingKey {
    case reason
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(reason: container.decode(IncompleteReason.self, forKey: .reason))
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(reason, forKey: .reason)
  }
}

extension ResponseUsage: Codable {
  private enum CodingKeys: String, CodingKey {
    case inputTokens = "input_tokens"
    case outputTokens = "output_tokens"
    case totalTokens = "total_tokens"
    case inputTokensDetails = "input_tokens_details"
    case outputTokensDetails = "output_tokens_details"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      inputTokens: container.decode(Int.self, forKey: .inputTokens),
      outputTokens: container.decode(Int.self, forKey: .outputTokens),
      totalTokens: container.decode(Int.self, forKey: .totalTokens),
      inputTokensDetails: container.decodeIfPresent(
        InputTokensDetails.self,
        forKey: .inputTokensDetails,
      ) ?? .init(),
      outputTokensDetails: container.decodeIfPresent(
        OutputTokensDetails.self,
        forKey: .outputTokensDetails,
      ) ?? .init(),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputTokens, forKey: .inputTokens)
    try container.encode(outputTokens, forKey: .outputTokens)
    try container.encode(totalTokens, forKey: .totalTokens)
    try container.encode(inputTokensDetails, forKey: .inputTokensDetails)
    try container.encode(outputTokensDetails, forKey: .outputTokensDetails)
  }
}

extension ResponseUsage.InputTokensDetails: Codable {
  private enum CodingKeys: String, CodingKey {
    case cachedTokens = "cached_tokens"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(cachedTokens: container.decodeIfPresent(Int.self, forKey: .cachedTokens) ?? 0)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(cachedTokens, forKey: .cachedTokens)
  }
}

extension ResponseUsage.OutputTokensDetails: Codable {
  private enum CodingKeys: String, CodingKey {
    case reasoningTokens = "reasoning_tokens"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(reasoningTokens: container.decodeIfPresent(Int.self, forKey: .reasoningTokens) ?? 0)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(reasoningTokens, forKey: .reasoningTokens)
  }
}

extension Response: Codable {
  private enum CodingKeys: String, CodingKey {
    case id
    case createdAt = "created_at"
    case model
    case output
    case object
    case completedAt = "completed_at"
    case status
    case incompleteDetails = "incomplete_details"
    case previousResponseId = "previous_response_id"
    case usage
    case instructions
    case error
    case tools
    case toolChoice = "tool_choice"
    case truncation
    case parallelToolCalls = "parallel_tool_calls"
    case text
    case temperature
    case topP = "top_p"
    case presencePenalty = "presence_penalty"
    case frequencyPenalty = "frequency_penalty"
    case topLogprobs = "top_logprobs"
    case reasoning
    case maxOutputTokens = "max_output_tokens"
    case maxToolCalls = "max_tool_calls"
    case store
    case background
    case serviceTier = "service_tier"
    case metadata
    case safetyIdentifier = "safety_identifier"
    case promptCacheKey = "prompt_cache_key"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      id: container.decode(String.self, forKey: .id),
      createdAt: container.decode(Int.self, forKey: .createdAt),
      model: container.decode(String.self, forKey: .model),
      output: container.decodeIfPresent([ResponseOutputItem].self, forKey: .output) ?? [],
      status: container.decodeIfPresent(ResponseStatus.self, forKey: .status),
      incompleteDetails: container.decodeIfPresent(IncompleteDetails.self, forKey: .incompleteDetails),
      usage: container.decodeIfPresent(ResponseUsage.self, forKey: .usage),
      instructions: container.decodeIfPresent(String.self, forKey: .instructions),
      temperature: container.decodeIfPresent(Double.self, forKey: .temperature),
      topP: container.decodeIfPresent(Double.self, forKey: .topP),
      maxOutputTokens: container.decodeIfPresent(Int.self, forKey: .maxOutputTokens),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(id, forKey: .id)
    try container.encode("response", forKey: .object)
    try container.encode(createdAt, forKey: .createdAt)
    try container.encodeNil(forKey: .completedAt)
    try container.encode(status ?? .inProgress, forKey: .status)
    try container.encodeIfPresent(incompleteDetails, forKey: .incompleteDetails)
    if incompleteDetails == nil {
      try container.encodeNil(forKey: .incompleteDetails)
    }
    try container.encode(model, forKey: .model)
    try container.encodeNil(forKey: .previousResponseId)
    if let instructions {
      try container.encode(instructions, forKey: .instructions)
    } else {
      try container.encodeNil(forKey: .instructions)
    }
    try container.encode(output, forKey: .output)
    try container.encodeNil(forKey: .error)
    try container.encode([String](), forKey: .tools)
    try container.encode("auto", forKey: .toolChoice)
    try container.encode("disabled", forKey: .truncation)
    try container.encode(true, forKey: .parallelToolCalls)
    try container.encode(OpenResponsesTextField(), forKey: .text)
    try container.encode(topP ?? 1, forKey: .topP)
    try container.encode(0, forKey: .presencePenalty)
    try container.encode(0, forKey: .frequencyPenalty)
    try container.encode(0, forKey: .topLogprobs)
    try container.encode(temperature ?? 1, forKey: .temperature)
    try container.encode(OpenResponsesReasoningField(), forKey: .reasoning)
    if let usage {
      try container.encode(usage, forKey: .usage)
    } else {
      try container.encodeNil(forKey: .usage)
    }
    if let maxOutputTokens {
      try container.encode(maxOutputTokens, forKey: .maxOutputTokens)
    } else {
      try container.encodeNil(forKey: .maxOutputTokens)
    }
    try container.encodeNil(forKey: .maxToolCalls)
    try container.encode(true, forKey: .store)
    try container.encode(false, forKey: .background)
    try container.encode("default", forKey: .serviceTier)
    try container.encode(EmptyObject(), forKey: .metadata)
    try container.encodeNil(forKey: .safetyIdentifier)
    try container.encodeNil(forKey: .promptCacheKey)
  }
}

// MARK: Output items

extension ItemStatus: Codable {}
extension ResponseOutputMessage.Role: Codable {}
extension ResponseOutputMessage.Phase: Codable {}

extension ResponseOutputItem: Codable {
  private enum ItemType: String, Codable {
    case message
    case functionCall = "function_call"
    case reasoning
    case functionCallOutput = "function_call_output"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: TypeCodingKeys.self)
    let type = try container.decode(ItemType.self, forKey: .type)
    switch type {
      case .message:
        self = try .message(ResponseOutputMessage(from: decoder))
      case .functionCall:
        self = try .functionCall(ResponseFunctionToolCall(from: decoder))
      case .reasoning:
        self = try .reasoning(ResponseReasoningItem(from: decoder))
      case .functionCallOutput:
        self = try .functionCallOutput(ResponseFunctionCallOutput(from: decoder))
    }
  }

  public func encode(to encoder: Encoder) throws {
    switch self {
      case let .message(value):
        try value.encode(to: encoder)
      case let .functionCall(value):
        try value.encode(to: encoder)
      case let .reasoning(value):
        try value.encode(to: encoder)
      case let .functionCallOutput(value):
        try value.encode(to: encoder)
    }
  }
}

extension ResponseOutputMessage: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case id
    case role
    case content
    case status
    case phase
  }

  public init(from decoder: Decoder) throws {
    try validateType("message", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      id: container.decode(String.self, forKey: .id),
      role: container.decodeIfPresent(Role.self, forKey: .role) ?? .assistant,
      content: container.decodeIfPresent([ResponseContentPart].self, forKey: .content) ?? [],
      status: container.decodeIfPresent(ItemStatus.self, forKey: .status) ?? .inProgress,
      phase: container.decodeIfPresent(Phase.self, forKey: .phase),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("message", forKey: .type)
    try container.encode(id, forKey: .id)
    try container.encode(role, forKey: .role)
    try container.encode(content, forKey: .content)
    try container.encode(status, forKey: .status)
    try container.encodeIfPresent(phase, forKey: .phase)
  }
}

extension ResponseFunctionToolCall: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case id
    case callId = "call_id"
    case name
    case arguments
    case status
  }

  public init(from decoder: Decoder) throws {
    try validateType("function_call", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      id: container.decode(String.self, forKey: .id),
      callId: container.decode(String.self, forKey: .callId),
      name: container.decode(String.self, forKey: .name),
      arguments: container.decode(String.self, forKey: .arguments),
      status: container.decodeIfPresent(ItemStatus.self, forKey: .status) ?? .inProgress,
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("function_call", forKey: .type)
    try container.encode(id, forKey: .id)
    try container.encode(callId, forKey: .callId)
    try container.encode(name, forKey: .name)
    try container.encode(arguments, forKey: .arguments)
    try container.encode(status, forKey: .status)
  }
}

extension ResponseReasoningItem: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case id
    case content
    case summary
    case encryptedContent = "encrypted_content"
    case status
  }

  public init(from decoder: Decoder) throws {
    try validateType("reasoning", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      id: container.decode(String.self, forKey: .id),
      content: container.decodeIfPresent([ResponseContentPart].self, forKey: .content) ?? [],
      summary: container.decodeIfPresent([SummaryPart].self, forKey: .summary) ?? [],
      encryptedContent: container.decodeIfPresent(String.self, forKey: .encryptedContent),
      status: container.decodeIfPresent(ItemStatus.self, forKey: .status) ?? .inProgress,
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("reasoning", forKey: .type)
    try container.encode(id, forKey: .id)
    try container.encode(content, forKey: .content)
    try container.encode(summary, forKey: .summary)
    try container.encodeIfPresent(encryptedContent, forKey: .encryptedContent)
    try container.encode(status, forKey: .status)
  }
}

extension ResponseReasoningItem.SummaryPart: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case text
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    if let type = try container.decodeIfPresent(String.self, forKey: .type), type != "summary_text" {
      throw DecodingError.dataCorruptedError(
        forKey: .type,
        in: container,
        debugDescription: "Expected type 'summary_text', found '\(type)'",
      )
    }
    try self.init(text: container.decode(String.self, forKey: .text))
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("summary_text", forKey: .type)
    try container.encode(text, forKey: .text)
  }
}

extension ResponseFunctionCallOutput: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case id
    case callId = "call_id"
    case output
    case status
  }

  public init(from decoder: Decoder) throws {
    try validateType("function_call_output", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      id: container.decode(String.self, forKey: .id),
      callId: container.decode(String.self, forKey: .callId),
      output: container.decode(Output.self, forKey: .output),
      status: container.decodeIfPresent(ItemStatus.self, forKey: .status) ?? .completed,
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("function_call_output", forKey: .type)
    try container.encode(id, forKey: .id)
    try container.encode(callId, forKey: .callId)
    try container.encode(output, forKey: .output)
    try container.encode(status, forKey: .status)
  }
}

extension ResponseFunctionCallOutput.Output: Codable {
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if let value = try? container.decode(String.self) {
      self = .string(value)
      return
    }
    self = try .content(container.decode([ResponseFunctionCallOutput.Content].self))
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
      case let .string(value):
        try container.encode(value)
      case let .content(value):
        try container.encode(value)
    }
  }
}

extension ResponseFunctionCallOutput.Content: Codable {
  private enum ContentType: String, Codable {
    case inputText = "input_text"
    case inputImage = "input_image"
    case inputFile = "input_file"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: TypeCodingKeys.self)
    let type = try container.decode(ContentType.self, forKey: .type)
    switch type {
      case .inputText:
        self = try .inputText(ResponseFunctionCallOutput.InputText(from: decoder))
      case .inputImage:
        self = try .inputImage(ResponseFunctionCallOutput.InputImage(from: decoder))
      case .inputFile:
        self = try .inputFile(ResponseFunctionCallOutput.InputFile(from: decoder))
    }
  }

  public func encode(to encoder: Encoder) throws {
    switch self {
      case let .inputText(value):
        try value.encode(to: encoder)
      case let .inputImage(value):
        try value.encode(to: encoder)
      case let .inputFile(value):
        try value.encode(to: encoder)
    }
  }
}

extension ResponseFunctionCallOutput.InputText: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case text
  }

  public init(from decoder: Decoder) throws {
    try validateType("input_text", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(text: container.decode(String.self, forKey: .text))
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("input_text", forKey: .type)
    try container.encode(text, forKey: .text)
  }
}

extension ResponseFunctionCallOutput.InputImage.Detail: Codable {
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let rawValue = try container.decode(String.self)
    guard let value = Self(rawValue: rawValue) else {
      throw DecodingError.dataCorruptedError(
        in: container,
        debugDescription: "Unsupported image detail '\(rawValue)'",
      )
    }
    self = value
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(rawValue)
  }
}

extension ResponseFunctionCallOutput.InputImage: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case imageURL = "image_url"
    case fileId = "file_id"
    case detail
  }

  public init(from decoder: Decoder) throws {
    try validateType("input_image", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      imageURL: container.decodeIfPresent(String.self, forKey: .imageURL),
      fileId: container.decodeIfPresent(String.self, forKey: .fileId),
      detail: container.decodeIfPresent(Detail.self, forKey: .detail),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("input_image", forKey: .type)
    if let imageURL {
      try container.encode(imageURL, forKey: .imageURL)
    } else {
      try container.encodeNil(forKey: .imageURL)
    }
    try container.encodeIfPresent(fileId, forKey: .fileId)
    try container.encode(detail ?? .auto, forKey: .detail)
  }
}

extension ResponseFunctionCallOutput.InputFile: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case fileId = "file_id"
    case filename
    case fileData = "file_data"
    case fileURL = "file_url"
  }

  public init(from decoder: Decoder) throws {
    try validateType("input_file", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      fileId: container.decodeIfPresent(String.self, forKey: .fileId),
      filename: container.decodeIfPresent(String.self, forKey: .filename),
      fileData: container.decodeIfPresent(String.self, forKey: .fileData),
      fileURL: container.decodeIfPresent(String.self, forKey: .fileURL),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("input_file", forKey: .type)
    try container.encodeIfPresent(fileId, forKey: .fileId)
    try container.encodeIfPresent(filename, forKey: .filename)
    try container.encodeIfPresent(fileData, forKey: .fileData)
    try container.encodeIfPresent(fileURL, forKey: .fileURL)
  }
}

// MARK: Content parts

extension ResponseContentPart: Codable {
  private enum PartType: String, Codable {
    case outputText = "output_text"
    case refusal
    case reasoningText = "reasoning_text"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: TypeCodingKeys.self)
    let type = try container.decode(PartType.self, forKey: .type)
    switch type {
      case .outputText:
        self = try .outputText(ResponseOutputText(from: decoder))
      case .refusal:
        self = try .refusal(ResponseOutputRefusal(from: decoder))
      case .reasoningText:
        self = try .reasoningText(ReasoningTextContent(from: decoder))
    }
  }

  public func encode(to encoder: Encoder) throws {
    switch self {
      case let .outputText(value):
        try value.encode(to: encoder)
      case let .refusal(value):
        try value.encode(to: encoder)
      case let .reasoningText(value):
        try value.encode(to: encoder)
    }
  }
}

extension ResponseOutputText: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case text
    case annotations
  }

  public init(from decoder: Decoder) throws {
    try validateType("output_text", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      text: container.decode(String.self, forKey: .text),
      annotations: container.decodeIfPresent([Annotation].self, forKey: .annotations) ?? [],
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("output_text", forKey: .type)
    try container.encode(text, forKey: .text)
    try container.encode(annotations, forKey: .annotations)
  }
}

extension ResponseOutputText.Annotation: Codable {
  private enum AnnotationType: String, Codable {
    case fileCitation = "file_citation"
    case urlCitation = "url_citation"
    case filePath = "file_path"
  }

  private enum CodingKeys: String, CodingKey {
    case type
    case fileId = "file_id"
    case filename
    case index
    case url
    case title
    case startIndex = "start_index"
    case endIndex = "end_index"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let type = try container.decode(AnnotationType.self, forKey: .type)
    switch type {
      case .fileCitation:
        self = try .fileCitation(
          fileId: container.decode(String.self, forKey: .fileId),
          filename: container.decode(String.self, forKey: .filename),
          index: container.decode(Int.self, forKey: .index),
        )
      case .urlCitation:
        self = try .urlCitation(
          url: container.decode(String.self, forKey: .url),
          title: container.decodeIfPresent(String.self, forKey: .title),
          startIndex: container.decode(Int.self, forKey: .startIndex),
          endIndex: container.decode(Int.self, forKey: .endIndex),
        )
      case .filePath:
        self = try .filePath(
          fileId: container.decode(String.self, forKey: .fileId),
          index: container.decode(Int.self, forKey: .index),
        )
    }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    switch self {
      case let .fileCitation(fileId, filename, index):
        try container.encode(AnnotationType.fileCitation, forKey: .type)
        try container.encode(fileId, forKey: .fileId)
        try container.encode(filename, forKey: .filename)
        try container.encode(index, forKey: .index)
      case let .urlCitation(url, title, startIndex, endIndex):
        try container.encode(AnnotationType.urlCitation, forKey: .type)
        try container.encode(url, forKey: .url)
        try container.encode(title ?? "", forKey: .title)
        try container.encode(startIndex, forKey: .startIndex)
        try container.encode(endIndex, forKey: .endIndex)
      case let .filePath(fileId, index):
        try container.encode(AnnotationType.filePath, forKey: .type)
        try container.encode(fileId, forKey: .fileId)
        try container.encode(index, forKey: .index)
    }
  }
}

extension ResponseOutputRefusal: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case refusal
  }

  public init(from decoder: Decoder) throws {
    try validateType("refusal", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(refusal: container.decode(String.self, forKey: .refusal))
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("refusal", forKey: .type)
    try container.encode(refusal, forKey: .refusal)
  }
}

extension ReasoningTextContent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case text
  }

  public init(from decoder: Decoder) throws {
    try validateType("reasoning_text", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(text: container.decode(String.self, forKey: .text))
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("reasoning_text", forKey: .type)
    try container.encode(text, forKey: .text)
  }
}

// MARK: Streaming events

extension ResponseStreamingEvent: Codable {
  private enum EventType: Decodable {
    case responseCreated
    case responseInProgress
    case responseCompleted
    case responseIncomplete
    case outputItemAdded
    case outputItemDone
    case contentPartAdded
    case contentPartDone
    case outputTextDelta
    case outputTextDone
    case functionCallArgumentsDelta
    case functionCallArgumentsDone
    case reasoningDelta
    case reasoningDone

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      let rawValue = try container.decode(String.self)
      switch rawValue {
        case "response.created": self = .responseCreated
        case "response.in_progress": self = .responseInProgress
        case "response.completed": self = .responseCompleted
        case "response.incomplete": self = .responseIncomplete
        case "response.output_item.added": self = .outputItemAdded
        case "response.output_item.done": self = .outputItemDone
        case "response.content_part.added": self = .contentPartAdded
        case "response.content_part.done": self = .contentPartDone
        case "response.output_text.delta": self = .outputTextDelta
        case "response.output_text.done": self = .outputTextDone
        case "response.function_call_arguments.delta": self = .functionCallArgumentsDelta
        case "response.function_call_arguments.done": self = .functionCallArgumentsDone
        // Open Responses currently specifies `response.reasoning.*`.
        // vLLM and SGLang still emit `response.reasoning_text.*`, so decode
        // both while continuing to encode the Open Responses discriminator.
        case "response.reasoning.delta", "response.reasoning_text.delta": self = .reasoningDelta
        case "response.reasoning.done", "response.reasoning_text.done": self = .reasoningDone
        default:
          throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "Unsupported response event type '\(rawValue)'",
          )
      }
    }
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: TypeCodingKeys.self)
    let type = try container.decode(EventType.self, forKey: .type)
    switch type {
      case .responseCreated:
        self = try .responseCreated(ResponseCreatedEvent(from: decoder))
      case .responseInProgress:
        self = try .responseInProgress(ResponseInProgressEvent(from: decoder))
      case .responseCompleted:
        self = try .responseCompleted(ResponseCompletedEvent(from: decoder))
      case .responseIncomplete:
        self = try .responseIncomplete(ResponseIncompleteEvent(from: decoder))
      case .outputItemAdded:
        self = try .outputItemAdded(ResponseOutputItemAddedEvent(from: decoder))
      case .outputItemDone:
        self = try .outputItemDone(ResponseOutputItemDoneEvent(from: decoder))
      case .contentPartAdded:
        self = try .contentPartAdded(ResponseContentPartAddedEvent(from: decoder))
      case .contentPartDone:
        self = try .contentPartDone(ResponseContentPartDoneEvent(from: decoder))
      case .outputTextDelta:
        self = try .outputTextDelta(ResponseTextDeltaEvent(from: decoder))
      case .outputTextDone:
        self = try .outputTextDone(ResponseTextDoneEvent(from: decoder))
      case .functionCallArgumentsDelta:
        self = try .functionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent(from: decoder))
      case .functionCallArgumentsDone:
        self = try .functionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent(from: decoder))
      case .reasoningDelta:
        self = try .reasoningDelta(ResponseReasoningDeltaEvent(from: decoder))
      case .reasoningDone:
        self = try .reasoningDone(ResponseReasoningDoneEvent(from: decoder))
    }
  }

  public func encode(to encoder: Encoder) throws {
    switch self {
      case let .responseCreated(value):
        try value.encode(to: encoder)
      case let .responseInProgress(value):
        try value.encode(to: encoder)
      case let .responseCompleted(value):
        try value.encode(to: encoder)
      case let .responseIncomplete(value):
        try value.encode(to: encoder)
      case let .outputItemAdded(value):
        try value.encode(to: encoder)
      case let .outputItemDone(value):
        try value.encode(to: encoder)
      case let .contentPartAdded(value):
        try value.encode(to: encoder)
      case let .contentPartDone(value):
        try value.encode(to: encoder)
      case let .outputTextDelta(value):
        try value.encode(to: encoder)
      case let .outputTextDone(value):
        try value.encode(to: encoder)
      case let .functionCallArgumentsDelta(value):
        try value.encode(to: encoder)
      case let .functionCallArgumentsDone(value):
        try value.encode(to: encoder)
      case let .reasoningDelta(value):
        try value.encode(to: encoder)
      case let .reasoningDone(value):
        try value.encode(to: encoder)
    }
  }
}

extension ResponseCreatedEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case response
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.created", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      response: container.decode(Response.self, forKey: .response),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.created", forKey: .type)
    try container.encode(response, forKey: .response)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseInProgressEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case response
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.in_progress", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      response: container.decode(Response.self, forKey: .response),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.in_progress", forKey: .type)
    try container.encode(response, forKey: .response)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseCompletedEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case response
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.completed", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      response: container.decode(Response.self, forKey: .response),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.completed", forKey: .type)
    try container.encode(response, forKey: .response)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseIncompleteEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case response
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.incomplete", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      response: container.decode(Response.self, forKey: .response),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.incomplete", forKey: .type)
    try container.encode(response, forKey: .response)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseOutputItemAddedEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case item
    case outputIndex = "output_index"
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.output_item.added", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      item: container.decode(ResponseOutputItem.self, forKey: .item),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.output_item.added", forKey: .type)
    try container.encode(item, forKey: .item)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseOutputItemDoneEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case item
    case outputIndex = "output_index"
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.output_item.done", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      item: container.decode(ResponseOutputItem.self, forKey: .item),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.output_item.done", forKey: .type)
    try container.encode(item, forKey: .item)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseContentPartAddedEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case contentIndex = "content_index"
    case part
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.content_part.added", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      contentIndex: container.decode(Int.self, forKey: .contentIndex),
      part: container.decode(ResponseContentPart.self, forKey: .part),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.content_part.added", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(contentIndex, forKey: .contentIndex)
    try container.encode(part, forKey: .part)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseContentPartDoneEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case contentIndex = "content_index"
    case part
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.content_part.done", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      contentIndex: container.decode(Int.self, forKey: .contentIndex),
      part: container.decode(ResponseContentPart.self, forKey: .part),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.content_part.done", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(contentIndex, forKey: .contentIndex)
    try container.encode(part, forKey: .part)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseTextDeltaEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case contentIndex = "content_index"
    case delta
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.output_text.delta", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      contentIndex: container.decode(Int.self, forKey: .contentIndex),
      delta: container.decode(String.self, forKey: .delta),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.output_text.delta", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(contentIndex, forKey: .contentIndex)
    try container.encode(delta, forKey: .delta)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseTextDoneEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case contentIndex = "content_index"
    case text
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.output_text.done", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      contentIndex: container.decode(Int.self, forKey: .contentIndex),
      text: container.decode(String.self, forKey: .text),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.output_text.done", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(contentIndex, forKey: .contentIndex)
    try container.encode(text, forKey: .text)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseFunctionCallArgumentsDeltaEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case delta
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.function_call_arguments.delta", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      delta: container.decode(String.self, forKey: .delta),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.function_call_arguments.delta", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(delta, forKey: .delta)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseFunctionCallArgumentsDoneEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case arguments
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    try validateType("response.function_call_arguments.done", from: decoder)
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      arguments: container.decode(String.self, forKey: .arguments),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.function_call_arguments.done", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(arguments, forKey: .arguments)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseReasoningDeltaEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case contentIndex = "content_index"
    case delta
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    // Accept vLLM/SGLang's legacy discriminator, but encode the current
    // Open Responses discriminator in `encode(to:)`.
    try validateType(
      "response.reasoning.delta",
      aliases: ["response.reasoning_text.delta"],
      from: decoder,
    )
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      contentIndex: container.decode(Int.self, forKey: .contentIndex),
      delta: container.decode(String.self, forKey: .delta),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.reasoning.delta", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(contentIndex, forKey: .contentIndex)
    try container.encode(delta, forKey: .delta)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}

extension ResponseReasoningDoneEvent: Codable {
  private enum CodingKeys: String, CodingKey {
    case type
    case itemId = "item_id"
    case outputIndex = "output_index"
    case contentIndex = "content_index"
    case text
    case sequenceNumber = "sequence_number"
  }

  public init(from decoder: Decoder) throws {
    // Accept vLLM/SGLang's legacy discriminator, but encode the current
    // Open Responses discriminator in `encode(to:)`.
    try validateType(
      "response.reasoning.done",
      aliases: ["response.reasoning_text.done"],
      from: decoder,
    )
    let container = try decoder.container(keyedBy: CodingKeys.self)
    try self.init(
      itemId: container.decode(String.self, forKey: .itemId),
      outputIndex: container.decode(Int.self, forKey: .outputIndex),
      contentIndex: container.decode(Int.self, forKey: .contentIndex),
      text: container.decode(String.self, forKey: .text),
      sequenceNumber: container.decode(Int.self, forKey: .sequenceNumber),
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode("response.reasoning.done", forKey: .type)
    try container.encode(itemId, forKey: .itemId)
    try container.encode(outputIndex, forKey: .outputIndex)
    try container.encode(contentIndex, forKey: .contentIndex)
    try container.encode(text, forKey: .text)
    try container.encode(sequenceNumber, forKey: .sequenceNumber)
  }
}
