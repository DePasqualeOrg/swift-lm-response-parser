// Copyright © Anthony DePasquale

import Foundation

/// One output item produced by the model during a response.
///
/// Parsers emit message, function-call, and reasoning items. Integration
/// layers can also synthesize function-call-output items when tool dispatch
/// runs inside the same logical response turn. The other spec variants are
/// hosted-product features (web search, file search, MCP, computer use,
/// image generation, and so on) with no source in the parser core. Future
/// parsers can add cases for variants that genuinely arise from local model
/// output without reshaping the protocol.
public enum ResponseOutputItem: Sendable, Equatable {
  case message(ResponseOutputMessage)
  case functionCall(ResponseFunctionToolCall)
  case reasoning(ResponseReasoningItem)
  case functionCallOutput(ResponseFunctionCallOutput)

  /// Convenience accessor for the item's stable spec ID (`msg_…`, `fc_…`,
  /// `rs_…`, `fco_…`). The ID is shared across every event that touches
  /// this item.
  public var id: String {
    switch self {
      case let .message(m): m.id
      case let .functionCall(f): f.id
      case let .reasoning(r): r.id
      case let .functionCallOutput(o): o.id
    }
  }
}

/// Status of an item at emit time. The values mirror the spec's three
/// shared states across `MessageStatus`, `FunctionCallStatus`, and the
/// reasoning-item status enum.
public enum ItemStatus: String, Sendable, Equatable, CaseIterable {
  case inProgress = "in_progress"
  case completed
  case incomplete
}

/// An assistant message item.
public struct ResponseOutputMessage: Sendable, Equatable {
  /// Item ID minted by the parser (`msg_…`).
  public var id: String

  /// Always `assistant` for items the parser emits – the spec allows other
  /// roles only on input items.
  public var role: Role

  /// The item's content parts. For a fully streamed message, contains one
  /// `outputText` part with the accumulated text. For interrupted messages,
  /// the array contains whatever was emitted before the truncation point.
  public var content: [ResponseContentPart]

  /// Item status at emit time. `inProgress` on the initial
  /// `output_item.added` event; `completed` or `incomplete` on the
  /// matching `output_item.done` event.
  public var status: ItemStatus

  /// Harmony-only field: `commentary` vs `final_answer`. nil for every
  /// other parser. Kept optional so message equality in test assertions
  /// works whether or not phase is populated.
  public var phase: Phase?

  public enum Role: String, Sendable, Equatable {
    case assistant
  }

  public enum Phase: String, Sendable, Equatable {
    case commentary
    case finalAnswer = "final_answer"
  }

  public init(
    id: String,
    role: Role = .assistant,
    content: [ResponseContentPart] = [],
    status: ItemStatus = .inProgress,
    phase: Phase? = nil,
  ) {
    self.id = id
    self.role = role
    self.content = content
    self.status = status
    self.phase = phase
  }

  /// Concatenated text from every `outputText` content part. Returns
  /// `""` for an interrupted message that opened but produced no text.
  public var text: String {
    content.compactMap { part in
      if case let .outputText(t) = part { return t.text }
      return nil
    }.joined()
  }
}

/// A function-call item.
///
/// Function-call arguments live on the item itself (as the `arguments`
/// JSON string) rather than inside a `content_part.added/done` envelope.
/// The streaming events for argument deltas
/// (`function_call_arguments.delta` / `function_call_arguments.done`) carry
/// the same string content.
public struct ResponseFunctionToolCall: Sendable, Equatable {
  /// Item ID minted by the parser (`fc_…`).
  public var id: String

  /// A separate ID minted at item-open time and threaded through the
  /// `call_id` field so consumers can correlate the call with its later
  /// `function_call_output` input item. Prefix is `call_…`, distinct
  /// from the item-scoped `fc_…` ID above.
  public var callId: String

  /// Name of the function the model wants to call.
  public var name: String

  /// JSON-encoded arguments object as a string, matching the spec.
  /// Argument-type coercion (for wire formats that emit untyped key/value
  /// pairs) is performed by the parsers that need it before the JSON is
  /// materialized. Treat this as parser output, not execution approval:
  /// dispatch code should still validate the tool name, decode the
  /// arguments into the expected type, enforce application policy, and
  /// decide whether to run the tool.
  public var arguments: String

  public var status: ItemStatus

  public init(
    id: String,
    callId: String,
    name: String,
    arguments: String,
    status: ItemStatus = .inProgress,
  ) {
    self.id = id
    self.callId = callId
    self.name = name
    self.arguments = arguments
    self.status = status
  }

  /// Decode ``arguments`` into a `Decodable` type.
  ///
  /// Throws a `DecodingError` when the JSON does not match `T` — either
  /// because the model emitted malformed or unexpected JSON, or because
  /// `T`'s shape differs from the function's parameter schema. In a
  /// tool-dispatch path, catch this error and return a model-visible tool
  /// output when the model should be allowed to correct its call; throw
  /// only when the response stream should abort.
  public func decodedArguments<T: Decodable>(
    as type: T.Type,
    decoder: JSONDecoder = JSONDecoder(),
  ) throws -> T {
    try decoder.decode(type, from: Data(arguments.utf8))
  }
}

/// A reasoning item (chain-of-thought).
///
/// `summary[]` is always empty (no open-source model emits summary content
/// in its wire format) and `encryptedContent` is always nil (encryption
/// belongs in a server layer). The actual reasoning text flows through
/// `content[]` as a `reasoningText` content part.
public struct ResponseReasoningItem: Sendable, Equatable {
  /// Item ID minted by the parser (`rs_…`).
  public var id: String

  /// Reasoning content parts. In practice always a single `reasoningText`
  /// part with the accumulated reasoning, or empty if the item was opened
  /// but no text was produced before truncation.
  public var content: [ResponseContentPart]

  /// Always empty; preserved as a structural field so the spec shape is
  /// intact.
  public var summary: [SummaryPart]

  /// Always nil from the parser; a server-side wrapper can populate it.
  public var encryptedContent: String?

  public var status: ItemStatus

  public struct SummaryPart: Sendable, Equatable {
    public var text: String
    public init(text: String) {
      self.text = text
    }
  }

  public init(
    id: String,
    content: [ResponseContentPart] = [],
    summary: [SummaryPart] = [],
    encryptedContent: String? = nil,
    status: ItemStatus = .inProgress,
  ) {
    self.id = id
    self.content = content
    self.summary = summary
    self.encryptedContent = encryptedContent
    self.status = status
  }

  /// Concatenated text from every `reasoningText` content part. Returns
  /// `""` for an interrupted item that opened but produced no text.
  public var text: String {
    content.compactMap { part in
      if case let .reasoningText(t) = part { return t.text }
      return nil
    }.joined()
  }
}

/// A function-call-output item (tool result).
///
/// Emitted by integration layers (not by parsers) to surface tool-dispatch
/// results back into the same Responses envelope as the function call that
/// triggered them. Paired by ``callId`` to the originating
/// ``ResponseFunctionToolCall`` so consumers can correlate the call with its
/// result.
///
/// The spec allows `output: string | Array<input_text | input_image |
/// input_file>`. Use ``Output/string(_:)`` for ordinary text and JSON
/// string payloads. Use ``Output/content(_:)`` when a tool result should
/// preserve typed text, image, or file parts for a Responses-compatible
/// consumer.
public struct ResponseFunctionCallOutput: Sendable, Equatable {
  /// Result returned by the dispatched tool.
  public enum Output: Sendable, Equatable {
    /// Text output. Structured JSON tool results should usually use this
    /// branch by returning a JSON string.
    case string(String)

    /// Typed content output matching the Responses API's content-array
    /// form for function-call output.
    case content([Content])

    /// Returns the underlying string for ``string(_:)`` output, or `nil`
    /// for typed content output.
    public var stringValue: String? {
      guard case let .string(value) = self else { return nil }
      return value
    }
  }

  /// A typed content part inside ``Output/content(_:)``.
  public enum Content: Sendable, Equatable {
    case inputText(InputText)
    case inputImage(InputImage)
    case inputFile(InputFile)
  }

  /// Text input content returned by a function tool.
  public struct InputText: Sendable, Equatable {
    public var text: String

    public init(text: String) {
      self.text = text
    }
  }

  /// Image input content returned by a function tool.
  public struct InputImage: Sendable, Equatable {
    public enum Detail: String, Sendable, Equatable, CaseIterable {
      case low
      case high
      case auto
      case original
    }

    /// A fully qualified URL, data URL, or local file URL for the image.
    public var imageURL: String?

    /// A hosted file identifier, when the host environment supports one.
    public var fileId: String?

    public var detail: Detail?

    public init(
      imageURL: String? = nil,
      fileId: String? = nil,
      detail: Detail? = nil,
    ) {
      self.imageURL = imageURL
      self.fileId = fileId
      self.detail = detail
    }
  }

  /// File input content returned by a function tool.
  public struct InputFile: Sendable, Equatable {
    /// A hosted file identifier, when the host environment supports one.
    public var fileId: String?

    public var filename: String?

    /// Base64-encoded file data.
    public var fileData: String?

    /// A URL for the file.
    public var fileURL: String?

    public init(
      fileId: String? = nil,
      filename: String? = nil,
      fileData: String? = nil,
      fileURL: String? = nil,
    ) {
      self.fileId = fileId
      self.filename = filename
      self.fileData = fileData
      self.fileURL = fileURL
    }
  }

  /// Item ID minted by the integration layer (`fco_…`).
  public var id: String

  /// The originating function call's ``ResponseFunctionToolCall/callId``.
  public var callId: String

  /// Result returned by the dispatched tool.
  public var output: Output

  public var status: ItemStatus

  public init(
    id: String,
    callId: String,
    output: Output,
    status: ItemStatus = .completed,
  ) {
    self.id = id
    self.callId = callId
    self.output = output
    self.status = status
  }

  public init(
    id: String,
    callId: String,
    output: String,
    status: ItemStatus = .completed,
  ) {
    self.init(id: id, callId: callId, output: .string(output), status: status)
  }
}
