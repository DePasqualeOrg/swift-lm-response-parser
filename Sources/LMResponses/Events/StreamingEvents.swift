// Copyright © Anthony DePasquale

import Foundation

/// One spec-shaped streaming event emitted during a response.
///
/// Covers the fourteen events the parser layer and the emitter together
/// produce:
///
/// - **Lifecycle envelope** (emitted by ``ResponseStream``):
///   `responseCreated`, `responseInProgress`, `responseCompleted`,
///   `responseIncomplete`.
/// - **Item lifecycle** (emitted by per-format parsers):
///   `outputItemAdded`, `outputItemDone`.
/// - **Content-part envelope** (emitted by per-format parsers, around message
///   and reasoning text):
///   `contentPartAdded`, `contentPartDone`.
/// - **Text deltas inside a message**:
///   `outputTextDelta`, `outputTextDone`.
/// - **Argument deltas inside a function call**:
///   `functionCallArgumentsDelta`, `functionCallArgumentsDone`.
/// - **Reasoning deltas inside a reasoning item**:
///   `reasoningDelta`, `reasoningDone`.
///
/// Other event types defined by the spec (`response.failed`,
/// `response.queued`, `response.refusal.*`, the
/// hosted-product event families like `code_interpreter.*`, `web_search.*`,
/// `mcp.*`, image-generation events, and so on) are not produced: either
/// they are hosted-product surface that has no source on a local device,
/// or the spec defines them but canonical client SDKs do not branch on
/// them.
public enum ResponseStreamingEvent: Sendable, Equatable {
  case responseCreated(ResponseCreatedEvent)
  case responseInProgress(ResponseInProgressEvent)
  case responseCompleted(ResponseCompletedEvent)
  case responseIncomplete(ResponseIncompleteEvent)

  case outputItemAdded(ResponseOutputItemAddedEvent)
  case outputItemDone(ResponseOutputItemDoneEvent)

  case contentPartAdded(ResponseContentPartAddedEvent)
  case contentPartDone(ResponseContentPartDoneEvent)

  case outputTextDelta(ResponseTextDeltaEvent)
  case outputTextDone(ResponseTextDoneEvent)

  case functionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent)
  case functionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent)

  case reasoningDelta(ResponseReasoningDeltaEvent)
  case reasoningDone(ResponseReasoningDoneEvent)

  /// Read or write the event's `sequence_number` field. The parser emits
  /// events with parser-local sequence numbers; the emitter substitutes
  /// response-scoped numbers (zero-based, monotonically increasing across
  /// the entire stream including envelope events) before yielding to the
  /// consumer.
  public package(set) var sequenceNumber: Int {
    get {
      switch self {
        case let .responseCreated(e): e.sequenceNumber
        case let .responseInProgress(e): e.sequenceNumber
        case let .responseCompleted(e): e.sequenceNumber
        case let .responseIncomplete(e): e.sequenceNumber
        case let .outputItemAdded(e): e.sequenceNumber
        case let .outputItemDone(e): e.sequenceNumber
        case let .contentPartAdded(e): e.sequenceNumber
        case let .contentPartDone(e): e.sequenceNumber
        case let .outputTextDelta(e): e.sequenceNumber
        case let .outputTextDone(e): e.sequenceNumber
        case let .functionCallArgumentsDelta(e): e.sequenceNumber
        case let .functionCallArgumentsDone(e): e.sequenceNumber
        case let .reasoningDelta(e): e.sequenceNumber
        case let .reasoningDone(e): e.sequenceNumber
      }
    }
    set {
      switch self {
        case var .responseCreated(e):
          e.sequenceNumber = newValue; self = .responseCreated(e)
        case var .responseInProgress(e):
          e.sequenceNumber = newValue; self = .responseInProgress(e)
        case var .responseCompleted(e):
          e.sequenceNumber = newValue; self = .responseCompleted(e)
        case var .responseIncomplete(e):
          e.sequenceNumber = newValue; self = .responseIncomplete(e)
        case var .outputItemAdded(e):
          e.sequenceNumber = newValue; self = .outputItemAdded(e)
        case var .outputItemDone(e):
          e.sequenceNumber = newValue; self = .outputItemDone(e)
        case var .contentPartAdded(e):
          e.sequenceNumber = newValue; self = .contentPartAdded(e)
        case var .contentPartDone(e):
          e.sequenceNumber = newValue; self = .contentPartDone(e)
        case var .outputTextDelta(e):
          e.sequenceNumber = newValue; self = .outputTextDelta(e)
        case var .outputTextDone(e):
          e.sequenceNumber = newValue; self = .outputTextDone(e)
        case var .functionCallArgumentsDelta(e):
          e.sequenceNumber = newValue; self = .functionCallArgumentsDelta(e)
        case var .functionCallArgumentsDone(e):
          e.sequenceNumber = newValue; self = .functionCallArgumentsDone(e)
        case var .reasoningDelta(e):
          e.sequenceNumber = newValue; self = .reasoningDelta(e)
        case var .reasoningDone(e):
          e.sequenceNumber = newValue; self = .reasoningDone(e)
      }
    }
  }

  package var terminalResponse: Response? {
    switch self {
      case let .responseCompleted(e): e.response
      case let .responseIncomplete(e): e.response
      default: nil
    }
  }
}

// MARK: Lifecycle envelope events

public struct ResponseCreatedEvent: Sendable, Equatable {
  public var response: Response
  public var sequenceNumber: Int

  public init(response: Response, sequenceNumber: Int) {
    self.response = response
    self.sequenceNumber = sequenceNumber
  }
}

public struct ResponseInProgressEvent: Sendable, Equatable {
  public var response: Response
  public var sequenceNumber: Int

  public init(response: Response, sequenceNumber: Int) {
    self.response = response
    self.sequenceNumber = sequenceNumber
  }
}

public struct ResponseCompletedEvent: Sendable, Equatable {
  public var response: Response
  public var sequenceNumber: Int

  public init(response: Response, sequenceNumber: Int) {
    self.response = response
    self.sequenceNumber = sequenceNumber
  }
}

/// Terminal envelope for max-output-token exhaustion.
///
/// Open Responses defines `response.incomplete` as a distinct terminal
/// event. vLLM and SGLang currently collapse this case into
/// `response.completed` with `response.status == incomplete`; Swift emits
/// the Open Responses discriminator and leaves vLLM/SGLang compatibility to
/// decoders where needed.
public struct ResponseIncompleteEvent: Sendable, Equatable {
  public var response: Response
  public var sequenceNumber: Int

  public init(response: Response, sequenceNumber: Int) {
    self.response = response
    self.sequenceNumber = sequenceNumber
  }
}

// MARK: Item lifecycle events

public struct ResponseOutputItemAddedEvent: Sendable, Equatable {
  public var item: ResponseOutputItem
  public var outputIndex: Int
  public var sequenceNumber: Int

  public init(item: ResponseOutputItem, outputIndex: Int, sequenceNumber: Int) {
    self.item = item
    self.outputIndex = outputIndex
    self.sequenceNumber = sequenceNumber
  }
}

public struct ResponseOutputItemDoneEvent: Sendable, Equatable {
  public var item: ResponseOutputItem
  public var outputIndex: Int
  public var sequenceNumber: Int

  public init(item: ResponseOutputItem, outputIndex: Int, sequenceNumber: Int) {
    self.item = item
    self.outputIndex = outputIndex
    self.sequenceNumber = sequenceNumber
  }
}

// MARK: Content-part envelope events

public struct ResponseContentPartAddedEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  public var contentIndex: Int
  public var part: ResponseContentPart
  public var sequenceNumber: Int

  public init(
    itemId: String,
    outputIndex: Int,
    contentIndex: Int,
    part: ResponseContentPart,
    sequenceNumber: Int,
  ) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.contentIndex = contentIndex
    self.part = part
    self.sequenceNumber = sequenceNumber
  }
}

public struct ResponseContentPartDoneEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  public var contentIndex: Int
  public var part: ResponseContentPart
  public var sequenceNumber: Int

  public init(
    itemId: String,
    outputIndex: Int,
    contentIndex: Int,
    part: ResponseContentPart,
    sequenceNumber: Int,
  ) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.contentIndex = contentIndex
    self.part = part
    self.sequenceNumber = sequenceNumber
  }
}

// MARK: Output-text events

/// **`logprobs` is omitted.** The Open Responses spec declares a
/// required `logprobs: Array<...>` field on `response.output_text.delta`
/// and `response.output_text.done`. The parser library has no logprob
/// source — token probabilities live one layer below, in the inference
/// engine. An SSE-proxy that needs strict spec conformance for a fussy
/// client should inject `logprobs: []` when serializing these events.
public struct ResponseTextDeltaEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  public var contentIndex: Int
  public var delta: String
  public var sequenceNumber: Int

  public init(
    itemId: String,
    outputIndex: Int,
    contentIndex: Int,
    delta: String,
    sequenceNumber: Int,
  ) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.contentIndex = contentIndex
    self.delta = delta
    self.sequenceNumber = sequenceNumber
  }
}

public struct ResponseTextDoneEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  public var contentIndex: Int
  public var text: String
  public var sequenceNumber: Int

  public init(
    itemId: String,
    outputIndex: Int,
    contentIndex: Int,
    text: String,
    sequenceNumber: Int,
  ) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.contentIndex = contentIndex
    self.text = text
    self.sequenceNumber = sequenceNumber
  }
}

// MARK: Function-call arguments events

public struct ResponseFunctionCallArgumentsDeltaEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  public var delta: String
  public var sequenceNumber: Int

  public init(itemId: String, outputIndex: Int, delta: String, sequenceNumber: Int) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.delta = delta
    self.sequenceNumber = sequenceNumber
  }
}

public struct ResponseFunctionCallArgumentsDoneEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  // vLLM currently includes a non-spec `name` field on this event. Open
  // Responses carries the function name on the surrounding `function_call`
  // item, so this event intentionally models only the final arguments.
  public var arguments: String
  public var sequenceNumber: Int

  public init(
    itemId: String,
    outputIndex: Int,
    arguments: String,
    sequenceNumber: Int,
  ) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.arguments = arguments
    self.sequenceNumber = sequenceNumber
  }
}

// MARK: Reasoning events

// Open Responses names these stream events `response.reasoning.delta/done`.
// vLLM and SGLang still use the older OpenAI SDK names
// `ResponseReasoningText*Event` and emit `response.reasoning_text.*`;
// CodableEvents accepts those wire aliases, but this source API follows
// the current Open Responses spec.

public struct ResponseReasoningDeltaEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  public var contentIndex: Int
  public var delta: String
  public var sequenceNumber: Int

  public init(
    itemId: String,
    outputIndex: Int,
    contentIndex: Int,
    delta: String,
    sequenceNumber: Int,
  ) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.contentIndex = contentIndex
    self.delta = delta
    self.sequenceNumber = sequenceNumber
  }
}

public struct ResponseReasoningDoneEvent: Sendable, Equatable {
  public var itemId: String
  public var outputIndex: Int
  public var contentIndex: Int
  public var text: String
  public var sequenceNumber: Int

  public init(
    itemId: String,
    outputIndex: Int,
    contentIndex: Int,
    text: String,
    sequenceNumber: Int,
  ) {
    self.itemId = itemId
    self.outputIndex = outputIndex
    self.contentIndex = contentIndex
    self.text = text
    self.sequenceNumber = sequenceNumber
  }
}
