// Copyright © Anthony DePasquale

import Foundation

/// The response-level status carried on the terminal response event.
///
/// The driver derives this from the upstream finish reason: `stop` →
/// `completed`, `length` → `incomplete`, cancellation → `cancelled`. The
/// parser never sets this field – it only knows about item status.
public enum ResponseStatus: String, Sendable, Equatable, CaseIterable {
  case completed
  case failed
  case inProgress = "in_progress"
  case cancelled
  case queued
  case incomplete
}

/// Reason the response is incomplete. The driver emits `maxOutputTokens`
/// for `length` stops. `contentFilter` exists for spec parity but is never
/// produced by a local-device parser.
public enum IncompleteReason: String, Sendable, Equatable, CaseIterable {
  case maxOutputTokens = "max_output_tokens"
  case contentFilter = "content_filter"
}

/// Carried on `Response.incomplete_details` when status is `.incomplete`.
public struct IncompleteDetails: Sendable, Equatable {
  public var reason: IncompleteReason

  public init(reason: IncompleteReason) {
    self.reason = reason
  }
}

/// Token-usage breakdown carried on the terminal response event.
public struct ResponseUsage: Sendable, Equatable {
  public var inputTokens: Int
  public var outputTokens: Int
  public var totalTokens: Int

  public var inputTokensDetails: InputTokensDetails
  public var outputTokensDetails: OutputTokensDetails

  public struct InputTokensDetails: Sendable, Equatable {
    public var cachedTokens: Int

    public init(cachedTokens: Int = 0) {
      self.cachedTokens = cachedTokens
    }
  }

  public struct OutputTokensDetails: Sendable, Equatable {
    public var reasoningTokens: Int

    public init(reasoningTokens: Int = 0) {
      self.reasoningTokens = reasoningTokens
    }
  }

  public init(
    inputTokens: Int,
    outputTokens: Int,
    totalTokens: Int,
    inputTokensDetails: InputTokensDetails = .init(),
    outputTokensDetails: OutputTokensDetails = .init(),
  ) {
    self.inputTokens = inputTokens
    self.outputTokens = outputTokens
    self.totalTokens = totalTokens
    self.inputTokensDetails = inputTokensDetails
    self.outputTokensDetails = outputTokensDetails
  }
}

/// The response-envelope object carried by `response.created`,
/// `response.in_progress`, and terminal response events.
///
/// Only the fields the parser layer can populate from local model output and
/// `ResponseStreamConfig` are kept here. The hosted-API spec defines many more
/// fields (`tool_choice`, `parallel_tool_calls`, `service_tier`,
/// `safety_identifier`, …); those are server-product concerns. `Codable`
/// encoding fills the required Open Responses wire defaults for those fields.
public struct Response: Sendable, Equatable {
  /// Response-scoped ID (`resp_…`). Minted once at construction and reused
  /// on every envelope event.
  public var id: String

  /// Unix timestamp (seconds) of when the response was created.
  public var createdAt: Int

  /// Model identifier (typically the HF repo ID) the response was generated
  /// from. Forwarded from `ResponseStreamConfig.model`.
  public var model: String

  /// Items the parser emitted during the response. On `response.created`
  /// and `response.in_progress` this is empty; on the terminal response
  /// event it contains the accumulated items.
  public var output: [ResponseOutputItem]

  /// Response-level status. `ResponseStream` emits `.inProgress` on
  /// `response.created` and `response.in_progress`, then sets the terminal
  /// status on the terminal response event.
  public var status: ResponseStatus?

  public var incompleteDetails: IncompleteDetails?

  public var usage: ResponseUsage?

  /// Optional system/developer instructions forwarded from the request.
  /// The parser library never sets this; it round-trips verbatim from
  /// ``ResponseStreamConfig/instructions`` so SSE-proxy / Open Responses
  /// consumers get spec-shaped envelopes without writing a wrapper.
  public var instructions: String?

  /// Optional sampling parameters round-tripped from ``ResponseStreamConfig``.
  /// The parser library never populates these — they exist solely so an
  /// SSE-proxy / Open Responses consumer's `Response` matches the OpenAI
  /// spec shape with the request's sampling params echoed back.
  public var temperature: Double?
  public var topP: Double?
  public var maxOutputTokens: Int?

  public init(
    id: String,
    createdAt: Int,
    model: String,
    output: [ResponseOutputItem] = [],
    status: ResponseStatus? = nil,
    incompleteDetails: IncompleteDetails? = nil,
    usage: ResponseUsage? = nil,
    instructions: String? = nil,
    temperature: Double? = nil,
    topP: Double? = nil,
    maxOutputTokens: Int? = nil,
  ) {
    self.id = id
    self.createdAt = createdAt
    self.model = model
    self.output = output
    self.status = status
    self.incompleteDetails = incompleteDetails
    self.usage = usage
    self.instructions = instructions
    self.temperature = temperature
    self.topP = topP
    self.maxOutputTokens = maxOutputTokens
  }
}
