// Copyright © Anthony DePasquale

import Foundation

/// Request-scoped fields the emitter needs to construct the response
/// envelope events. The consumer fills this in from the request that drove
/// generation; the parser layer has no business holding request state
/// itself.
///
/// All fields except `model` are optional. The `model` field is required
/// because every Responses API event carries it on the `Response` envelope.
public struct ResponseStreamConfig: Sendable {
  /// Model identifier – typically the HF repo ID – that produced the
  /// response. Carried on the `Response` envelope as `model`.
  public var model: String

  /// Optional system/developer instructions text from the request. Carried
  /// on the `Response.instructions` field.
  public var instructions: String?

  /// Optional sampling parameters forwarded from the request.
  public var temperature: Double?
  public var topP: Double?
  public var maxOutputTokens: Int?

  /// Tools the model was given. Held here so a server layer wrapping the
  /// emitter can echo them back on the envelope; not used by the emitter
  /// directly. Tools are supplied to per-format parsers separately at
  /// parser construction time.
  public var tools: [ToolSpec]

  /// Override for the `created_at` timestamp. nil means "use the wall clock
  /// at `start()` time" – set this only for deterministic test fixtures.
  public var createdAt: Int?

  public init(
    model: String,
    instructions: String? = nil,
    temperature: Double? = nil,
    topP: Double? = nil,
    maxOutputTokens: Int? = nil,
    tools: [ToolSpec] = [],
    createdAt: Int? = nil,
  ) {
    self.model = model
    self.instructions = instructions
    self.temperature = temperature
    self.topP = topP
    self.maxOutputTokens = maxOutputTokens
    self.tools = tools
    self.createdAt = createdAt
  }
}
