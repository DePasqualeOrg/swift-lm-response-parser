// Copyright © Anthony DePasquale

import Foundation

/// A piece of content inside an assistant message or reasoning item.
///
/// The `reasoning_text` part is emitted inside the same
/// `response.content_part.added/done` envelope as message text parts –
/// reasoning items are not given their own envelope event type. This is
/// what Open Responses specifies. vLLM currently emits a non-spec
/// `response.reasoning_part.*` envelope for this shape; this library keeps
/// the Open Responses envelope so canonical client SDKs accumulate the
/// snapshot correctly.
public enum ResponseContentPart: Sendable, Equatable {
  case outputText(ResponseOutputText)
  case refusal(ResponseOutputRefusal)
  case reasoningText(ReasoningTextContent)
}

/// A text output part inside a message's content.
public struct ResponseOutputText: Sendable, Equatable {
  /// The text content.
  public var text: String

  /// Annotations on the text. Cohere's grounded-answer parser emits one
  /// ``Annotation/cohereToolResultCitation(toolCallIndex:toolResultIndices:startIndex:endIndex:)``
  /// per `<co>…</co: …>` citation span; the spec-defined variants
  /// (`file_citation`, `url_citation`, `file_path`) are reserved for
  /// hosted-product surfaces that do not currently produce output here.
  public var annotations: [Annotation]

  /// Annotation variants the spec defines, plus the Cohere extension
  /// declared via the spec's `provider:slug` extension policy. The
  /// spec-native variants stay empty in practice; the Cohere variant is
  /// populated by the Cohere parser.
  public enum Annotation: Sendable, Equatable {
    case fileCitation(fileId: String, filename: String, index: Int)
    case urlCitation(url: String, title: String?, startIndex: Int, endIndex: Int)
    case filePath(fileId: String, index: Int)
    /// Cohere grounded-answer citation. Mirrors melody's
    /// `FilterCitation` / `Source` decomposition: one annotation per
    /// `(tool_call_index, tool_result_indices)` group, with character
    /// indices given in UTF-16 code units to match OpenAI's
    /// ``Annotation/urlCitation(url:title:startIndex:endIndex:)``
    /// convention. Discriminator string is
    /// `cohere:tool_result_citation`.
    case cohereToolResultCitation(
      toolCallIndex: Int,
      toolResultIndices: [Int],
      startIndex: Int,
      endIndex: Int,
    )
  }

  public init(text: String, annotations: [Annotation] = []) {
    self.text = text
    self.annotations = annotations
  }
}

/// A refusal part inside a message's content.
///
/// Local models do not emit a structured refusal channel, so this variant is
/// provided for spec completeness only – no parser currently produces one.
public struct ResponseOutputRefusal: Sendable, Equatable {
  public var refusal: String

  public init(refusal: String) {
    self.refusal = refusal
  }
}

/// A reasoning text part inside a reasoning item's content.
///
/// Wrapped in the same `content_part.added/done` envelope as message text
/// parts. The dedicated `response.reasoning.delta/done` events stream
/// the actual deltas inside that envelope. This intentionally differs from
/// vLLM's `response.reasoning_part.*` helper events in favor of the Open
/// Responses event union.
public struct ReasoningTextContent: Sendable, Equatable {
  public var text: String

  public init(text: String) {
    self.text = text
  }
}
