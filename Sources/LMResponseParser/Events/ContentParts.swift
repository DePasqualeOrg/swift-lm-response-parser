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

  /// Annotations on the text. We do not currently emit any, but the field
  /// is structurally present so the spec shape is preserved.
  public var annotations: [Annotation]

  /// Annotation variants the spec defines. We don't synthesize any of these
  /// from local model output, so the array stays empty in practice.
  public enum Annotation: Sendable, Equatable {
    case fileCitation(fileId: String, filename: String, index: Int)
    case urlCitation(url: String, title: String?, startIndex: Int, endIndex: Int)
    case filePath(fileId: String, index: Int)
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
