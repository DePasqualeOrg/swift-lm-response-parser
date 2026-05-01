// Copyright © Anthony DePasquale

import Foundation

/// A piece of content inside an assistant message or reasoning item.
///
/// The `reasoning_text` part is emitted inside the same
/// `response.content_part.added/done` envelope as message text parts –
/// reasoning items are not given their own envelope event type. This is
/// what canonical client SDKs accumulate against; emitting a separate
/// `reasoning_part` envelope would break their snapshot logic.
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
/// parts. The dedicated `response.reasoning_text.delta/done` events stream
/// the actual deltas inside that envelope.
public struct ReasoningTextContent: Sendable, Equatable {
  public var text: String

  public init(text: String) {
    self.text = text
  }
}
