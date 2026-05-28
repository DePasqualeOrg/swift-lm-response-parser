// Copyright © Anthony DePasquale

import Foundation

/// Role of an input chat message. Engine-agnostic so the llama and MLX
/// session APIs share one `role:` type. Raw values match both the chat
/// template convention (`"user"`, `"system"`, …) and MLX's
/// `Chat.Message.Role`, so bridges convert with `rawValue`/`init(rawValue:)`.
public enum Role: String, Sendable, Equatable, CaseIterable {
  case system
  case user
  case assistant
  case tool
}
