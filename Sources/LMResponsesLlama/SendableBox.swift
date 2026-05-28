// Copyright © Anthony DePasquale

/// Ferries non-`Sendable` values across task boundaries when they are
/// consumed exactly once and stay within one task scope. Ported rather
/// than imported because `MLXLMCommon`'s equivalent is `package`-internal.
final class SendableBox<T>: @unchecked Sendable {
  private var value: T?

  init(_ value: consuming T) {
    self.value = consume value
  }

  consuming func consume() -> T {
    guard let value else {
      fatalError("SendableBox value already consumed")
    }
    self.value = nil
    return value
  }
}
