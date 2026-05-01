// Copyright © Anthony DePasquale

/// Mirrors `MLXLMCommon.SendableBox` (`SerialAccessContainer.swift:104-118`)
/// – `package`-internal upstream, so the bridge ports it. Used to ferry
/// non-`Sendable` values (chiefly ``MLXLMCommon/ModelContext``) across task
/// boundaries when we know they are only consumed once and stay within one
/// task scope.
///
/// Internal because nothing on the bridge's public surface returns one –
/// the `ModelContainer`-based helper overloads use this internally to hide
/// the perform/box dance from consumers.
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
