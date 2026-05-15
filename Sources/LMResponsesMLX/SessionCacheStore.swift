// Copyright © Anthony DePasquale

import Foundation
import MLXLMCommon

/// A mutex providing exclusive access across `async` blocks.
///
/// Mirrors `MLXLMCommon.SerialAccessContainer`'s private `AsyncMutex`
/// (`SerialAccessContainer.swift:8-37`); ports the same primitive into the
/// bridge package because the upstream is `package`-internal. Normal locks
/// don't work with `async` blocks, and an `actor` doesn't guarantee
/// exclusive access for the duration of an `async` function – every
/// `await` releases the actor's lock.
///
/// File-private to keep the primitive an implementation detail of
/// ``SessionCacheStore``; consumers never see it.
private actor AsyncMutex {
  private var isLocked = false
  private var waiters: [CheckedContinuation<Void, Never>] = []

  private func lock() async {
    if !isLocked {
      isLocked = true
      return
    }
    await withCheckedContinuation { continuation in
      waiters.append(continuation)
    }
  }

  private func unlock() {
    if let next = waiters.first {
      waiters.removeFirst()
      next.resume()
    } else {
      isLocked = false
    }
  }

  func withLock<T>(_ body: () async throws -> sending T) async rethrows -> sending T {
    await lock()
    defer { unlock() }
    return try await body()
  }
}

/// Holds session-scoped state behind an ``AsyncMutex`` so multiple
/// `streamResponseEvents(to:)` calls on the same session serialize
/// end-to-end. Mirrors `MLXLMCommon.SerialAccessContainer<T>`'s `update`
/// shape.
///
/// `final class @unchecked Sendable` is the same compromise the upstream
/// makes: the wrapped state may include non-`Sendable` values (`[KVCache]`
/// itself is not `Sendable`), but exclusive access is guaranteed by the
/// surrounding ``AsyncMutex``.
final class SessionCacheStore<T>: @unchecked Sendable {
  private var value: T
  private let lock = AsyncMutex()

  init(_ value: consuming T) {
    self.value = consume value
  }

  func read<R>(
    _ body: @Sendable (T) async throws -> sending R,
  ) async rethrows -> sending R {
    try await lock.withLock {
      try await body(self.value)
    }
  }

  func update<R>(
    _ body: @Sendable (inout T) async throws -> sending R,
  ) async rethrows -> sending R {
    try await lock.withLock {
      try await body(&self.value)
    }
  }
}
