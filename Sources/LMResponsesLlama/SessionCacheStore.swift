// Copyright © Anthony DePasquale

import Foundation

/// A mutex providing exclusive access across `async` blocks. Normal locks
/// don't work with `async` blocks, and an `actor` doesn't guarantee
/// exclusive access for the duration of an `async` function – every
/// `await` releases the actor's lock.
///
/// Honors task cancellation: a task cancelled while waiting for the
/// lock throws `CancellationError` from `lock()` and is removed from
/// the queue immediately, instead of sitting until eventually woken
/// and running an unwanted body.
///
/// File-private to keep the primitive an implementation detail of
/// ``SessionCacheStore``.
private actor AsyncMutex {
  private var isLocked = false
  // Keyed by ID so the cancellation handler can target the specific
  // waiter to remove. Using a dictionary + sorted-key FIFO order
  // because `removeValue(forKey:)` from arbitrary positions is what
  // cancellation needs; the sort cost is negligible at the contention
  // levels session locks see.
  private var waiters: [Int: CheckedContinuation<Void, Error>] = [:]
  private var nextWaiterID = 0

  private func lock() async throws {
    // No upfront `Task.checkCancellation()`: an already-cancelled
    // task that finds the lock free should still be able to take it
    // and run cleanup work. Sessions rely on this — the catch block
    // after a cancelled `runTurn` re-acquires the lock to clear the
    // underlying KV; throwing upfront here would leak stale KV
    // across the cancellation boundary.
    if !isLocked {
      isLocked = true
      return
    }
    let id = nextWaiterID
    nextWaiterID &+= 1
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        // Actor isolation guarantees this insertion completes before
        // cancelWaiter (also actor-isolated) can run, so the cancel
        // handler can't miss the waiter it's trying to remove.
        waiters[id] = continuation
      }
    } onCancel: {
      Task { [weak self] in await self?.cancelWaiter(id: id) }
    }
  }

  private func cancelWaiter(id: Int) {
    if let cont = waiters.removeValue(forKey: id) {
      cont.resume(throwing: CancellationError())
    }
  }

  private func unlock() {
    // FIFO via the lowest outstanding waiter ID.
    if let id = waiters.keys.min() {
      let cont = waiters.removeValue(forKey: id)!
      cont.resume()
    } else {
      isLocked = false
    }
  }

  func withLock<T>(_ body: () async throws -> sending T) async throws -> sending T {
    try await lock()
    defer { unlock() }
    return try await body()
  }
}

/// Holds session-scoped state behind an ``AsyncMutex`` so multiple
/// `streamResponseEvents(to:)` calls on the same session serialize
/// end-to-end.
final class SessionCacheStore<T>: @unchecked Sendable {
  private var value: T
  private let lock = AsyncMutex()

  init(_ value: consuming T) {
    self.value = consume value
  }

  func read<R>(
    _ body: @Sendable (T) async throws -> sending R,
  ) async throws -> sending R {
    try await lock.withLock {
      try await body(self.value)
    }
  }

  func update<R>(
    _ body: @Sendable (inout T) async throws -> sending R,
  ) async throws -> sending R {
    try await lock.withLock {
      try await body(&self.value)
    }
  }
}
