// Copyright © Anthony DePasquale

import Foundation

/// Per-format streaming parser for one model's response.
///
/// The protocol is intentionally minimal: two mutating methods, both
/// returning arrays of spec-shaped streaming events. The parser is stateful
/// within a response and stateless across responses; each response gets a
/// fresh instance, constructed via
/// ``ResponseFormat/makeParser(tokenizer:tools:priorOutput:)``.
///
/// **Sequence numbers.** Events emitted by the parser carry parser-local
/// sequence numbers (zero-based per parser instance). The driver layer
/// (``ResponseStream``) substitutes response-scoped sequence numbers
/// before yielding to consumers, so the parser does not need to know about
/// the lifecycle envelope events.
///
/// **Item IDs.** The parser is responsible for minting item-scoped IDs
/// (``IDFactory/Prefix/message``, ``IDFactory/Prefix/functionCall`` plus a
/// separate ``IDFactory/Prefix/callId`` for the function call's `call_id`,
/// ``IDFactory/Prefix/reasoning``) at item-open time and reusing them on
/// every subsequent event for that item. Item IDs cannot be deferred to a
/// downstream layer because the spec requires every item event to carry
/// `item_id` from the moment it is emitted.
///
/// **Validation boundary.** Parsers are best-effort structural
/// extractors. They recognize a model family's wire format, emit
/// Responses-shaped items, and mark truncated structures as
/// ``ItemStatus/incomplete``. They do not enforce tool allow-lists,
/// permissions, or application-specific argument constraints. Some
/// parsers consult `ToolSpec` schemas to coerce untyped wire values into
/// JSON argument strings; that is format normalization, not semantic
/// validation. The host dispatch layer decides whether a completed
/// function call is executable and how recoverable validation failures are
/// reported back to the model.
package protocol ResponseFormatParser: Sendable {
  /// Process a chunk of model output. Returns any spec streaming events
  /// produced by this chunk; may be empty if the chunk is mid-marker or
  /// otherwise produces no complete events yet.
  mutating func process(_ chunk: ParserInput) -> [ResponseStreamingEvent]

  /// Signal end of input. Flushes any held-back partial-marker bytes
  /// (EOS proves they were plain content rather than the start of a guard
  /// token) and emits closing events for any open items. Items that
  /// reached a clean close per the format's state machine get
  /// ``ItemStatus/completed``; truncated items get ``ItemStatus/incomplete``.
  ///
  /// **Lifecycle contract.** Drivers may skip ``finalize()`` on
  /// abnormal exit paths (consumer cancellation, error throw mid-pass).
  /// Parsers that hold non-trivial resources (file handles, network
  /// connections, large buffers) must release them via `deinit`,
  /// not via ``finalize()``.
  mutating func finalize() -> [ResponseStreamingEvent]
}
