// Copyright © Anthony DePasquale

import Foundation

/// One-shot parser: feed the complete text in one call, finalize, and return
/// the accumulated items. For per-chunk streaming, use ``ResponseStream``.
public func parseResponse(
  _ text: String,
  format: ResponseFormat,
  tokenizer: any ResponseTokenizer,
  tools: [ToolSpec] = [],
) -> [ResponseOutputItem] {
  var parser = format.makeParser(tokenizer: tokenizer, tools: tools)
  var events = parser.process(ParserInput(text: text))
  events += parser.finalize()
  return accumulateItems(from: events)
}

/// Walk a stream of spec events and produce the items they describe.
///
/// One-shot convenience over the package-level `ResponseItemsAccumulator`:
/// builds a fresh accumulator, ingests every event, and returns its items.
/// Use this when you have the complete event array; for incremental
/// streaming, hold a `ResponseItemsAccumulator` across chunks instead so
/// each token is O(events-in-chunk) rather than O(events-so-far).
public func accumulateItems(from events: [ResponseStreamingEvent]) -> [ResponseOutputItem] {
  var accumulator = ResponseItemsAccumulator()
  accumulator.ingest(events)
  return accumulator.items
}

// MARK: Accumulator

/// Stateful accumulator that backs ``ResponseStream/items`` and
/// ``accumulateItems(from:)``. Holds the live ``ResponseOutputItem`` snapshot
/// as events arrive, applying each event in place so per-chunk work is
/// O(events-in-chunk) instead of the O(events-so-far) cost of re-walking the
/// full event history.
///
/// Cross-SDK precedent for the snapshot-mutating pattern: OpenAI TS SDK's
/// `ResponseStream.#accumulateResponse` keeps a `#currentResponseSnapshot`
/// and mutates `output[]` / `content[].text` / `arguments` in place; Vercel
/// AI SDK's `stream-text.ts` keeps `recordedContent` plus
/// `activeTextContent` / `activeReasoningContent` maps and appends deltas
/// directly. vLLM's `parse_streaming_increment` and sglang's analog return
/// per-delta classified output instead, which fits a different consumer
/// shape (delta sinks rather than snapshot consumers).
package struct ResponseItemsAccumulator {
  /// Cumulative snapshot at the current point in the stream. Items whose
  /// matching `output_item.done` has not yet arrived appear with their
  /// in-progress status – consumers that don't want to render partial
  /// state should filter on `status` themselves. Cheap to read on every
  /// access (Swift's COW makes this a retain, not a deep copy).
  package private(set) var items: [ResponseOutputItem] = []

  package init() {}

  /// Apply a single event to the cumulative snapshot. Events must be
  /// ingested in the order the parser emitted them; reordering (e.g.
  /// processing an `output_item.done` before its matching `added`) will
  /// produce incorrect output.
  package mutating func ingest(_ event: ResponseStreamingEvent) {
    applyEvent(event, to: &items)
  }

  /// Apply a chunk of events in order. Equivalent to calling
  /// ``ingest(_:)-(ResponseStreamingEvent)`` for each element.
  package mutating func ingest(_ events: some Sequence<ResponseStreamingEvent>) {
    for event in events {
      applyEvent(event, to: &items)
    }
  }
}

// MARK: Event application

/// Apply a single event to a mutable items array. Shared between
/// ``accumulateItems(from:)`` and ``ResponseItemsAccumulator``.
///
/// Items are built up from `output_item.added` + delta events, and each slot
/// is replaced with the canonical item on `output_item.done`. Items that
/// never see a matching `output_item.done` retain whatever status they had
/// at the last update – this helper does not synthesize a closing event for
/// them.
private func applyEvent(_ event: ResponseStreamingEvent, to items: inout [ResponseOutputItem]) {
  switch event {
    case let .outputItemAdded(e):
      _ = ensureSlot(in: &items, at: e.outputIndex, with: e.item)

    case let .outputItemDone(e):
      guard ensureSlot(in: &items, at: e.outputIndex, with: e.item) else { return }
      items[e.outputIndex] = e.item

    case let .contentPartAdded(e):
      guard e.outputIndex < items.count else { return }
      appendPart(e.part, to: &items[e.outputIndex])

    case let .outputTextDelta(e):
      guard e.outputIndex < items.count else { return }
      appendText(e.delta, atContentIndex: e.contentIndex, to: &items[e.outputIndex])

    case let .reasoningDelta(e):
      guard e.outputIndex < items.count else { return }
      appendReasoningText(e.delta, atContentIndex: e.contentIndex, to: &items[e.outputIndex])

    case let .functionCallArgumentsDelta(e):
      guard e.outputIndex < items.count else { return }
      appendArguments(e.delta, to: &items[e.outputIndex])

    case .contentPartDone, .outputTextDone, .reasoningDone,
         .functionCallArgumentsDone, .responseCreated, .responseInProgress, .responseCompleted,
         .responseIncomplete:
      // The done events for inner content carry the final accumulated
      // value but it's already been built up via deltas; matching
      // `output_item.done` will overwrite the whole slot anyway.
      // Lifecycle envelope events do not carry per-item updates.
      break
  }
}

@discardableResult
private func ensureSlot(
  in items: inout [ResponseOutputItem],
  at index: Int,
  with item: ResponseOutputItem,
) -> Bool {
  if index < items.count {
    return true
  }
  // The spec emits items in increasing-output-index order, so a gap would
  // be a parser bug. Drop the invalid event rather than inventing placeholder
  // items that never existed in the stream.
  guard index == items.count else {
    return false
  }
  items.append(item)
  return true
}

private func appendPart(_ part: ResponseContentPart, to item: inout ResponseOutputItem) {
  switch item {
    case var .message(m):
      m.content.append(part)
      item = .message(m)
    case var .reasoning(r):
      r.content.append(part)
      item = .reasoning(r)
    case .functionCall:
      // Function calls do not carry content parts; arguments live on the
      // item itself.
      return
    case .functionCallOutput:
      // Function-call-output items carry the result string on the item
      // itself, with no content parts.
      return
  }
}

private func appendText(
  _ delta: String,
  atContentIndex contentIndex: Int,
  to item: inout ResponseOutputItem,
) {
  guard case var .message(m) = item, contentIndex < m.content.count else { return }
  if case var .outputText(t) = m.content[contentIndex] {
    t.text += delta
    m.content[contentIndex] = .outputText(t)
    item = .message(m)
  }
}

private func appendReasoningText(
  _ delta: String,
  atContentIndex contentIndex: Int,
  to item: inout ResponseOutputItem,
) {
  guard case var .reasoning(r) = item, contentIndex < r.content.count else { return }
  if case var .reasoningText(t) = r.content[contentIndex] {
    t.text += delta
    r.content[contentIndex] = .reasoningText(t)
    item = .reasoning(r)
  }
}

private func appendArguments(_ delta: String, to item: inout ResponseOutputItem) {
  guard case var .functionCall(f) = item else { return }
  f.arguments += delta
  item = .functionCall(f)
}
