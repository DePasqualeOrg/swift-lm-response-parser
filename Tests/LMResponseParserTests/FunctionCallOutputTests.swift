// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

@Suite("ResponseFunctionCallOutput — synthetic events round-trip")
struct FunctionCallOutputItemTests {
  @Test
  func `output_item.added followed by output_item.done lands the canonical item`() {
    let id = "fco_test"
    let inProgress = ResponseFunctionCallOutput(
      id: id, callId: "call_abc", output: "", status: .inProgress,
    )
    let completed = ResponseFunctionCallOutput(
      id: id, callId: "call_abc", output: "{\"ok\":true}", status: .completed,
    )
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(
        item: .functionCallOutput(inProgress),
        outputIndex: 0,
        sequenceNumber: 0,
      )),
      .outputItemDone(.init(
        item: .functionCallOutput(completed),
        outputIndex: 0,
        sequenceNumber: 1,
      )),
    ]
    let items = accumulateItems(from: events)
    #expect(items.count == 1)
    guard case let .functionCallOutput(o) = items[0] else {
      Issue.record("Expected functionCallOutput"); return
    }
    #expect(o.id == id)
    #expect(o.callId == "call_abc")
    #expect(o.output == .string("{\"ok\":true}"))
    #expect(o.status == .completed)
  }

  @Test
  func `output_item.added alone leaves the item in_progress`() {
    let id = "fco_partial"
    let item = ResponseFunctionCallOutput(
      id: id, callId: "call_x", output: "", status: .inProgress,
    )
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(
        item: .functionCallOutput(item),
        outputIndex: 0,
        sequenceNumber: 0,
      )),
    ]
    let items = accumulateItems(from: events)
    guard case let .functionCallOutput(o) = items[0] else {
      Issue.record("Expected functionCallOutput"); return
    }
    #expect(o.status == .inProgress)
  }

  @Test
  func `ResponseOutputItem.id returns the function-call-output id`() {
    let item = ResponseOutputItem.functionCallOutput(
      .init(id: "fco_xyz", callId: "call_xyz", output: "result"),
    )
    #expect(item.id == "fco_xyz")
  }

  @Test
  func `function-call-output can carry typed content`() {
    let output = ResponseFunctionCallOutput.Output.content([
      .inputText(.init(text: "Chart generated.")),
      .inputImage(.init(imageURL: "file:///tmp/chart.png", detail: .high)),
      .inputFile(.init(filename: "report.csv", fileURL: "file:///tmp/report.csv")),
    ])
    let item = ResponseFunctionCallOutput(
      id: "fco_content",
      callId: "call_content",
      output: output,
    )

    #expect(item.output == output)
    #expect(item.output.stringValue == nil)
  }

  @Test
  func `Stray content_part.added on a function-call-output item is a no-op`() {
    let id = "fco_test"
    let item = ResponseFunctionCallOutput(
      id: id, callId: "call_x", output: "ok", status: .completed,
    )
    let events: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(
        item: .functionCallOutput(item),
        outputIndex: 0,
        sequenceNumber: 0,
      )),
      .contentPartAdded(.init(
        itemId: id, outputIndex: 0, contentIndex: 0,
        part: .outputText(.init(text: "")),
        sequenceNumber: 1,
      )),
    ]
    let items = accumulateItems(from: events)
    // Item type should remain functionCallOutput; appendPart is a no-op
    // for this case so the slot is unchanged.
    guard case .functionCallOutput = items[0] else {
      Issue.record("Expected functionCallOutput"); return
    }
  }

  @Test
  func `ResponseItemsAccumulator handles function-call-output items across chunks`() {
    let id = "fco_x"
    let inProgress = ResponseFunctionCallOutput(
      id: id, callId: "call_q", output: "", status: .inProgress,
    )
    let completed = ResponseFunctionCallOutput(
      id: id, callId: "call_q", output: "42", status: .completed,
    )
    let chunkA: [ResponseStreamingEvent] = [
      .outputItemAdded(.init(
        item: .functionCallOutput(inProgress),
        outputIndex: 0,
        sequenceNumber: 0,
      )),
    ]
    let chunkB: [ResponseStreamingEvent] = [
      .outputItemDone(.init(
        item: .functionCallOutput(completed),
        outputIndex: 0,
        sequenceNumber: 1,
      )),
    ]

    var accumulator = ResponseItemsAccumulator()
    accumulator.ingest(chunkA)
    guard case let .functionCallOutput(mid) = accumulator.items[0] else {
      Issue.record("Expected functionCallOutput after chunk A"); return
    }
    #expect(mid.status == .inProgress)

    accumulator.ingest(chunkB)
    guard case let .functionCallOutput(final) = accumulator.items[0] else {
      Issue.record("Expected functionCallOutput after chunk B"); return
    }
    #expect(final.status == .completed)
    #expect(final.output == .string("42"))
  }
}

@Suite("IDFactory.Prefix.functionCallOutput")
struct FunctionCallOutputIDFactoryTests {
  @Test
  func `Prefix raw value is fco`() {
    #expect(IDFactory.Prefix.functionCallOutput.rawValue == "fco")
  }

  @Test
  func `make(.functionCallOutput) produces fco_… IDs`() {
    let id = IDFactory.make(.functionCallOutput)
    #expect(id.hasPrefix("fco_"))
    let suffix = id.dropFirst("fco_".count)
    #expect(suffix.count == IDFactory.suffixLength)
  }
}

@Suite("ResponseFormat.stopTokenPolicy")
struct StopTokenPolicyTests {
  @Test
  func `Harmony declares <|call|> and <|return|> as included stop tokens`() {
    let policy = ResponseFormat.harmony.stopTokenPolicy
    #expect(policy.includedStopTokens == ["<|call|>", "<|return|>"])
    #expect(policy.requiredExtraEOSTokens == ["<|call|>", "<|return|>"])
    #expect(policy.includeStopToken == true)
  }

  @Test
  func `Every non-Harmony format declares an empty stop-token policy`() {
    for format in ResponseFormat.allCases where format != .harmony {
      let policy = format.stopTokenPolicy
      #expect(policy.includedStopTokens.isEmpty, "Expected empty includedStopTokens for \(format)")
      #expect(policy.requiredExtraEOSTokens.isEmpty, "Expected empty requiredExtraEOSTokens for \(format)")
      #expect(policy.includeStopToken == false, "Expected includeStopToken == false for \(format)")
    }
  }

  @Test
  func `Every format reports a deterministic stop-token policy`() {
    for format in ResponseFormat.allCases {
      let a = format.stopTokenPolicy
      let b = format.stopTokenPolicy
      #expect(a == b)
    }
  }
}
