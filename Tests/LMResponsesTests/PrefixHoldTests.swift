// Copyright © Anthony DePasquale

@testable import LMResponses
import Testing

@Suite("PrefixHold")
struct PrefixHoldTests {
  @Test
  func `Empty buffer emits nothing and holds nothing`() {
    let (emit, hold) = PrefixHold.split(text: "", markers: ["<tool_call>"])
    #expect(emit == "")
    #expect(hold == "")
  }

  @Test
  func `Buffer with no marker overlap is fully emitted`() {
    let (emit, hold) = PrefixHold.split(text: "hello world", markers: ["<tool_call>"])
    #expect(emit == "hello world")
    #expect(hold == "")
  }

  @Test
  func `Partial marker suffix is held back`() {
    let (emit, hold) = PrefixHold.split(text: "abc <thi", markers: ["<think>"])
    #expect(emit == "abc ")
    #expect(hold == "<thi")
  }

  @Test
  func `Single character that opens a marker is held`() {
    let (emit, hold) = PrefixHold.split(text: "abc <", markers: ["<think>"])
    #expect(emit == "abc ")
    #expect(hold == "<")
  }

  @Test
  func `Complete marker is not held — that is the parser's job`() {
    let (emit, hold) = PrefixHold.split(text: "abc <think>", markers: ["<think>"])
    #expect(emit == "abc <think>")
    #expect(hold == "")
  }

  @Test
  func `Longest overlap across multiple markers wins`() {
    let (emit, hold) = PrefixHold.split(text: "x<tool_ca", markers: ["<think>", "<tool_call>"])
    #expect(emit == "x")
    #expect(hold == "<tool_ca")
  }

  @Test
  func `Buffer shorter than marker is fully held when it matches a prefix`() {
    let (emit, hold) = PrefixHold.split(text: "<th", markers: ["<think>"])
    #expect(emit == "")
    #expect(hold == "<th")
  }

  @Test
  func `Empty markers list emits everything`() {
    let (emit, hold) = PrefixHold.split(text: "<think>any", markers: [])
    #expect(emit == "<think>any")
    #expect(hold == "")
  }

  @Test
  func `Empty marker entries are ignored without crashing`() {
    let (emit, hold) = PrefixHold.split(text: "<thi", markers: ["", "<think>"])
    #expect(emit == "")
    #expect(hold == "<thi")
  }

  @Test
  func `Nested-prefix markers: short marker with prefix-of-longer holds the longer`() {
    // "abc" is a complete marker AND a strict prefix of "abcd". When
    // the buffer is exactly "abc", the next chunk could extend it into
    // the longer marker "abcd", so PrefixHold must hold all of "abc".
    let (emit, hold) = PrefixHold.split(text: "abc", markers: ["abc", "abcd"])
    #expect(emit == "")
    #expect(hold == "abc")
  }

  @Test
  func `Nested-prefix markers: extra char rules out the longer marker`() {
    // After more bytes arrive, "abc " can't grow into "abcd" — so we
    // emit and let the per-format parser consume the complete "abc".
    let (emit, hold) = PrefixHold.split(text: "abc ", markers: ["abc", "abcd"])
    #expect(emit == "abc ")
    #expect(hold == "")
  }
}
