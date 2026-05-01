// Copyright © Anthony DePasquale

import LMResponseParserMLX
import Testing

// No `import LMResponseParser` — re-export should expose these.

@Suite("Re-export check")
struct ReexportCheck {
  @Test
  func `Parser-library symbols are visible via LMResponseParserMLX re-export`() {
    let _: ResponseFormat = .json
    let _: IDFactory.Prefix = .functionCallOutput
    let _: ResponseFunctionCallOutput = .init(id: "x", callId: "y", output: "z")
    let _: ResponseStreamingEvent? = nil
    #expect(ResponseFormat.harmony.stopTokenPolicy.includeStopToken == true)
  }
}
