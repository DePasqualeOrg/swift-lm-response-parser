// Copyright © Anthony DePasquale

@testable import LMResponsesLlama
import Testing

@Test func `re exports LM responses symbols`() {
  // Smoke test that the @_exported import surfaces parser-library types.
  _ = ResponseFormat.json
}
