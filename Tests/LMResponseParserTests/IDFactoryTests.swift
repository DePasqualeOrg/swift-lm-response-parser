// Copyright © Anthony DePasquale

@testable import LMResponseParser
import Testing

@Suite("IDFactory")
struct IDFactoryTests {
  @Test
  func `Each prefix produces an ID with the expected shape`() {
    for prefix in IDFactory.Prefix.allCases {
      let id = IDFactory.make(prefix)
      #expect(id.hasPrefix("\(prefix.rawValue)_"), "ID \(id) does not start with \(prefix.rawValue)_")

      let suffix = id.dropFirst(prefix.rawValue.count + 1)
      #expect(suffix.count == IDFactory.suffixLength, "Suffix \(suffix) is not \(IDFactory.suffixLength) chars")

      for char in suffix {
        #expect(
          "0123456789abcdefghjkmnpqrstvwxyz".contains(char),
          "Suffix \(suffix) contains non-alphabet character \(char)",
        )
      }
    }
  }

  @Test
  func `Two consecutive IDs differ`() {
    let a = IDFactory.make(.message)
    let b = IDFactory.make(.message)
    #expect(a != b)
  }

  @Test
  func `Prefix raw values match OpenAI conventions`() {
    #expect(IDFactory.Prefix.message.rawValue == "msg")
    #expect(IDFactory.Prefix.functionCall.rawValue == "fc")
    #expect(IDFactory.Prefix.callId.rawValue == "call")
    #expect(IDFactory.Prefix.reasoning.rawValue == "rs")
    #expect(IDFactory.Prefix.response.rawValue == "resp")
  }

  @Test
  func `makeMistralStrict produces 9 alphanumeric characters with no prefix`() {
    let id = IDFactory.makeMistralStrict()
    #expect(id.count == 9)
    for char in id {
      #expect(
        "0123456789abcdefghjkmnpqrstvwxyz".contains(char),
        "Character \(char) is not in the Crockford-base32 alphabet",
      )
    }
    // The upstream Mistral chat templates enforce
    // `tool_call.id|length != 9`; the prefixed shape would fail.
    #expect(!id.contains("_"))
  }

  @Test
  func `makeMistralStrict produces distinct IDs across consecutive calls`() {
    let a = IDFactory.makeMistralStrict()
    let b = IDFactory.makeMistralStrict()
    #expect(a != b)
  }
}
