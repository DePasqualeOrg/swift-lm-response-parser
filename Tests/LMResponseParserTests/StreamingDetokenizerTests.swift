// Copyright © Anthony DePasquale

import Foundation
@testable import LMResponseParser
import Testing

/// Backing store for ``MockTokenizer``: an ordered list of byte sequences
/// keyed by their integer token id. Decoding concatenates the bytes for the
/// requested ids and returns them as a UTF-8 string, allowing tests to
/// construct streams that fragment multi-byte scalars across consecutive
/// tokens.
private struct MockVocabulary {
  let bytesByTokenId: [Int: [UInt8]]
  let stringByTokenId: [Int: String]

  init(_ entries: [Int: [UInt8]]) {
    bytesByTokenId = entries
    stringByTokenId = entries.mapValues { String(decoding: $0, as: UTF8.self) }
  }
}

/// Test tokenizer whose `decode` is byte-prefix-monotonic by construction:
/// concatenating the bytes of any ids prefix produces the same bytes as the
/// full sequence's prefix. Useful for reaching every branch of the streaming
/// algorithm without relying on a real model.
private struct MockTokenizer: ParserTokenizer {
  let vocabulary: MockVocabulary
  let invalidIds: Set<Int>

  init(_ entries: [Int: [UInt8]], invalidIds: Set<Int> = []) {
    vocabulary = MockVocabulary(entries)
    self.invalidIds = invalidIds
  }

  private struct UnknownTokenId: Error { let id: Int }
  private struct InjectedFailure: Error {}

  func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
    []
  }

  func decode(tokenIds: [Int], skipSpecialTokens _: Bool) throws -> String {
    var bytes: [UInt8] = []
    for id in tokenIds {
      if invalidIds.contains(id) {
        throw InjectedFailure()
      }
      guard let chunk = vocabulary.bytesByTokenId[id] else {
        throw UnknownTokenId(id: id)
      }
      bytes.append(contentsOf: chunk)
    }
    return String(decoding: bytes, as: UTF8.self)
  }

  func convertTokenToId(_: String) -> Int? {
    nil
  }
}

@Suite("StreamingDetokenizer")
struct StreamingDetokenizerSuite {
  /// "Hello world" split into single-byte ASCII tokens.
  private static let asciiTokenizer = MockTokenizer([
    1: Array("H".utf8),
    2: Array("e".utf8),
    3: Array("l".utf8),
    4: Array("o".utf8),
    5: Array(" ".utf8),
    6: Array("w".utf8),
    7: Array("r".utf8),
    8: Array("d".utf8),
  ])

  /// 🌍 (U+1F30D, four UTF-8 bytes) split byte-by-byte across tokens 10–13.
  private static let fragmentedEmojiTokenizer = MockTokenizer([
    10: [0xF0],
    11: [0x9F],
    12: [0x8C],
    13: [0x8D],
    20: Array("!".utf8),
  ])

  @Test
  func `emits each ascii token`() throws {
    let stream = Self.asciiTokenizer.streamingDetokenizer()
    let ids = [1, 2, 3, 3, 4] // "Hello"
    var out = ""
    for id in ids {
      if let chunk = try stream.consume(id) {
        out.append(chunk)
      }
    }
    #expect(out == "Hello")
  }

  @Test
  func `multi byte scalar buffered across tokens`() throws {
    let stream = Self.fragmentedEmojiTokenizer.streamingDetokenizer()
    var collected: [String?] = []
    for id in [10, 11, 12, 13] {
      try collected.append(stream.consume(id))
    }
    // The first three consumes return nil because the buffered bytes still
    // form an incomplete scalar; the fourth completes the scalar and emits.
    #expect(collected[0] == nil)
    #expect(collected[1] == nil)
    #expect(collected[2] == nil)
    #expect(collected[3] == "🌍")
  }

  @Test
  func `buffer stays bounded across long stream`() throws {
    let helloIds = [1, 2, 3, 3, 4, 5, 6, 4, 7, 3, 8] // "Hello world"
    let ids = Array(repeating: helloIds, count: 1000).flatMap { $0 }
    let stream = Self.asciiTokenizer.streamingDetokenizer()

    var output = ""
    for id in ids {
      if let chunk = try stream.consume(id) {
        output.append(chunk)
      }
    }
    let oneShot = try Self.asciiTokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
    #expect(output == oneShot)

    let mirror = Mirror(reflecting: stream)
    let storedIds = mirror.children.first { $0.label == "ids" }?.value as? [Int]
    #expect(storedIds != nil)
    if let storedIds {
      #expect(storedIds.count < 5)
    }
  }

  @Test
  func `consume batch returns concatenation`() throws {
    let stream = Self.asciiTokenizer.streamingDetokenizer()
    let chunk = try stream.consume([1, 2, 3, 3, 4])
    #expect(chunk == "Hello")
  }

  @Test
  func `seeding from initial ids`() throws {
    let stream = Self.asciiTokenizer.streamingDetokenizer(initialTokenIds: [1, 2])
    // The first consume should establish the prefix for the seeded ids
    // and emit only the new token's bytes.
    let chunk = try stream.consume(3)
    #expect(chunk == "l")
  }

  @Test
  func `consume rolls back when decode throws for unknown token`() throws {
    // When the tokenizer rejects the just-appended id, `consume`
    // must not mutate any internal state — the next valid id has
    // to behave as if the rejected one had never been fed.
    let tokenizer = MockTokenizer(
      [
        1: Array("Hi".utf8),
        2: Array(" ".utf8),
        3: Array("there".utf8),
      ],
      invalidIds: [99],
    )
    let stream = tokenizer.streamingDetokenizer()

    _ = try stream.consume(1)

    let beforeMirror = Mirror(reflecting: stream)
    let beforeIds = beforeMirror.children.first { $0.label == "ids" }?.value as? [Int]
    let beforePrefix = beforeMirror.children.first { $0.label == "prefix" }?.value as? String
    let beforeIndex = beforeMirror.children.first { $0.label == "prefixIndex" }?.value as? Int
    #expect(beforeIds != nil && beforePrefix != nil && beforeIndex != nil)

    do {
      _ = try stream.consume(99)
      Issue.record("Expected decode failure")
    } catch {
      // Expected.
    }

    let afterMirror = Mirror(reflecting: stream)
    let afterIds = afterMirror.children.first { $0.label == "ids" }?.value as? [Int]
    let afterPrefix = afterMirror.children.first { $0.label == "prefix" }?.value as? String
    let afterIndex = afterMirror.children.first { $0.label == "prefixIndex" }?.value as? Int

    #expect(beforeIds == afterIds)
    #expect(beforePrefix == afterPrefix)
    #expect(beforeIndex == afterIndex)

    // Subsequent valid consumes still produce the right text — i.e. the
    // poison id was fully rolled back, not buffered.
    let chunk2 = try stream.consume(2)
    let chunk3 = try stream.consume(3)
    let combined = (chunk2 ?? "") + (chunk3 ?? "")
    #expect(combined == " there")
  }

  @Test
  func `consume rolls back when later decode throws`() throws {
    // The algorithm makes more than one `decode` call per
    // `consume`: one to compute the working text and another to
    // refresh the cached prefix after a successful chunk is
    // emitted. This mock makes the first decode succeed and the
    // second one throw — the rollback property must hold for
    // either failure point, leaving `ids`, `prefix`, and
    // `prefixIndex` exactly as they were before the call.
    struct PartialFailureTokenizer: ParserTokenizer {
      struct Failure: Error {}
      func convertTokenToId(_: String) -> Int? {
        nil
      }

      func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
        []
      }

      func decode(tokenIds: [Int], skipSpecialTokens _: Bool) throws -> String {
        switch tokenIds {
          case [1]: return "Hello"
          case [1, 2]: return "Hello world"
          case [2]: throw Failure()
          default: return ""
        }
      }
    }

    let stream = PartialFailureTokenizer().streamingDetokenizer()
    let first = try stream.consume(1)
    #expect(first == "Hello")

    let beforeMirror = Mirror(reflecting: stream)
    let beforeIds = beforeMirror.children.first { $0.label == "ids" }?.value as? [Int]
    let beforePrefix = beforeMirror.children.first { $0.label == "prefix" }?.value as? String
    let beforeIndex = beforeMirror.children.first { $0.label == "prefixIndex" }?.value as? Int
    #expect(beforeIds == [1])
    #expect(beforePrefix == "Hello")
    #expect(beforeIndex == 1)

    do {
      _ = try stream.consume(2)
      Issue.record("Expected decode failure")
    } catch is PartialFailureTokenizer.Failure {
      // Expected.
    }

    let afterMirror = Mirror(reflecting: stream)
    let afterIds = afterMirror.children.first { $0.label == "ids" }?.value as? [Int]
    let afterPrefix = afterMirror.children.first { $0.label == "prefix" }?.value as? String
    let afterIndex = afterMirror.children.first { $0.label == "prefixIndex" }?.value as? Int
    #expect(beforeIds == afterIds)
    #expect(beforePrefix == afterPrefix)
    #expect(beforeIndex == afterIndex)
  }

  @Test
  func `consume returns nil when token adds no bytes`() throws {
    // A "special" token whose decoded form is dropped by
    // `skipSpecialTokens: true` adds no bytes to the cumulative
    // decode. `consume` must withhold (return nil) rather than
    // throwing — the cached prefix is still a valid prefix of the
    // unchanged decode.
    struct SkipSpecial: ParserTokenizer {
      static let regularBytes: [Int: [UInt8]] = [
        0: Array("a".utf8),
        1: Array("b".utf8),
      ]
      func convertTokenToId(_: String) -> Int? {
        nil
      }

      func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
        []
      }

      func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
        var bytes: [UInt8] = []
        for id in tokenIds {
          if id == 99 {
            if skipSpecialTokens { continue }
            bytes.append(contentsOf: Array("<eos>".utf8))
            continue
          }
          if let chunk = Self.regularBytes[id] {
            bytes.append(contentsOf: chunk)
          }
        }
        return String(decoding: bytes, as: UTF8.self)
      }
    }

    let stream = SkipSpecial().streamingDetokenizer(skipSpecialTokens: true)
    #expect(try stream.consume(0) == "a")
    #expect(try stream.consume(99) == nil)
    #expect(try stream.consume(1) == "b")
  }

  @Test
  func `invalid streaming prefix throws`() throws {
    // Construct a tokenizer whose decode result is non-monotonic: the
    // single-token decode produces text that is *not* a prefix of the
    // two-token decode, breaking the invariant.
    struct NonMonotonic: ParserTokenizer {
      func encode(text _: String, addSpecialTokens _: Bool) throws -> [Int] {
        []
      }

      func decode(tokenIds: [Int], skipSpecialTokens _: Bool) throws -> String {
        if tokenIds == [1] { return "abc" }
        if tokenIds == [1, 2] { return "xyzlonger" }
        return ""
      }

      func convertTokenToId(_: String) -> Int? {
        nil
      }
    }

    let stream = NonMonotonic().streamingDetokenizer()
    _ = try stream.consume(1)
    do {
      _ = try stream.consume(2)
      Issue.record("Expected invalidStreamingPrefix throw")
    } catch let error as StreamingDetokenizerError {
      guard case let .invalidStreamingPrefix(tokenId, expectedPrefix, actualString) = error
      else {
        Issue.record("Expected .invalidStreamingPrefix, got \(error)")
        return
      }
      #expect(tokenId == 2)
      #expect(expectedPrefix == "abc")
      #expect(actualString == "xyzlonger")
    }
  }
}
