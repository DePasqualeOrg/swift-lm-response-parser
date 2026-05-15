// Copyright © Anthony DePasquale

import Foundation

/// Mints opaque per-item and per-response IDs that follow OpenAI's prefix
/// conventions: a fixed prefix joined with a 24-character base32 suffix
/// (120 bits – enough entropy for response-scoped uniqueness without any
/// coordination between the parser and the driver).
package enum IDFactory {
  package enum Prefix: String, CaseIterable {
    /// Assistant message item.
    case message = "msg"

    /// Function-call item.
    case functionCall = "fc"

    /// `call_id` field on a function-call item, distinct from the
    /// item's `fc_…` ID. Downstream `function_call_output` items
    /// reference the call by `call_id`, so it has to outlive any
    /// rename of the item ID.
    case callId = "call"

    /// Reasoning item.
    case reasoning = "rs"

    /// Function-call-output item (tool result). The spec's natural
    /// prefix is `fco_…`; vLLM uses `mcpo_…` for an MCP-specific
    /// variant we do not produce.
    case functionCallOutput = "fco"

    /// Response-scoped envelope ID.
    case response = "resp"
  }

  /// Mint a fresh ID with the given prefix. Each call produces a new
  /// random suffix.
  package static func make(_ prefix: Prefix) -> String {
    "\(prefix.rawValue)_\(randomBase32Suffix(length: suffixLength))"
  }

  /// Mint an ID matching the shape Mistral chat templates expect:
  /// exactly 9 alphanumeric characters with no prefix. The upstream
  /// HF templates for `mistralai/Mistral-7B-Instruct-v0.3` and
  /// `mistralai/Ministral-8B-Instruct-2410` enforce
  /// `tool_call.id|length != 9` and insert the id verbatim, so the
  /// generic prefixed shape would raise a Jinja exception. Mirrors
  /// vLLM's `MistralToolCall.generate_random_id`.
  package static func makeMistralStrict() -> String {
    randomBase32Suffix(length: mistralStrictLength)
  }

  static let suffixLength = 24
  static let mistralStrictLength = 9

  // Crockford-style alphabet (no I/L/O/U) so suffixes don't get misread
  // in logs or CLI output.
  private static let alphabet: [Character] = Array("0123456789abcdefghjkmnpqrstvwxyz")

  private static func randomBase32Suffix(length: Int) -> String {
    var bytes = [UInt8](repeating: 0, count: length)
    for i in 0 ..< length {
      bytes[i] = UInt8.random(in: 0 ... 31)
    }
    return String(bytes.map { alphabet[Int($0)] })
  }
}
