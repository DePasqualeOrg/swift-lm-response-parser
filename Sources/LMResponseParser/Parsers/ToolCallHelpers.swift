// Copyright © Anthony DePasquale

import Foundation

/// Best-effort pull of `"name": "..."` from a tool-call JSON snippet.
/// Returns nil when the name field hasn't been written yet (still streaming).
/// Walks JSON structure manually so a `"name"` inside a string value or a
/// nested object is not mistaken for the top-level key.
///
/// Assumes a `{...}` outer shape (the Hermes spec). For the rare
/// `[{...}]` array shape, depth-1 sits inside the array (not the object),
/// so no `"name"` key is found at the searched depth and the call drops.
/// sglang's one-shot `detect_and_parse` accepts arrays via list/dict
/// polymorphism, but its streaming path silently skips them too – and no
/// production Hermes/Qwen template emits the array form.
func extractToolName(from jsonText: String) -> String? {
  guard let valueStart = findTopLevelKey("name", in: jsonText) else { return nil }
  let chars = Array(jsonText)
  var i = valueStart
  while i < chars.count, chars[i].isWhitespace {
    i += 1
  }
  guard i < chars.count, chars[i] == "\"" else { return nil }
  i += 1
  var name = ""
  var escape = false
  while i < chars.count {
    let c = chars[i]
    if escape { name.append(c); escape = false; i += 1; continue }
    if c == "\\" { escape = true; i += 1; continue }
    if c == "\"" { return name }
    name.append(c)
    i += 1
  }
  return nil
}

/// Extract the JSON-encoded `arguments` value text from a tool-call snippet.
/// When `isComplete`, walks the JSON structure to find the exact end of the
/// arguments value (object/array/string/primitive), so a wire form that
/// places `arguments` before `name` (e.g.
/// `{"arguments":{"x":1},"name":"f"}`) yields `{"x":1}`, not the trailing
/// remainder. When the region is still being streamed, returns whatever
/// has been written so far. Locates the key with depth-and-string-aware
/// scanning so an `"arguments":"…"` token nested inside a string value or
/// inside a sub-object isn't mistaken for the top-level key.
///
/// Falls back to the `parameters` alias when `arguments` is missing.
/// sglang's `parse_base_json` accepts either key, and several models –
/// notably the granite/llama variants that reuse the Hermes envelope –
/// emit `parameters` instead. Mirrors the dual-key handling already in
/// `Llama3Parser`, `JSONFallbackParser`, `MistralParser`, and `Phi4MiniParser`.
func extractArgumentsText(from jsonText: String, isComplete: Bool) -> String? {
  let valueStart: Int
  if let s = findTopLevelKey("arguments", in: jsonText) {
    valueStart = s
  } else if let s = findTopLevelKey("parameters", in: jsonText) {
    valueStart = s
  } else {
    return nil
  }
  let chars = Array(jsonText)
  if isComplete {
    let end = endOfJSONValue(in: chars, from: valueStart)
    let raw = String(chars[valueStart ..< end])
    return raw.trimmingCharacters(in: .whitespacesAndNewlines)
  }
  return String(chars[valueStart ..< chars.count])
}

/// Walk forward from `valueStart` in a JSON character array, returning the
/// end-exclusive index of the value at that position. Balances `{}` /
/// `[]`, walks through string literals respecting backslash escapes, and
/// stops primitive values (numbers, booleans, `null`) at the first comma
/// or closing bracket of the parent.
func endOfJSONValue(in chars: [Character], from valueStart: Int) -> Int {
  guard valueStart < chars.count else { return valueStart }
  let first = chars[valueStart]
  if first == "{" || first == "[" {
    let opening = first
    let closing: Character = opening == "{" ? "}" : "]"
    var depth = 0
    var inString = false
    var escape = false
    var i = valueStart
    while i < chars.count {
      let c = chars[i]
      if inString {
        if escape { escape = false; i += 1; continue }
        if c == "\\" { escape = true; i += 1; continue }
        if c == "\"" { inString = false }
        i += 1
        continue
      }
      if c == "\"" { inString = true; i += 1; continue }
      if c == opening { depth += 1 }
      else if c == closing {
        depth -= 1
        if depth == 0 { return i + 1 }
      }
      i += 1
    }
    return chars.count
  }
  if first == "\"" {
    var i = valueStart + 1
    var escape = false
    while i < chars.count {
      let c = chars[i]
      if escape { escape = false; i += 1; continue }
      if c == "\\" { escape = true; i += 1; continue }
      if c == "\"" { return i + 1 }
      i += 1
    }
    return chars.count
  }
  var i = valueStart
  while i < chars.count {
    let c = chars[i]
    if c == "," || c == "}" || c == "]" { return i }
    i += 1
  }
  return chars.count
}

/// Scan `jsonText` for a top-level (depth-1) key matching `key` and return
/// the index of the first character of its value (i.e., the position after
/// the `:` and any following whitespace). Returns nil when the key isn't
/// present or hasn't yet been written.
///
/// "Top-level" means the key sits immediately inside the outer `{...}`.
/// Keys inside nested objects, arrays, or string literals are skipped so
/// e.g. `{"name":"echo","arguments":{"text":"\"arguments\":42"}}` resolves
/// `arguments` to the actual value, not the inner string-literal occurrence.
func findTopLevelKey(_ key: String, in jsonText: String) -> Int? {
  let chars = Array(jsonText)
  let target = Array(key)
  var depth = 0
  var inString = false
  var escape = false
  var i = 0
  while i < chars.count {
    let c = chars[i]
    if escape { escape = false; i += 1; continue }
    if c == "\\" { escape = true; i += 1; continue }
    if c == "\"" {
      if !inString {
        // Possible key opener at depth 1. Try to match the literal.
        if depth == 1, i + 1 + target.count < chars.count {
          let endQuoteIdx = i + 1 + target.count
          if chars[(i + 1) ..< endQuoteIdx].elementsEqual(target), chars[endQuoteIdx] == "\"" {
            // Walk past optional whitespace and `:` to land at value.
            var j = endQuoteIdx + 1
            while j < chars.count, chars[j].isWhitespace {
              j += 1
            }
            guard j < chars.count, chars[j] == ":" else {
              // Not a key – slide past the literal as a string.
              i = endQuoteIdx + 1
              continue
            }
            j += 1
            while j < chars.count, chars[j].isWhitespace {
              j += 1
            }
            return j
          }
        }
        inString = true
        i += 1
        continue
      } else {
        inString = false
        i += 1
        continue
      }
    }
    if inString { i += 1; continue }
    if c == "{" || c == "[" { depth += 1 }
    else if c == "}" || c == "]" { depth -= 1 }
    i += 1
  }
  return nil
}

func isValidJSON(_ text: String) -> Bool {
  guard let data = text.data(using: .utf8) else { return false }
  return (try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])) != nil
}

extension [Character] {
  /// Find the start index of `substring` in this character array, scanning
  /// from `after` (inclusive). Returns nil when not found.
  func firstIndexOf(substring: String, after: Int) -> Int? {
    let needle = Array(substring)
    if needle.isEmpty || count - after < needle.count { return nil }
    var i = after
    let last = count - needle.count
    while i <= last {
      if self[i ..< (i + needle.count)].elementsEqual(needle) {
        return i
      }
      i += 1
    }
    return nil
  }
}

/// Length of the longest prefix of `tag` that matches a suffix of `chars`.
/// Returns 0 when there's no overlap. Operates on character arrays so
/// non-ASCII markers work correctly.
func partialOverlap(suffixOf chars: [Character], with tag: [Character]) -> Int {
  if chars.isEmpty || tag.count <= 1 { return 0 }
  let maxCheck = Swift.min(tag.count - 1, chars.count)
  var k = maxCheck
  while k > 0 {
    if chars[(chars.count - k)...].elementsEqual(tag[..<k]) {
      return k
    }
    k -= 1
  }
  return 0
}
