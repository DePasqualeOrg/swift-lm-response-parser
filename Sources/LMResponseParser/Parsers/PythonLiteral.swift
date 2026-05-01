// Copyright © Anthony DePasquale

import Foundation

/// Shared parser for Python literal expressions emitted by language models.
/// Converts Python source for strings, numbers, booleans, `None`, lists,
/// tuples, parenthesized expressions, and dicts into the JSON equivalent. Used by ``PythonicParser`` for
/// argument values inside `[fn(arg=value)]` syntax and by ``Llama3Parser``
/// to recover from Python-dict output where strict JSON was expected.
///
/// The parser mirrors Python's `ast.literal_eval` for string escape
/// sequences (`\n`, `\t`, `\xHH`, `\uHHHH`, `\UHHHHHHHH`, octal) and
/// accepts JSON-style `null`/`true`/`false` alongside `None`/`True`/
/// `False` because some models mix the two notations.
///
/// **Diverges from `ast.literal_eval` for unknown bare tokens.** The
/// Python reference raises `ValueError` on identifiers like `undefined`
/// or `NaN`; vLLM's pythonic parser propagates that to drop the entire
/// tool call. This parser instead JSON-encodes the unknown token as a
/// string, on the principle that recovering a tool call with one weird
/// argument is more useful than discarding the whole call. The output is
/// always valid JSON either way.
enum PythonLiteral {
  /// Parse a Python literal expression starting at `start` in `text`. On
  /// success, returns the JSON-encoded value and the index of the first
  /// character past the literal.
  static func parseValue(
    in text: String,
    from start: String.Index,
  ) -> (json: String, end: String.Index)? {
    guard start < text.endIndex else { return nil }
    let ch = text[start]
    if ch == "\"" || ch == "'" {
      return parseString(in: text, from: start)
    } else if ch == "[" {
      return parseContainer(in: text, from: start, close: "]", isList: true)
    } else if ch == "(" {
      return parseTupleOrParenthesized(in: text, from: start)
    } else if ch == "{" {
      return parseContainer(in: text, from: start, close: "}", isList: false)
    } else {
      return parseBareToken(in: text, from: start)
    }
  }

  /// Parse a complete top-level Python dict literal and return its JSON
  /// encoding. Returns nil if `text` does not parse as a single dict.
  static func parseTopLevelDict(_ text: String) -> String? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}") else { return nil }
    guard let (json, end) = parseValue(in: trimmed, from: trimmed.startIndex) else {
      return nil
    }
    // Reject trailing non-whitespace content after the dict.
    let tail = trimmed[end...].trimmingCharacters(in: .whitespacesAndNewlines)
    guard tail.isEmpty else { return nil }
    return json
  }

  private static func parseString(
    in text: String,
    from start: String.Index,
  ) -> (json: String, end: String.Index)? {
    let quote = text[start]
    var i = text.index(after: start)
    var content = ""
    while i < text.endIndex {
      let ch = text[i]
      if ch == "\\" {
        let escIdx = text.index(after: i)
        guard escIdx < text.endIndex else { return nil }
        guard let next = applyEscape(in: text, at: escIdx, into: &content) else {
          return nil
        }
        i = next
        continue
      }
      if ch == quote {
        return (jsonEncodeString(content), text.index(after: i))
      }
      content.append(ch)
      i = text.index(after: i)
    }
    return nil
  }

  private static func parseBareToken(
    in text: String,
    from start: String.Index,
  ) -> (json: String, end: String.Index)? {
    var i = start
    while i < text.endIndex {
      let ch = text[i]
      if ch == "," || ch == ")" || ch == "]" || ch == "}" || ch == ":" || ch.isWhitespace {
        break
      }
      i = text.index(after: i)
    }
    let token = String(text[start ..< i])
    if token.isEmpty { return nil }
    let json: String = switch token {
      case "True", "true": "true"
      case "False", "false": "false"
      case "None", "null": "null"
      default:
        if let intJSON = parseIntegerLiteral(token) {
          intJSON
        } else if let d = Double(stripUnderscores(token)) {
          // Re-serialize so that Python forms like `.5` become valid
          // JSON (`0.5`); also normalizes scientific notation.
          if let data = try? JSONSerialization.data(
            withJSONObject: [d],
            options: [.fragmentsAllowed],
          ),
            let encoded = String(data: data, encoding: .utf8),
            encoded.count >= 2
          {
            String(encoded.dropFirst().dropLast())
          } else {
            token
          }
        } else {
          jsonEncodeString(token)
        }
    }
    return (json, i)
  }

  /// Parse the numeric forms ``ast.literal_eval`` accepts for Python
  /// integer literals: plain decimal, hex (`0x`), octal (`0o`), binary
  /// (`0b`), each with optional sign and PEP 515 digit separators
  /// (e.g. `1_000`). Returns the JSON-encoded integer, or nil if the
  /// token does not parse as one of these forms.
  private static func parseIntegerLiteral(_ token: String) -> String? {
    let stripped = stripUnderscores(token)
    guard !stripped.isEmpty else { return nil }
    var sign = ""
    var body = Substring(stripped)
    if body.first == "+" {
      body = body.dropFirst()
    } else if body.first == "-" {
      sign = "-"
      body = body.dropFirst()
    }
    guard !body.isEmpty else { return nil }
    if body.count >= 3, body.first == "0" {
      let prefix = body[body.index(after: body.startIndex)]
      let digits = body.dropFirst(2)
      switch prefix {
        case "x", "X":
          if let i = Int(digits, radix: 16) { return "\(sign)\(i)" }
        case "o", "O":
          if let i = Int(digits, radix: 8) { return "\(sign)\(i)" }
        case "b", "B":
          if let i = Int(digits, radix: 2) { return "\(sign)\(i)" }
        default:
          break
      }
    }
    if body.allSatisfy(isASCIIDigit) {
      if let i = Int(stripped) { return String(i) }
      let digits = body.drop { $0 == "0" }
      guard !digits.isEmpty else { return "0" }
      return "\(sign)\(digits)"
    }
    if let i = Int(stripped) { return String(i) }
    return nil
  }

  private static func stripUnderscores(_ s: String) -> String {
    guard s.contains("_") else { return s }
    return s.replacingOccurrences(of: "_", with: "")
  }

  private static func isASCIIDigit(_ ch: Character) -> Bool {
    guard ch.unicodeScalars.count == 1, let scalar = ch.unicodeScalars.first else {
      return false
    }
    return scalar.value >= 48 && scalar.value <= 57
  }

  private static func parseContainer(
    in text: String,
    from start: String.Index,
    close: Character,
    isList: Bool,
  ) -> (json: String, end: String.Index)? {
    var i = text.index(after: start)
    var items: [String] = []
    while i < text.endIndex {
      while i < text.endIndex, text[i].isWhitespace || text[i] == "," {
        i = text.index(after: i)
      }
      if i >= text.endIndex { return nil }
      if text[i] == close {
        let body = items.joined(separator: ", ")
        let json = isList ? "[\(body)]" : "{\(body)}"
        return (json, text.index(after: i))
      }
      if isList {
        guard let (val, next) = parseValue(in: text, from: i) else { return nil }
        items.append(val)
        i = next
      } else {
        guard let (key, afterKey) = parseValue(in: text, from: i) else { return nil }
        var j = afterKey
        while j < text.endIndex, text[j].isWhitespace {
          j = text.index(after: j)
        }
        guard j < text.endIndex, text[j] == ":" else { return nil }
        j = text.index(after: j)
        while j < text.endIndex, text[j].isWhitespace {
          j = text.index(after: j)
        }
        guard let (value, afterValue) = parseValue(in: text, from: j) else { return nil }
        items.append("\(key): \(value)")
        i = afterValue
      }
    }
    return nil
  }

  private static func parseTupleOrParenthesized(
    in text: String,
    from start: String.Index,
  ) -> (json: String, end: String.Index)? {
    var i = text.index(after: start)
    var items: [String] = []
    var sawComma = false
    while i < text.endIndex {
      while i < text.endIndex, text[i].isWhitespace {
        i = text.index(after: i)
      }
      if i >= text.endIndex { return nil }
      if text[i] == ")" {
        let body = items.joined(separator: ", ")
        return ("[\(body)]", text.index(after: i))
      }
      guard let (val, next) = parseValue(in: text, from: i) else { return nil }
      items.append(val)
      i = next
      while i < text.endIndex, text[i].isWhitespace {
        i = text.index(after: i)
      }
      if i < text.endIndex, text[i] == "," {
        sawComma = true
        i = text.index(after: i)
        continue
      }
      if i < text.endIndex, text[i] == ")" {
        if !sawComma, items.count == 1 {
          return (items[0], text.index(after: i))
        }
        let body = items.joined(separator: ", ")
        return ("[\(body)]", text.index(after: i))
      }
      return nil
    }
    return nil
  }

  private static func applyEscape(
    in text: String,
    at idx: String.Index,
    into content: inout String,
  ) -> String.Index? {
    let ch = text[idx]
    switch ch {
      case "\\": content.append("\\"); return text.index(after: idx)
      case "'": content.append("'"); return text.index(after: idx)
      case "\"": content.append("\""); return text.index(after: idx)
      case "a": content.append("\u{07}"); return text.index(after: idx)
      case "b": content.append("\u{08}"); return text.index(after: idx)
      case "f": content.append("\u{0C}"); return text.index(after: idx)
      case "n": content.append("\n"); return text.index(after: idx)
      case "r": content.append("\r"); return text.index(after: idx)
      case "t": content.append("\t"); return text.index(after: idx)
      case "v": content.append("\u{0B}"); return text.index(after: idx)
      case "\n":
        return text.index(after: idx)
      case "\r":
        let after = text.index(after: idx)
        if after < text.endIndex, text[after] == "\n" {
          return text.index(after: after)
        }
        return after
      case "x":
        return readHex(in: text, after: idx, count: 2, into: &content)
      case "u":
        return readHex(in: text, after: idx, count: 4, into: &content)
      case "U":
        return readHex(in: text, after: idx, count: 8, into: &content)
      default:
        if let asc = ch.asciiValue, asc >= 0x30, asc <= 0x37 {
          return readOctal(in: text, from: idx, into: &content)
        }
        content.append("\\")
        content.append(ch)
        return text.index(after: idx)
    }
  }

  private static func readHex(
    in text: String,
    after specifier: String.Index,
    count: Int,
    into content: inout String,
  ) -> String.Index? {
    var j = text.index(after: specifier)
    var hex = ""
    while hex.count < count, j < text.endIndex, text[j].isHexDigit {
      hex.append(text[j])
      j = text.index(after: j)
    }
    guard hex.count == count,
          let value = UInt32(hex, radix: 16),
          let scalar = Unicode.Scalar(value) else { return nil }
    content.unicodeScalars.append(scalar)
    return j
  }

  private static func readOctal(
    in text: String,
    from start: String.Index,
    into content: inout String,
  ) -> String.Index? {
    var oct = ""
    var j = start
    while oct.count < 3, j < text.endIndex,
          let asc = text[j].asciiValue, asc >= 0x30, asc <= 0x37
    {
      oct.append(text[j])
      j = text.index(after: j)
    }
    guard !oct.isEmpty,
          let value = UInt32(oct, radix: 8),
          let scalar = Unicode.Scalar(value) else { return nil }
    content.unicodeScalars.append(scalar)
    return j
  }

  private static func jsonEncodeString(_ s: String) -> String {
    if let data = try? JSONSerialization.data(withJSONObject: [s], options: []),
       let encoded = String(data: data, encoding: .utf8),
       encoded.count >= 2
    {
      return String(encoded.dropFirst().dropLast())
    }
    var out = "\""
    for ch in s {
      switch ch {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out.append(ch)
      }
    }
    out += "\""
    return out
  }
}
