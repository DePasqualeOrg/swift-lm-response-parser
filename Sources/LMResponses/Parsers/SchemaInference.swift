// Copyright © Anthony DePasquale

import Foundation

/// Infer the primary JSON-schema type for a parameter from a (possibly
/// complex) schema definition, so callers can decide whether to coerce a
/// raw string-typed value into a number, boolean, object, array, or leave
/// it as a string. Mirrors sglang's `infer_type_from_json_schema`.
///
/// Resolution order:
///
/// 1. `type` field – string or non-empty array (the first non-null entry).
/// 2. `anyOf` / `oneOf` – recurse on each branch; if all branches agree on
///    the type, return it; if any branch is `string`, prefer that; else
///    return the first inferred type.
/// 3. `enum` – infer from the runtime types of the enum values; if all
///    values share a type, return it; otherwise default to `string`.
/// 4. `allOf` – recurse and return the first non-`string` type.
/// 5. `properties` present → `object`.
/// 6. `items` present → `array`.
///
/// Returns nil for schemas that match none of these rules. Callers can
/// then pick a sensible default (typically `string`).
func inferTypeFromJsonSchema(_ schema: Any) -> String? {
  guard let dict = schema as? [String: Any] else { return nil }

  if let typeValue = dict["type"] {
    if let s = typeValue as? String {
      return s.lowercased()
    }
    if let arr = typeValue as? [String], !arr.isEmpty {
      let nonNull = arr.first { $0.lowercased() != "null" }
      return (nonNull ?? "string").lowercased()
    }
  }

  if let variants = (dict["anyOf"] ?? dict["oneOf"]) as? [Any] {
    var types: [String] = []
    for variant in variants {
      if let inferred = inferTypeFromJsonSchema(variant) {
        types.append(inferred)
      }
    }
    if !types.isEmpty {
      let unique = Set(types)
      if unique.count == 1 { return types[0] }
      if unique.contains("string") { return "string" }
      return types[0]
    }
  }

  if let enumValues = dict["enum"] as? [Any] {
    if enumValues.isEmpty { return "string" }
    var enumTypes: Set<String> = []
    for value in enumValues {
      if value is NSNull { enumTypes.insert("null") }
      else if let n = value as? NSNumber, n.isBool { enumTypes.insert("boolean") }
      else if value is Int { enumTypes.insert("integer") }
      else if value is Double || value is Float { enumTypes.insert("number") }
      else if value is String { enumTypes.insert("string") }
      else if value is [Any] { enumTypes.insert("array") }
      else if value is [String: Any] { enumTypes.insert("object") }
    }
    if enumTypes.count == 1, let t = enumTypes.first { return t }
    return "string"
  }

  if let all = dict["allOf"] as? [Any] {
    for sub in all {
      if let inferred = inferTypeFromJsonSchema(sub), inferred != "string" {
        return inferred
      }
    }
    return "string"
  }

  if dict["properties"] != nil { return "object" }
  if dict["items"] != nil { return "array" }
  return nil
}

/// Extract the full set of candidate types from a JSON schema, including
/// every branch of `anyOf` / `oneOf` / `allOf` and every entry in a `type`
/// array. Mirrors sglang's `_extract_types_from_schema` and vLLM's helper
/// of the same name in `minimax_m2_tool_parser.py`.
///
/// Where ``inferTypeFromJsonSchema`` collapses to a single string by
/// preferring `string` (or the first inferred type) when branches disagree,
/// this returns the union so callers can try each candidate in priority
/// order. Use this when the consumer needs sglang's `integer > number >
/// boolean > object > array > string` precedence – currently MiniMax M2,
/// which receives string-typed parameter values and must decide whether
/// `"5"` should serialize as `5` or `"5"` based on schema candidates.
///
/// Returns `["string"]` for nil/non-dict schemas or when no other type
/// can be inferred, matching sglang's default.
func extractTypesFromJsonSchema(_ schema: Any) -> [String] {
  var types: Set<String> = []
  walkSchemaTypes(schema, into: &types)
  if types.isEmpty { return ["string"] }
  return Array(types)
}

private func walkSchemaTypes(_ schema: Any, into types: inout Set<String>) {
  guard let dict = schema as? [String: Any] else { return }

  if let typeValue = dict["type"] {
    if let s = typeValue as? String {
      types.insert(s.lowercased())
    } else if let arr = typeValue as? [String] {
      for t in arr {
        types.insert(t.lowercased())
      }
    }
  }

  if let enumValues = dict["enum"] as? [Any], !enumValues.isEmpty {
    for value in enumValues {
      if value is NSNull { types.insert("null") }
      else if let n = value as? NSNumber, n.isBool { types.insert("boolean") }
      else if value is Int { types.insert("integer") }
      else if value is Double || value is Float { types.insert("number") }
      else if value is String { types.insert("string") }
      else if value is [Any] { types.insert("array") }
      else if value is [String: Any] { types.insert("object") }
    }
  }

  for choiceField in ["anyOf", "oneOf", "allOf"] {
    if let choices = dict[choiceField] as? [Any] {
      for choice in choices {
        walkSchemaTypes(choice, into: &types)
      }
    }
  }
}

private extension NSNumber {
  /// True when the number was constructed from a Swift `Bool`. Used to
  /// disambiguate `Bool` from `Int`, since `NSNumber` boxes both as
  /// numeric values and Swift's runtime check against `Bool.self` after
  /// the cast through `Any` is unreliable.
  var isBool: Bool {
    #if canImport(Darwin)
    return CFGetTypeID(self) == CFBooleanGetTypeID()
    #else
    return String(cString: objCType) == "c"
    #endif
  }
}
