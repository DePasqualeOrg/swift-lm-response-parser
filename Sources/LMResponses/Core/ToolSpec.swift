// Copyright © Anthony DePasquale

import Foundation

/// A tool's JSON-schema dictionary as parsed from a request.
///
/// The schema is a free-form `[String: any Sendable]` rather than a
/// strongly typed value because parsers only need the schema half
/// (parameter names and JSON-schema types) to do argument-type coercion.
/// A strongly typed `Tool` would also carry a generic handler closure,
/// which the parser layer has no business with.
/// Parser-side coercion exists only to normalize model wire formats that
/// emit untyped parameter text; it is not a promise that the arguments are
/// valid for execution.
///
/// **Name validation is the consumer's responsibility.** Parsers forward
/// every tool call the model emits, including hallucinated names not
/// present in the supplied tools. sglang's base detector filters unknown
/// names by default (gated by `SGLANG_FORWARD_UNKNOWN_TOOLS`); we
/// deliberately diverge because tool dispatch happens in the host
/// application, not in this library. Consumers that want sglang-style
/// filtering can drop unrecognized `function_call` items by name after
/// `accumulateItems`.
public typealias ToolSpec = [String: any Sendable]
