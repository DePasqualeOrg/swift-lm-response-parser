# Tests

This package has three test targets:

- **`LMResponseParserTests`**: unit tests for the parser core (per-format parsers, streaming envelope, items accumulator).
- **`LMResponseParserMLXTests`**: unit tests for the MLX bridge that don't require a downloaded model.
- **`LMResponseParserMLXIntegrationTests`**: end-to-end tests that download a small MLX model and drive it through `ResponseChatSession`.

`LMResponseParserTests` and `LMResponseParserMLXTests` run with plain `swift test`. The integration target is gated and described below.

## Integration tests

`LMResponseParserMLXIntegrationTests` downloads a small MLX model and drives it through `ResponseChatSession`. It's gated behind `LMRESPONSE_PARSER_INTEGRATION_TESTS=1` so ordinary consumers don't pull `swift-hf-api` and `swift-tokenizers` into their dependency graph. Set the env var before evaluating the package to include the target.

**In Xcode**: the env var must be present when Xcode resolves the package, which happens on launch. The easiest persistent option is `launchctl setenv LMRESPONSE_PARSER_INTEGRATION_TESTS 1` (run once, then reopen Xcode). The integration suite then appears in the test navigator alongside the unit tests.

**From the command line**: MLX's `default.metallib` is built by Xcode-specific build phases, not by plain `swift test`, so use `xcodebuild test` to drive the integration suite. SwiftPM packages auto-generate library schemes (which aren't test-configured) plus a `…-Package` scheme that runs every test target – the integration tests go through that one with `-only-testing:` scoped to the integration target. The model snapshot is cached to `~/.cache/huggingface/hub` (the standard Hugging Face location) and reused by the `hf` CLI and other tools across runs.

```bash
# Run all integration tests
LMRESPONSE_PARSER_INTEGRATION_TESTS=1 xcodebuild test \
  -scheme swift-lm-response-parser-Package \
  -destination 'platform=macOS,arch=arm64' \
  -only-testing:LMResponseParserMLXIntegrationTests

# Run a single integration test
LMRESPONSE_PARSER_INTEGRATION_TESTS=1 xcodebuild test \
  -scheme swift-lm-response-parser-Package \
  -destination 'platform=macOS,arch=arm64' \
  -only-testing:LMResponseParserMLXIntegrationTests/Qwen3IntegrationTests/envelopeShape

# Run against an already-cached snapshot without checking the hub
HF_OFFLINE=1 LMRESPONSE_PARSER_INTEGRATION_TESTS=1 xcodebuild test \
  -scheme swift-lm-response-parser-Package \
  -destination 'platform=macOS,arch=arm64' \
  -only-testing:LMResponseParserMLXIntegrationTests
```

First-run cost is ~400 MB for `mlx-community/Qwen3-0.6B-4bit`.
