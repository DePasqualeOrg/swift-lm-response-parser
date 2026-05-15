// swift-tools-version: 6.1

import PackageDescription

// Integration tests live in their own test target so they can be gated
// behind the `LMRESPONSE_PARSER_INTEGRATION_TESTS=1` environment
// variable at manifest-evaluation time. The integration tests download
// MLX model snapshots and need MLX's Metal shader library, which is only
// produced by Xcode's build phases — not by plain `swift test` — so we
// avoid pulling the extra deps (HFAPI, swift-tokenizers, MLXLLM) into
// the dependency graph for ordinary consumers and contributors who only
// need to run unit tests.
//
// Mirrors swift-tokenizers' Benchmarks-target pattern.
let integrationTestsEnabled =
  Context.environment["LMRESPONSE_PARSER_INTEGRATION_TESTS"] == "1"

// MLX is Apple-only (Metal/Accelerate). The MLX-backed library, its
// tests, and the integration tests are excluded on non-Apple platforms
// so Linux CI can build and test the pure-Swift `LMResponseParser`
// target without dragging MLX into the build graph.
#if canImport(Darwin)
let isApplePlatform = true
#else
let isApplePlatform = false
#endif

var packageDependencies: [Package.Dependency] = [
  // Pulled from main to pick up two unreleased Swift-DocC fixes:
  // PR #1327 (combined-documentation cross-target symbol-link resolution)
  // and PR #1417 (live-reload preview). Drop these once both ship in a
  // released Xcode toolchain — they are only used by scripts/verify-docs.sh
  // and scripts/preview-docs.sh, not by the library targets.
  .package(url: "https://github.com/swiftlang/swift-docc", branch: "main"),
  .package(url: "https://github.com/swiftlang/swift-docc-plugin", branch: "main"),
]

if isApplePlatform {
  // Temporarily pointed at the DePasqualeOrg fork's `main` branch to pick
  // up the throwing `MLXLMCommon.Tokenizer` API. Revert to the upstream
  // `ml-explore/mlx-swift-lm` package (and bump the `from:` floor) once
  // the throwing API ships in a tagged release.
  packageDependencies.append(
    .package(
      url: "https://github.com/DePasqualeOrg/swift-lm.git",
      branch: "main",
    ),
  )
}

if isApplePlatform, integrationTestsEnabled {
  packageDependencies.append(contentsOf: [
    .package(url: "https://github.com/DePasqualeOrg/swift-hf-api.git", from: "0.2.2"),
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "0.4.2"),
  ])
}

var packageProducts: [Product] = [
  .library(
    name: "LMResponseParser",
    targets: ["LMResponseParser"],
  ),
]

if isApplePlatform {
  packageProducts.append(
    .library(
      name: "LMResponseParserMLX",
      targets: ["LMResponseParserMLX"],
    ),
  )
}

var packageTargets: [Target] = [
  .target(
    name: "LMResponseParser",
    path: "Sources/LMResponseParser",
  ),
  .testTarget(
    name: "LMResponseParserTests",
    dependencies: ["LMResponseParser"],
    path: "Tests/LMResponseParserTests",
  ),
]

if isApplePlatform {
  packageTargets.append(contentsOf: [
    .target(
      name: "LMResponseParserMLX",
      dependencies: [
        "LMResponseParser",
        .product(name: "MLXLMCommon", package: "swift-lm"),
      ],
      path: "Sources/LMResponseParserMLX",
    ),
    .testTarget(
      name: "LMResponseParserMLXTests",
      dependencies: [
        "LMResponseParserMLX",
        .product(name: "MLXLMCommon", package: "swift-lm"),
      ],
      path: "Tests/LMResponseParserMLXTests",
    ),
  ])
}

if isApplePlatform, integrationTestsEnabled {
  packageTargets.append(
    .testTarget(
      name: "LMResponseParserMLXIntegrationTests",
      dependencies: [
        "LMResponseParserMLX",
        .product(name: "MLXLMCommon", package: "swift-lm"),
        .product(name: "MLXLLM", package: "swift-lm"),
        .product(name: "HFAPI", package: "swift-hf-api"),
        .product(name: "Tokenizers", package: "swift-tokenizers"),
      ],
      path: "Tests/LMResponseParserMLXIntegrationTests",
    ),
  )
}

let package = Package(
  name: "swift-lm-response-parser",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
    .tvOS(.v17),
    .visionOS(.v1),
  ],
  products: packageProducts,
  dependencies: packageDependencies,
  targets: packageTargets,
)
