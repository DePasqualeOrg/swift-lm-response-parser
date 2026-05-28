// Copyright © Anthony DePasquale

import Foundation
import HFAPI
import Llama

enum IntegrationTestError: LocalizedError {
  case invalidRepositoryID(String)
  case ggufNotFoundInSnapshot(repo: String, filename: String)

  var errorDescription: String? {
    switch self {
      case let .invalidRepositoryID(id):
        "Invalid Hugging Face repository ID '\(id)' (expected 'namespace/name')."
      case let .ggufNotFoundInSnapshot(repo, filename):
        "GGUF '\(filename)' not found in snapshot of '\(repo)'."
    }
  }
}

/// GGUF model fixture. Vocab + chat template + architecture all live in
/// the GGUF, so tests don't need a separate HF tokenizer-assets repo.
struct LlamaTestFixture {
  let ggufRepoID: String
  let ggufFilename: String
  let modelName: String

  /// Qwen3 0.6B Q4_K_M. ~400 MB GGUF.
  static let qwen3_0_6b = LlamaTestFixture(
    ggufRepoID: "unsloth/Qwen3-0.6B-GGUF",
    ggufFilename: "Qwen3-0.6B-Q4_K_M.gguf",
    modelName: "Qwen3-0.6B",
  )
}

/// VLM fixture: language GGUF + mmproj GGUF (usually same repo).
struct LlamaVLMTestFixture {
  let ggufRepoID: String
  let modelFilename: String
  let mmprojFilename: String
  let modelName: String

  /// Gemma 4 E2B Instruct, Q4_K_M + f16 mmproj.
  static let gemma4_e2b = LlamaVLMTestFixture(
    ggufRepoID: "unsloth/gemma-4-E2B-it-GGUF",
    modelFilename: "gemma-4-E2B-it-Q4_K_M.gguf",
    mmprojFilename: "mmproj-F16.gguf",
    modelName: "Gemma-4-E2B-it",
  )

  /// Voxtral-Mini-3B (Mistral's audio model). GGUF lives in ggml-org's repo.
  static let voxtral_mini_3b = LlamaVLMTestFixture(
    ggufRepoID: "ggml-org/Voxtral-Mini-3B-2507-GGUF",
    modelFilename: "Voxtral-Mini-3B-2507-Q4_K_M.gguf",
    mmprojFilename: "mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf",
    modelName: "Voxtral-Mini-3B-2507",
  )
}

actor IntegrationFixtures {
  private let client: HubClient
  private var ggufResolved: [String: Task<URL, Error>] = [:]
  private var vlmPairResolved: [String: Task<(model: URL, mmproj: URL), Error>] = [:]

  init(client: HubClient = HubClient()) {
    self.client = client
  }

  /// Downloads the GGUF and returns its local path.
  func ggufURL(for fixture: LlamaTestFixture) async throws -> URL {
    let key = "\(fixture.ggufRepoID)/\(fixture.ggufFilename)"
    if let task = ggufResolved[key] { return try await task.value }
    let client = client
    let task = Task {
      try await Self.downloadGGUF(
        repoID: fixture.ggufRepoID,
        filename: fixture.ggufFilename,
        client: client,
      )
    }
    ggufResolved[key] = task
    return try await task.value
  }

  func vlmGGUFURLs(for fixture: LlamaVLMTestFixture) async throws -> (model: URL, mmproj: URL) {
    let key = "\(fixture.ggufRepoID)/\(fixture.modelFilename)+\(fixture.mmprojFilename)"
    if let task = vlmPairResolved[key] { return try await task.value }
    let client = client
    let task = Task {
      try await Self.downloadVLMPair(
        repoID: fixture.ggufRepoID,
        modelFilename: fixture.modelFilename,
        mmprojFilename: fixture.mmprojFilename,
        client: client,
      )
    }
    vlmPairResolved[key] = task
    return try await task.value
  }

  private static func downloadVLMPair(
    repoID: String,
    modelFilename: String,
    mmprojFilename: String,
    client: HubClient,
  ) async throws -> (model: URL, mmproj: URL) {
    guard let repo = HFAPI.Repo.ID(rawValue: repoID) else {
      throw IntegrationTestError.invalidRepositoryID(repoID)
    }
    let label = "\(modelFilename) + \(mmprojFilename)"
    print("Downloading \(label) from \(repoID)…")
    let snapshot = try await client.downloadSnapshot(
      of: repo,
      revision: "main",
      matching: [modelFilename, mmprojFilename],
      progressHandler: logProgress(label),
    )
    let modelURL = snapshot.appending(path: modelFilename)
    let mmprojURL = snapshot.appending(path: mmprojFilename)
    for (filename, url) in [(modelFilename, modelURL), (mmprojFilename, mmprojURL)] {
      guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
        throw IntegrationTestError.ggufNotFoundInSnapshot(repo: repoID, filename: filename)
      }
    }
    return (modelURL, mmprojURL)
  }

  private static func downloadGGUF(
    repoID: String,
    filename: String,
    client: HubClient,
  ) async throws -> URL {
    guard let repo = HFAPI.Repo.ID(rawValue: repoID) else {
      throw IntegrationTestError.invalidRepositoryID(repoID)
    }
    print("Downloading \(filename) from \(repoID)…")
    let snapshot = try await client.downloadSnapshot(
      of: repo,
      revision: "main",
      matching: [filename],
      progressHandler: logProgress(filename),
    )
    let url = snapshot.appending(path: filename)
    guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
      throw IntegrationTestError.ggufNotFoundInSnapshot(repo: repoID, filename: filename)
    }
    return url
  }
}

func logProgress(_ label: String) -> @Sendable (Foundation.Progress) -> Void {
  let lock = NSLock()
  nonisolated(unsafe) var lastThreshold = -1
  return { progress in
    let pct = Int(progress.fractionCompleted * 100)
    let threshold = pct / 10
    lock.lock()
    let shouldPrint = threshold > lastThreshold
    if shouldPrint { lastThreshold = threshold }
    lock.unlock()
    if shouldPrint {
      print("  \(label): \(pct)%")
    }
  }
}

var integrationTestsEnabled: Bool {
  ProcessInfo.processInfo.environment["LMRESPONSES_INTEGRATION_TESTS"] == "1"
}
