#!/bin/bash
# Preview documentation with live reload on http://localhost:8080/documentation/lmresponses.
#
# Unlike `swift package preview-documentation`, this script loads the symbol
# graphs for every library target so cross-target symbol links (e.g.
# ``/LMResponses/ResponseStream`` from inside an LMResponsesMLX
# article) resolve correctly. The preview plugin only supports a single
# --target at a time, so we invoke `docc preview` directly, passing the
# parent symbol-graph directory that contains every target's extracted graphs.
#
# Currently uses a docc binary built from swift-docc's main branch for
# live-reload support (swift-docc PR #1417, not yet in a released toolchain).
#
# TODO: once Apple ships a Swift-DocC release containing PR #1417, swap
# `$DOCC` for `xcrun docc` and drop the manual build step.
#
# `docc preview` watches the `.docc` catalog for changes to Markdown files,
# so editing articles reloads automatically. Source (Swift) changes do NOT
# trigger re-extraction of symbol graphs — rerun the script after editing
# doc comments in Swift source.

set -e

cd "$(dirname "$0")/.." || exit 1

CHILD_PID=""

kill_tree() {
    local pid=$1
    local children
    children=$(pgrep -P "$pid" 2>/dev/null) || true
    for child in $children; do
        kill_tree "$child"
    done
    kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
    trap - EXIT INT TERM HUP
    if [ -n "$CHILD_PID" ]; then
        kill_tree "$CHILD_PID"
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup EXIT INT TERM HUP

# Build docc from the swift-docc main-branch checkout pulled via Package.swift.
swift package resolve > /dev/null
SWIFT_DOCC_DIR=$(find .build/checkouts -maxdepth 1 -type d -name swift-docc -print -quit)
if [ -z "$SWIFT_DOCC_DIR" ]; then
    echo "Failed to locate swift-docc checkout under .build/checkouts/." >&2
    exit 1
fi

DOCC="$SWIFT_DOCC_DIR/.build/debug/docc"
if [ ! -x "$DOCC" ]; then
    echo "Building docc from $SWIFT_DOCC_DIR..."
    swift build --package-path "$SWIFT_DOCC_DIR" --product docc
fi

# HTML render templates from the system toolchain
DOCC_HTML_DIR="$(xcrun --find docc | sed 's|/bin/docc$|/share/docc/render|')"

# Discover library product targets from Package.swift.
TARGETS=$(swift package dump-package | python3 -c "
import json, sys
pkg = json.load(sys.stdin)
targets = set()
for p in pkg['products']:
    if p['type'].get('library') is not None:
        targets.update(p['targets'])
for t in sorted(targets):
    print(t)
")

# Extract symbol graphs for every library target by running a combined docs
# build. The archive output is discarded; we only need the side-effect of
# populating .build/.../extracted-symbols/.
TARGET_ARGS=()
while IFS= read -r TARGET; do
    TARGET_ARGS+=(--target "$TARGET")
done <<< "$TARGETS"

echo "Extracting symbol graphs..."
swift package generate-documentation \
    --enable-experimental-combined-documentation \
    "${TARGET_ARGS[@]}" \
    > /dev/null

# Locate the extracted-symbols parent directory. The subdirectory name is
# derived from the package identity (often the repo directory), not from
# `Package.swift`'s `name:` field, so glob for the only child instead of
# constructing the path.
EXTRACTED_PARENT=$(find .build -type d -name extracted-symbols -print -quit)
if [ -z "$EXTRACTED_PARENT" ]; then
    echo "Failed to locate extracted-symbols directory under .build." >&2
    exit 1
fi
SYMBOL_GRAPH_DIR=$(find "$EXTRACTED_PARENT" -mindepth 1 -maxdepth 1 -type d -print -quit)
if [ -z "$SYMBOL_GRAPH_DIR" ]; then
    echo "extracted-symbols at $EXTRACTED_PARENT contains no package directory." >&2
    exit 1
fi

# Preview LMResponses as the home; symbols from sibling targets remain
# resolvable via the combined symbol-graph directory passed below.
CATALOG="Sources/LMResponses/Documentation.docc"

echo "Starting preview server..."
DOCC_HTML_DIR="$DOCC_HTML_DIR" \
"$DOCC" preview "$CATALOG" \
    --additional-symbol-graph-dir "$SYMBOL_GRAPH_DIR" \
    --fallback-display-name LMResponses \
    --fallback-bundle-identifier LMResponses &
CHILD_PID=$!

wait "$CHILD_PID"
