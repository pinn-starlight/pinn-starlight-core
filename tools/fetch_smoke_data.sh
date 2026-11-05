#!/usr/bin/env bash
# Extract the minimal dataset needed by `python -m tools.smoke_test`.
#
# The formal study uses ~10.85 GB of synthetic arrays plus 16 real camera
# images.  The smoke test only needs two validation samples (one Stage-A, one
# Stage-B), the two manifests and the frozen configuration locks, which is
# roughly 280 MB.
#
# Usage:
#   bash tools/fetch_smoke_data.sh                 # from the repo root
#   ARCHIVES=/path/to/stage11_transfer_archives bash tools/fetch_smoke_data.sh
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ARCHIVES="${ARCHIVES:-$PROJECT_ROOT/artifacts/transfers/stage11_transfer_archives}"
STAGE_A_SAMPLE="${STAGE_A_SAMPLE:-clean_single_002_multi_source_high}"
STAGE_B_SAMPLE="${STAGE_B_SAMPLE:-clean_single_002_protected_overlap_high}"

need() {
    [[ -f "$1" ]] || { echo "ERROR: missing archive: $1" >&2; exit 1; }
}

echo "archives : $ARCHIVES"
echo "stage-A  : $STAGE_A_SAMPLE"
echo "stage-B  : $STAGE_B_SAMPLE"
echo

need "$ARCHIVES/archive_inventory.csv"

# Locate the archive that holds a given repository path.
#   dir/*  -> prefix match on the inventory relative_path column
#   file   -> exact match
archive_for() {
    local pattern="$1"
    if [[ "$pattern" == *"/*" ]]; then
        local prefix="${pattern%\*}"
        awk -F, -v p="$prefix" 'NR>1 && index($5,p)==1 {print $8; exit}' \
            "$ARCHIVES/archive_inventory.csv" | tr -d '\r'
    else
        awk -F, -v p="$pattern" 'NR>1 && $5==p {print $8; exit}' \
            "$ARCHIVES/archive_inventory.csv" | tr -d '\r'
    fi
}

extract() {
    local pattern="$1" archive
    archive="$(archive_for "$pattern")"
    if [[ -z "$archive" ]]; then
        echo "ERROR: no archive records $pattern" >&2
        exit 1
    fi
    echo "  <- $archive :: $pattern"
    unzip -o -q "$ARCHIVES/$archive" "$pattern" -d "$PROJECT_ROOT"
}

echo "Extracting Stage-A validation sample..."
extract "collections/synthetic_formal/${STAGE_A_SAMPLE}/*"
extract "collections/synthetic_formal/manifest.csv"

echo "Extracting Stage-B validation sample..."
extract "collections/synthetic_stage_b/${STAGE_B_SAMPLE}/*"
extract "collections/synthetic_stage_b/manifest.csv"

echo "Extracting frozen configuration locks..."
extract "experiments/outputs/configuration_lock/global_configuration_lock.json"
extract "experiments/outputs/stage_b/configuration_lock/stage_b_global_configuration_lock.json"

echo
echo "Done. Sizes:"
du -sh collections/synthetic_formal collections/synthetic_stage_b experiments/outputs 2>/dev/null || true
echo
echo "Next:  python -m tools.smoke_test"
