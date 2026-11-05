#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v pixi >/dev/null 2>&1; then
    echo "ERROR: Pixi was not found in PATH." >&2
    exit 127
fi
if [[ ! -f "$PROJECT_ROOT/pyproject.toml" || ! -f "$PROJECT_ROOT/pixi.lock" ]]; then
    echo "ERROR: run this from the Stage11 project root with pyproject.toml and pixi.lock." >&2
    exit 2
fi

REQUESTED_ENV="${STAGE11_ENVIRONMENT:-auto}"
if [[ "$REQUESTED_ENV" != "auto" ]]; then
    PIXI_ENVIRONMENT="$REQUESTED_ENV"
else
    PIXI_ENVIRONMENT="remote"
    if [[ -f "$PROJECT_ROOT/.stage11_local" ]]; then
        PIXI_ENVIRONMENT="default"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | \
            grep -Eqi 'RTX[[:space:]]*(PRO[[:space:]]*)?50[0-9]{2}|Blackwell|B100|B200|GB200|GB300'; then
            PIXI_ENVIRONMENT="remote-blackwell"
        fi
    fi
fi

echo "=== Stage11 forced-reference metric backfill ==="
echo "Project root : $PROJECT_ROOT"
echo "Pixi env     : $PIXI_ENVIRONMENT"
echo "Policy       : frozen predictions only; no training/selection/lock modification"
echo

exec pixi run -e "$PIXI_ENVIRONMENT" --locked \
    python -u tools/backfill_forced_reference_metrics.py "$@"
