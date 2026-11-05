#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v pixi >/dev/null 2>&1; then
    echo "ERROR: pixi not found in PATH" >&2
    exit 127
fi

detect_env() {
    if [[ -n "${STAGE11_PIXI_ENVIRONMENT:-}" ]]; then
        printf '%s\n' "$STAGE11_PIXI_ENVIRONMENT"
        return
    fi

    local gpu=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"
    fi

    if printf '%s' "$gpu" | grep -Eqi 'RTX[[:space:]]*5090|Blackwell'; then
        if grep -q 'remote-blackwell' "$PROJECT_ROOT/pyproject.toml" 2>/dev/null; then
            printf '%s\n' "remote-blackwell"
        else
            # Compatibility with the older 5090 package whose cu130 profile was
            # named simply "remote".
            printf '%s\n' "remote"
        fi
        return
    fi

    if [[ "$(uname -s 2>/dev/null || true)" == "Linux" ]]; then
        printf '%s\n' "remote"
    else
        printf '%s\n' "default"
    fi
}

PIXI_ENVIRONMENT="$(detect_env)"

# Hard guard: this exploratory mechanism experiment is standard split only.
unset PINN_STARLIGHT_SPLIT_MODE || true
unset PINN_STARLIGHT_CV_FOLD || true

echo
echo "=== Frozen PINN preservation exploratory validation ==="
echo "Project   : $PROJECT_ROOT"
echo "Pixi env  : $PIXI_ENVIRONMENT"
echo "Boundary  : VALIDATION ONLY; no A/B test rows"
echo "Claims    : no Pareto claim; no runtime comparison"
echo

exec pixi run -e "$PIXI_ENVIRONMENT" --locked \
    python -u -m research_diagnostics.pinn.pinn_preservation_frozen "$@"
