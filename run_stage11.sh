#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

COMMAND="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

if ! command -v pixi >/dev/null 2>&1; then
    echo "ERROR: Pixi was not found in PATH." >&2
    echo "Install Pixi first, then reopen the shell." >&2
    exit 127
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi was not found. Verify the NVIDIA driver first." >&2
    exit 127
fi

if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo "ERROR: pyproject.toml not found in $PROJECT_ROOT" >&2
    exit 2
fi

if [[ ! -f "$PROJECT_ROOT/pixi.lock" ]]; then
    echo "ERROR: pixi.lock is missing." >&2
    echo "Run 'pixi lock' once after replacing pyproject.toml." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

mapfile -t GPU_ROWS < <(
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null
)
if (( ${#GPU_ROWS[@]} == 0 )); then
    echo "ERROR: nvidia-smi reported zero GPUs." >&2
    exit 2
fi

GPU_NAMES=()
DRIVER_MAJORS=()
HAS_BLACKWELL=0

for row in "${GPU_ROWS[@]}"; do
    name="${row%,*}"
    driver="${row##*,}"
    name="$(printf '%s' "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    driver="$(printf '%s' "$driver" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    GPU_NAMES+=("$name")
    DRIVER_MAJORS+=("${driver%%.*}")

    if printf '%s' "$name" | grep -Eqi \
        'RTX[[:space:]]*(PRO[[:space:]]*)?50[0-9]{2}|Blackwell|B100|B200|GB200|GB300'; then
        HAS_BLACKWELL=1
    fi
done

REQUESTED_ENV="${STAGE11_ENVIRONMENT:-auto}"

if [[ "$REQUESTED_ENV" != "auto" ]]; then
    PIXI_ENVIRONMENT="$REQUESTED_ENV"
    if (( HAS_BLACKWELL == 1 )) && [[ "$PIXI_ENVIRONMENT" != "remote-blackwell" ]]; then
        echo "ERROR: Blackwell/RTX-50 GPU detected; use remote-blackwell or auto." >&2
        exit 2
    fi
else
    if (( HAS_BLACKWELL == 1 )); then
        PIXI_ENVIRONMENT="remote-blackwell"
    elif [[ -f "$PROJECT_ROOT/.stage11_local" ]]; then
        PIXI_ENVIRONMENT="default"
    else
        PIXI_ENVIRONMENT="remote"
    fi
fi

MIN_DRIVER_MAJOR=99999
for major in "${DRIVER_MAJORS[@]}"; do
    if [[ "$major" =~ ^[0-9]+$ ]] && (( major < MIN_DRIVER_MAJOR )); then
        MIN_DRIVER_MAJOR="$major"
    fi
done

if [[ "$PIXI_ENVIRONMENT" == "remote-blackwell" ]]; then
    if (( MIN_DRIVER_MAJOR < 580 )); then
        echo "ERROR: remote-blackwell uses cu130 and needs an R580+ driver." >&2
        echo "Detected minimum driver major: $MIN_DRIVER_MAJOR" >&2
        exit 2
    fi
else
    if (( MIN_DRIVER_MAJOR < 525 )); then
        echo "ERROR: cu126 profile needs an R525+ CUDA-12-family driver on Linux." >&2
        echo "Detected minimum driver major: $MIN_DRIVER_MAJOR" >&2
        exit 2
    fi
fi

echo
echo "=== Stage11 launcher ==="
echo "Host platform : $(uname -s 2>/dev/null || echo unknown)"
for row in "${GPU_ROWS[@]}"; do
    echo "GPU           : $row"
done
echo "Pixi env      : $PIXI_ENVIRONMENT"
if [[ "$PIXI_ENVIRONMENT" == "remote-blackwell" ]]; then
    echo "CUDA profile  : torch 2.14.0 + cu130"
else
    echo "CUDA profile  : torch 2.14.0 + cu126"
fi
echo "GPU scheduler : ${STAGE11_GPUS:-auto} (CV5 auto task parallelism)"
echo "Data policy   : missing A/B data are generated; existing formal data are reused"
echo "                --force retrains only; --force-data explicitly regenerates datasets"
echo

RUNNER_ARGS=("$COMMAND")
HAS_GPUS_ARG=0
HAS_MAX_PARALLEL_ARG=0
for arg in "$@"; do
    case "$arg" in
        --gpus|--gpus=*) HAS_GPUS_ARG=1 ;;
        --max-parallel|--max-parallel=*) HAS_MAX_PARALLEL_ARG=1 ;;
    esac
done

if (( HAS_GPUS_ARG == 0 )); then
    RUNNER_ARGS+=("--gpus" "${STAGE11_GPUS:-auto}")
fi
if [[ -n "${STAGE11_MAX_PARALLEL:-}" ]] && (( HAS_MAX_PARALLEL_ARG == 0 )); then
    RUNNER_ARGS+=("--max-parallel" "$STAGE11_MAX_PARALLEL")
fi

RUNNER_ARGS+=("$@")

exec pixi run -e "$PIXI_ENVIRONMENT" --locked \
    python -u run_stage11.py "${RUNNER_ARGS[@]}"
