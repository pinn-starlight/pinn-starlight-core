#!/usr/bin/env bash
# Run the mechanism probes (alpha_sweep / init_memory) with pre-flight checks.
#
#   bash tools/run_diagnostics.sh                     # both probes, --suite full
#   bash tools/run_diagnostics.sh --suite smoke       # plumbing only, CPU ok
#   bash tools/run_diagnostics.sh --only alpha --dry-run
#
# Both probes are read-only diagnostics: validation split, no config lock, no
# formal output directory.  They do not change seeds, budgets or protocol.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUITE="full"
ONLY="alpha,init"
GPUS=""
SERIAL=0
DRY_RUN=0
CURVES=0
PLOT=1
ALLOW_CPU=0
FORCE=0
MAX_SAMPLES=""
SAMPLES=""
DATASET="stage_a"
DEVICE=""
LOG_DIR=""

usage() {
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  -s, --suite smoke|full   budget (default: full)
      --only alpha,init    which probes to run (default: both)
  -g, --gpus 0,1           physical GPUs to use, one probe per card
      --serial             run the probes one after another (one card)
      --device DEVICE      override the python --device value (e.g. cuda:0, cpu)
      --max-samples N      cap validation samples per probe (subset result)
      --samples A,B        explicit validation sample ids (needs --only)
      --dataset D          alpha_sweep dataset: stage_a|stage_b (default stage_a)
      --curves             init_memory: also run the truncated step curves
      --no-plot            alpha_sweep: skip alpha_curves.png
      --allow-cpu          permit --suite full on CPU (hours, not representative)
      --force              start even if another probe instance is running
      --log-dir DIR        where the logs go (default build/diag_logs/<stamp>)
      --dry-run            print the resolved commands and exit

Environment:
  PYTHON_CMD   explicit interpreter launcher, e.g. "uv run python"
  PIXI_ENV     pixi environment name when pixi is used (default: remote)
  ALPHA_ARGS   extra arguments appended to the alpha_sweep command
  INIT_ARGS    extra arguments appended to the init_memory command
EOF
}

while (($# > 0)); do
    case "$1" in
        -s | --suite) SUITE="${2:?--suite needs a value}"; shift 2 ;;
        --only) ONLY="${2:?--only needs a value}"; shift 2 ;;
        -g | --gpus) GPUS="${2:?--gpus needs a value}"; shift 2 ;;
        --serial) SERIAL=1; shift ;;
        --device) DEVICE="${2:?--device needs a value}"; shift 2 ;;
        --max-samples) MAX_SAMPLES="${2:?--max-samples needs a value}"; shift 2 ;;
        --samples) SAMPLES="${2:?--samples needs a value}"; shift 2 ;;
        --dataset) DATASET="${2:?--dataset needs a value}"; shift 2 ;;
        --curves) CURVES=1; shift ;;
        --no-plot) PLOT=0; shift ;;
        --allow-cpu) ALLOW_CPU=1; shift ;;
        --force) FORCE=1; shift ;;
        --log-dir) LOG_DIR="${2:?--log-dir needs a value}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$SUITE" in
    smoke | full) ;;
    *) echo "ERROR: --suite must be smoke or full, got '$SUITE'" >&2; exit 2 ;;
esac
case "$ONLY" in
    alpha | init | alpha,init | init,alpha) ;;
    *) echo "ERROR: --only must be alpha, init or alpha,init" >&2; exit 2 ;;
esac
RUN_ALPHA=0
RUN_INIT=0
[[ "$ONLY" == *alpha* ]] && RUN_ALPHA=1
[[ "$ONLY" == *init* ]] && RUN_INIT=1
if [[ -n "$SAMPLES" && $((RUN_ALPHA + RUN_INIT)) -ne 1 ]]; then
    echo "ERROR: --samples needs exactly one probe (--only alpha or --only init)" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# interpreter: explicit override > pixi > .venv > PATH python
# --------------------------------------------------------------------------
PY=()
if [[ -n "${PYTHON_CMD:-}" ]]; then
    read -r -a PY <<<"$PYTHON_CMD"
elif command -v pixi >/dev/null 2>&1 && [[ -f "$ROOT/pixi.lock" ]]; then
    PY=(pixi run -e "${PIXI_ENV:-remote}" python)
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY=("$ROOT/.venv/bin/python")
else
    PY=(python)
fi

echo "=== mechanism probes ==="
echo "repository : $ROOT"
echo "interpreter: ${PY[*]}"
echo "suite      : $SUITE"
echo "probes     : $ONLY"

# --------------------------------------------------------------------------
# pre-flight: interpreter, torch, CUDA
# --------------------------------------------------------------------------
if ! ENV_INFO="$("${PY[@]}" - <<'PY' 2>&1
import sys
import torch
print(f"python {sys.version.split()[0]} | torch {torch.__version__}")
print(f"cuda available {torch.cuda.is_available()} | devices {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print("gpus " + ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
PY
)"; then
    echo "ERROR: interpreter check failed:" >&2
    echo "$ENV_INFO" >&2
    exit 2
fi
echo "$ENV_INFO" | sed 's/^/  /'

CUDA_OK=0
if echo "$ENV_INFO" | grep -q "^cuda available True"; then
    CUDA_OK=1
fi

if [[ "$SUITE" == "full" && $CUDA_OK -eq 0 && $ALLOW_CPU -eq 0 ]]; then
    echo "ERROR: --suite full needs CUDA; use --suite smoke, or --allow-cpu to force." >&2
    exit 2
fi

# --------------------------------------------------------------------------
# pre-flight: validation samples on disk
# --------------------------------------------------------------------------
echo "--- validation manifest ---"
"${PY[@]}" - <<'PY'
from experiments.scripts.diagnostics import _harness as harness

for dataset in ("stage_a", "stage_b"):
    try:
        rows = harness.load_rows(dataset)
    except Exception as exc:  # noqa: BLE001 - report, do not abort
        print(f"  {dataset}: ERROR {exc}")
        continue
    print(f"  {dataset}: {len(rows)} validation samples")
PY

# --------------------------------------------------------------------------
# pre-flight: refuse to stack a second copy on the same card
# --------------------------------------------------------------------------
if [[ $FORCE -eq 0 ]] && command -v pgrep >/dev/null 2>&1; then
    RUNNING="$(pgrep -af 'experiments\.scripts\.diagnostics\.(alpha_sweep|init_memory)' || true)"
    if [[ -n "$RUNNING" ]]; then
        echo "ERROR: a probe is already running; --force to start anyway:" >&2
        echo "$RUNNING" | sed 's/^/  /' >&2
        exit 2
    fi
fi

# --------------------------------------------------------------------------
# GPU assignment
# --------------------------------------------------------------------------
GPU_LIST=()
if [[ -n "$GPUS" ]]; then
    IFS=',' read -r -a GPU_LIST <<<"$GPUS"
elif [[ $CUDA_OK -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r index; do
        GPU_LIST+=("${index//[[:space:]]/}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
fi

if [[ -z "$DEVICE" ]]; then
    if [[ ${#GPU_LIST[@]} -gt 0 ]]; then
        DEVICE="cuda:0"
    else
        DEVICE="cpu"
    fi
fi

if [[ "$DEVICE" == cpu* && "$SUITE" == "full" && $ALLOW_CPU -eq 0 ]]; then
    echo "ERROR: --suite full on cpu; pass --device cuda:0 or --allow-cpu." >&2
    exit 2
fi

if [[ $SERIAL -eq 1 || ${#GPU_LIST[@]} -lt 2 ]]; then
    PARALLEL=0
else
    PARALLEL=1
fi
if [[ $((RUN_ALPHA + RUN_INIT)) -lt 2 ]]; then
    PARALLEL=0
fi

ALPHA_GPU="${GPU_LIST[0]:-}"
INIT_GPU="${GPU_LIST[1]:-${GPU_LIST[0]:-}}"
if [[ $PARALLEL -eq 0 ]]; then
    INIT_GPU=""
fi

echo "device     : $DEVICE"
if [[ ${#GPU_LIST[@]} -gt 0 ]]; then
    echo "gpus       : ${GPU_LIST[*]} | mode: $([[ $PARALLEL -eq 1 ]] && echo parallel || echo serial)"
fi

# --------------------------------------------------------------------------
# command lines
# --------------------------------------------------------------------------
ALPHA_CMD=("${PY[@]}" -m experiments.scripts.diagnostics.alpha_sweep
    --suite "$SUITE" --dataset "$DATASET" --device "$DEVICE")
if [[ $PLOT -eq 1 ]]; then ALPHA_CMD+=(--plot); fi
if [[ -n "$MAX_SAMPLES" ]]; then ALPHA_CMD+=(--max-samples "$MAX_SAMPLES"); fi
if [[ -n "$SAMPLES" && $RUN_ALPHA -eq 1 ]]; then ALPHA_CMD+=(--samples "$SAMPLES"); fi
if [[ -n "${ALPHA_ARGS:-}" ]]; then read -r -a EXTRA <<<"$ALPHA_ARGS"; ALPHA_CMD+=("${EXTRA[@]}"); fi

INIT_CMD=("${PY[@]}" -m experiments.scripts.diagnostics.init_memory
    --suite "$SUITE" --device "$DEVICE")
if [[ $CURVES -eq 1 ]]; then INIT_CMD+=(--curves); fi
if [[ -n "$MAX_SAMPLES" ]]; then INIT_CMD+=(--max-samples "$MAX_SAMPLES"); fi
if [[ -n "$SAMPLES" && $RUN_INIT -eq 1 ]]; then INIT_CMD+=(--samples "$SAMPLES"); fi
if [[ -n "${INIT_ARGS:-}" ]]; then read -r -a EXTRA <<<"$INIT_ARGS"; INIT_CMD+=("${EXTRA[@]}"); fi

# Serial runs keep the tqdm bars (you are watching the terminal); parallel runs
# write to log files, where a refreshed bar would only add carriage returns.
if [[ $PARALLEL -eq 1 ]]; then
    ALPHA_CMD+=(--no-progress)
    INIT_CMD+=(--no-progress)
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "--- dry run ---"
    if [[ $RUN_ALPHA -eq 1 ]]; then
        if [[ -n "$ALPHA_GPU" ]]; then
            echo "CUDA_VISIBLE_DEVICES=$ALPHA_GPU ${ALPHA_CMD[*]}"
        else
            echo "${ALPHA_CMD[*]}"
        fi
    fi
    if [[ $RUN_INIT -eq 1 ]]; then
        if [[ -n "$INIT_GPU" ]]; then
            echo "CUDA_VISIBLE_DEVICES=$INIT_GPU ${INIT_CMD[*]}"
        else
            echo "${INIT_CMD[*]}"
        fi
    fi
    exit 0
fi

# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT/build/diag_logs/$STAMP}"
mkdir -p "$LOG_DIR"
echo "logs       : ${LOG_DIR#"$ROOT"/}"

STARTED_AT=$SECONDS
ALPHA_STATUS=0
INIT_STATUS=0

run_serial() {
    local name="$1" gpu="$2"
    shift 2
    local log="$LOG_DIR/$name.log"
    local started=$SECONDS
    local status=0
    if [[ -n "$gpu" ]]; then
        env CUDA_VISIBLE_DEVICES="$gpu" "$@" 2>&1 | tee "$log" || status=$?
    else
        "$@" 2>&1 | tee "$log" || status=$?
    fi
    echo "[$name] finished in $((SECONDS - started))s (exit $status)"
    return "$status"
}

if [[ $PARALLEL -eq 1 ]]; then
    ALPHA_PID=""
    INIT_PID=""
    if [[ $RUN_ALPHA -eq 1 ]]; then
        env CUDA_VISIBLE_DEVICES="$ALPHA_GPU" "${ALPHA_CMD[@]}" \
            >"$LOG_DIR/alpha.log" 2>&1 &
        ALPHA_PID=$!
        echo "alpha_sweep -> gpu ${ALPHA_GPU:-default} (pid $ALPHA_PID, log alpha.log)"
    fi
    if [[ $RUN_INIT -eq 1 ]]; then
        env CUDA_VISIBLE_DEVICES="$INIT_GPU" "${INIT_CMD[@]}" \
            >"$LOG_DIR/init.log" 2>&1 &
        INIT_PID=$!
        echo "init_memory -> gpu ${INIT_GPU:-default} (pid $INIT_PID, log init.log)"
    fi
    [[ -n "$ALPHA_PID" ]] && { wait "$ALPHA_PID" || ALPHA_STATUS=$?; }
    [[ -n "$INIT_PID" ]] && { wait "$INIT_PID" || INIT_STATUS=$?; }
else
    [[ $RUN_ALPHA -eq 1 ]] && { run_serial alpha "$ALPHA_GPU" "${ALPHA_CMD[@]}" || ALPHA_STATUS=$?; }
    [[ $RUN_INIT -eq 1 ]] && { run_serial init "$INIT_GPU" "${INIT_CMD[@]}" || INIT_STATUS=$?; }
fi

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
echo
echo "=== result (${SECONDS}s total) ==="
if [[ $RUN_ALPHA -eq 1 ]]; then
    if [[ $ALPHA_STATUS -eq 0 ]]; then
        echo "alpha_sweep : OK"
    else
        echo "alpha_sweep : FAILED (exit $ALPHA_STATUS) -> tail $LOG_DIR/alpha.log"
    fi
fi
if [[ $RUN_INIT -eq 1 ]]; then
    if [[ $INIT_STATUS -eq 0 ]]; then
        echo "init_memory : OK"
    else
        echo "init_memory : FAILED (exit $INIT_STATUS) -> tail $LOG_DIR/init.log"
    fi
fi

if [[ $ALPHA_STATUS -eq 0 && $INIT_STATUS -eq 0 ]]; then
    echo
    echo "outputs (newest run per probe):"
    for probe in alpha_sweep init_memory; do
        [[ -d "experiments/outputs/diagnostics/$probe" ]] || continue
        newest="$(find "experiments/outputs/diagnostics/$probe" -mindepth 1 -maxdepth 1 \
            -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
        [[ -n "$newest" ]] && echo "  ${newest#"$ROOT"/}"
    done
    echo
    echo "send back: attribution.csv, anchor_curve.csv, findings.md, summary.json,"
    echo "           alpha_summary.json, summary.csv, alpha_curves.png"
fi

[[ $ALPHA_STATUS -eq 0 && $INIT_STATUS -eq 0 ]] || exit 1
