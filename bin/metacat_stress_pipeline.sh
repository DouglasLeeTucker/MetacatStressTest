#!/usr/bin/env bash
#
# metacat_stress_pipeline.sh
#
# Combined MetaCat stress-test pipeline:
#   1) Generate synthetic files
#   2) Declare synthetic files
#   3) Create child-of-N-parents file
#   4) Declare child file
#
# Exits immediately on any error.

set -euo pipefail

########################################
# Defaults
########################################

NAMESPACE_DEFAULT="dtucker_metacat_stress_tests"
DATASET_DEFAULT="metacat_stress_test_20260212"
WORKDIR_DEFAULT="$HOME/WORK/GitHub/MetacatStressTest/python"
SYNTHETIC_DIR_DEFAULT="synthetic_minimal_n100000"
CHILD_NAME_DEFAULT="metacat_stress_test_20260212_child_of_100000_parents"
NUM_PARENTS_DEFAULT=100000

RUN_GENERATE=false
RUN_DECLARE_SYNTHETIC=false
RUN_CREATE_CHILD=false
RUN_DECLARE_CHILD=false
DEBUG_ENV=false

NAMESPACE="$NAMESPACE_DEFAULT"
DATASET="$DATASET_DEFAULT"
WORKDIR="$WORKDIR_DEFAULT"
SYNTHETIC_DIR="$SYNTHETIC_DIR_DEFAULT"
CHILD_NAME="$CHILD_NAME_DEFAULT"
NUM_PARENTS="$NUM_PARENTS_DEFAULT"

VERBOSE=false
DRY_RUN=false
LOGFILE=""

########################################
# Helpers
########################################

usage() {
    cat <<EOF
Usage: $0 [options] [stage flags]

Options:
  --namespace <ns>
  --dataset <name>
  --workdir <path>
  --synthetic-dir <path>
  --child-name <name>
  --num-parents <N>
  --verbose
  --dry-run
  --logfile <path>
  --debug-env        Print environment diagnostics and exit

Stage flags:
  --run-generate
  --run-declare-synthetic
  --run-create-child
  --run-declare-child
EOF
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
    local cmd="$*"
    if $VERBOSE; then log "CMD: $cmd"; fi
    if $DRY_RUN; then
        log "[DRY-RUN] Would run: $cmd"
    else
        eval "$cmd"
    fi
}

time_stage() {
    local label="$1"; shift
    log "=== START: $label ==="
    local start end
    start=$(date +%s)
    "$@"
    end=$(date +%s)
    log "=== END:   $label (duration: $((end - start)) s) ==="
}

########################################
# Parse arguments
########################################

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace) NAMESPACE="$2"; shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        --workdir) WORKDIR="$2"; shift 2;;
        --synthetic-dir) SYNTHETIC_DIR="$2"; shift 2;;
        --child-name) CHILD_NAME="$2"; shift 2;;
        --num-parents) NUM_PARENTS="$2"; shift 2;;
        --verbose) VERBOSE=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --logfile) LOGFILE="$2"; shift 2;;
        --run-generate) RUN_GENERATE=true; shift;;
        --run-declare-synthetic) RUN_DECLARE_SYNTHETIC=true; shift;;
        --run-create-child) RUN_CREATE_CHILD=true; shift;;
        --run-declare-child) RUN_DECLARE_CHILD=true; shift;;
        --debug-env) DEBUG_ENV=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1"; usage; exit 1;;
    esac
done

########################################
# Logging setup
########################################

if [[ -n "$LOGFILE" ]]; then
    mkdir -p "$(dirname "$LOGFILE")"
    exec > >(tee -a "$LOGFILE") 2>&1
fi

log "Starting MetaCat stress pipeline"
log "Namespace:      $NAMESPACE"
log "Dataset:        $DATASET"
log "Workdir:        $WORKDIR"
log "Synthetic dir:  $SYNTHETIC_DIR"
log "Child name:     $CHILD_NAME"
log "Num parents:    $NUM_PARENTS"
log "Verbose:        $VERBOSE"
log "Dry-run:        $DRY_RUN"
log "Debug-env:      $DEBUG_ENV"

########################################
# Environment setup (Spack first, Conda second)
########################################

setup_env() {

    # Load MetaCat + Spack first
    run_cmd "source /cvmfs/fermilab.opensciencegrid.org/packages/common/setup-env.sh"
    run_cmd "spack load r-m-dd-config@1.8 experiment=hypot"
    run_cmd "htgettoken -i hypot -a htvaultprod.fnal.gov"
    run_cmd "metacat auth login -m token \$USER"

    # CRITICAL FIX: remove Spack's Python from import paths
    unset PYTHONPATH

    # Now activate Conda so its Python and packages win
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate metacat-stress

    if $VERBOSE; then
        log "Python in use: $(which python3)"
        log "Python version: $(python3 --version)"
    fi
}

time_stage "Environment setup" setup_env

########################################
# Debug environment mode
########################################

if $DEBUG_ENV; then
    log "=== DEBUG ENVIRONMENT ==="
    echo "PATH=$PATH"
    echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
    echo
    echo "which python3:"
    which python3 || echo "python3 not found"
    echo
    echo "type -a python3:"
    type -a python3 || echo "python3 not found in type -a"
    echo
    echo "python3 --version:"
    python3 --version || echo "python3 --version failed"
    echo
    echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
    echo
    echo "Environment variables matching CONDA|PATH|PYTHON:"
    env | grep -E 'CONDA|PATH|PYTHON' || echo "No matching env vars"
    echo
    if [[ -f "$WORKDIR/synthetic_generator_minimal.py" ]]; then
        echo "Shebang of synthetic_generator_minimal.py:"
        head -n 1 "$WORKDIR/synthetic_generator_minimal.py"
    else
        echo "synthetic_generator_minimal.py not found at $WORKDIR"
    fi
    echo
    log "Debug-env mode complete; exiting."
    exit 0
fi

########################################
# Verify commands AFTER environment setup
########################################

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command '$1' not found" >&2
        exit 1
    fi
}

require_cmd metacat
require_cmd jq
require_cmd python3
require_cmd htgettoken
require_cmd spack

########################################
# Stage 1: Generate synthetic files
########################################

stage_generate() {
    run_cmd "cd \"$WORKDIR\""
    local outdir="$WORKDIR/$SYNTHETIC_DIR"
    run_cmd "mkdir -p \"$outdir\""

    # Force Conda python3
    run_cmd "python3 ./synthetic_generator_minimal.py \
        -o \"$outdir\" \
        -n $NUM_PARENTS \
	--metacat-batch "batch_${DATASET}_1.json" \
        --min-rows 50 \
        --max-rows 200 \
        --namespace \"$NAMESPACE\" \
        --owner \"$USER\" \
        --dataset-prefix \"$DATASET\" \
        --num-datasets 1"
}

if $RUN_GENERATE; then
    time_stage "Stage 1: Generate synthetic files" stage_generate
else
    log "Skipping Stage 1"
fi

########################################
# Stage 2: Declare synthetic files
########################################

stage_declare_synthetic() {
    local outdir="$WORKDIR/$SYNTHETIC_DIR"
    run_cmd "cd \"$outdir\""

    if ! $DRY_RUN; then
        metacat dataset create "$NAMESPACE:$DATASET" || \
            log "Dataset exists; continuing."
    else
        run_cmd "metacat dataset create \"$NAMESPACE:$DATASET\" || true"
    fi

    run_cmd "metacat dataset list \"$NAMESPACE:*\" -cl"

    local batch_json="batch_${DATASET}_1.json"
    if [[ ! -f "$batch_json" && $DRY_RUN == false ]]; then
        echo "Error: batch JSON '$batch_json' not found" >&2
        exit 1
    fi

    run_cmd "metacat file declare-many \"$batch_json\" \"$NAMESPACE:$DATASET\""
    run_cmd "metacat dataset files \"$NAMESPACE:$DATASET\" -m | wc"
}

if $RUN_DECLARE_SYNTHETIC; then
    time_stage "Stage 2: Declare synthetic files" stage_declare_synthetic
else
    log "Skipping Stage 2"
fi

########################################
# Stage 3: Create child-of-N-parents file
########################################

stage_create_child() {
    local outdir="$WORKDIR/$SYNTHETIC_DIR"
    run_cmd "cd \"$outdir\""

    if $DRY_RUN; then
        log "[DRY-RUN] Would run Python child generator"
        return
    fi

    python3 - <<EOF
import pyarrow as pa, pyarrow.parquet as pq, zlib, os, json, subprocess, sys

namespace = "${NAMESPACE}"
dataset   = "${DATASET}"
child_name = "${CHILD_NAME}"
expected_parents = int("${NUM_PARENTS}")

table = pa.table({"value": [1, 2, 3]})
child_filename = "child_of_{}_parents.parquet".format(expected_parents)
pq.write_table(table, child_filename)

size = os.path.getsize(child_filename)
with open(child_filename, "rb") as f:
    checksum = format(zlib.adler32(f.read()) & 0xffffffff, "08x")

cmd = ["metacat", "query", f"files from {namespace}:{dataset}"]
parents_raw = subprocess.check_output(cmd).decode().strip().split("\\n")
parents = [p.strip() for p in parents_raw if p.strip()]

if len(parents) != expected_parents:
    print(f"ERROR: Expected {expected_parents} parents, got {len(parents)}", file=sys.stderr)
    sys.exit(1)

entry = {
    "namespace": namespace,
    "name": child_name,
    "size": size,
    "checksum": checksum,
    "checksum_type": "adler32",
    "metadata": {
        "fn.description": f"Child of {expected_parents} synthetic files",
        "fn.owner": "${USER}"
    },
    "parents": [ {"did": did} for did in parents ],
    "file_path": os.path.abspath(child_filename)
}

json_name = "child_of_{}_parents_entry.json".format(expected_parents)
with open(json_name, "w") as f:
    json.dump(entry, f, indent=2)

print(f"Created {json_name} with {len(parents)} parents.")
EOF
}

if $RUN_CREATE_CHILD; then
    time_stage "Stage 3: Create child-of-N-parents file" stage_create_child
else
    log "Skipping Stage 3"
fi

########################################
# Stage 4: Declare child-of-N-parents file
########################################

stage_declare_child() {
    local outdir="$WORKDIR/$SYNTHETIC_DIR"
    run_cmd "cd \"$outdir\""

    local json_name="child_of_${NUM_PARENTS}_parents_entry.json"
    if [[ ! -f "$json_name" && $DRY_RUN == false ]]; then
        echo "Error: child JSON '$json_name' not found" >&2
        exit 1
    fi
    run_cmd "metacat file declare -f \"$json_name\" \"$NAMESPACE:$DATASET\""

    if $VERBOSE && ! $DRY_RUN; then
        run_cmd "metacat file show \"$NAMESPACE:$CHILD_NAME\""
        #run_cmd "metacat file parents \"$NAMESPACE:$CHILD_NAME\" | wc -l"
	# Try modern MetaCat CLI first
	PARENT_COUNT=$(
	    metacat file parents "$NAMESPACE:$CHILD_NAME" 2>/dev/null | wc -l || echo ""
		    )
	# If the command failed (empty output), fall back to query engine
	if [[ -z "$PARENT_COUNT" || "$PARENT_COUNT" -le 1 ]]; then
	    # Query engine returns one DID per line
	    PARENT_COUNT=$(
		metacat query "parents($NAMESPACE:$CHILD_NAME)" 2>/dev/null | wc -l
			)
	fi
	log "Parent count for $CHILD_NAME: $PARENT_COUNT"
    fi
}

if $RUN_DECLARE_CHILD; then
    time_stage "Stage 4: Declare child-of-N-parents file" stage_declare_child
else
    log "Skipping Stage 4"
fi

log "MetaCat stress pipeline completed."
