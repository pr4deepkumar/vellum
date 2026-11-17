#!/usr/bin/env bash
# Run VELLUM on a Linux + NVIDIA GPU box (an L4 or bigger) and write results/<gpu>/report.md:
# HF Transformers and vLLM on Mistral 7B, the from-scratch engine at every rung of its ladder
# (needs the `pip install -e ".[mistral]"` extra), and vLLM's AWQ and prefix caching under load.
#
#   GPU_COST=0.85 bash scripts/run_all.sh
#   STAGES="engine scratch" SCRATCH_RUNGS="naive gqa graph" GPU_COST=0.85 bash scripts/run_all.sh
#
# GPU_COST is what YOU pay for this GPU in USD/hr; it is only used for $/1M-token math. A stage
# that fails is logged and skipped, so the run always reaches the report.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
STAGES=${STAGES:-"engine scratch quant prefix"}
SCRATCH_RUNGS=${SCRATCH_RUNGS:-"naive lastlogit nosync gqa sdpa chunked graph awq triton"}
AWQ_QUALITY=${AWQ_QUALITY:-1}    # 0: skip the perplexity check (bf16 / AWQ / RTN), e.g. when only speed changed
AWQ_BASELINE=${AWQ_BASELINE:-1}  # 0: skip vLLM's AWQ yardstick, e.g. when it was already measured
if [[ -z "${AWQ_MODEL:-}" ]]; then
  if [[ "$MODEL" == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
    AWQ_MODEL=solidrust/Mistral-7B-Instruct-v0.3-AWQ  # an AutoAWQ build of the same model
  elif [[ " $STAGES " == *" quant "* ]]; then  # never compare one model's fp16 to another's AWQ
    echo "AWQ_MODEL is not set: the quant stage needs an AWQ build of $MODEL" >&2; exit 1
  fi
fi
CONCURRENCY=${CONCURRENCY:-"1 2 4 8 16 32 64"}
BATCH_SIZES=${BATCH_SIZES:-"1 2 4 8 16 32 64"}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
PREFIX_LEN=${PREFIX_LEN:-1024}      # shared "system prompt" for the prefix-cache experiment
PREFIX_UNIQUE=${PREFIX_UNIQUE:-256} # unique user turn appended to it
PORT=${PORT:-8000}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
OUT=${OUT:-results/$(echo "$GPU_NAME" | tr ' /' '--' | tr -cd '[:alnum:]-')}
mkdir -p "$OUT" logs

if [[ -z "${GPU_COST:-}" ]]; then
  echo "WARNING: GPU_COST (USD/hr) not set -- report will omit cost per 1M tokens." >&2
fi

# ---- record environment so results are reproducible / honest -------------------------
{
  echo "date: $(date -u +%FT%TZ)"
  echo "gpu: $GPU_NAME"
  nvidia-smi --query-gpu=memory.total,driver_version,compute_cap --format=csv,noheader | sed 's/^/gpu_info: /'
  python -c "import vllm, torch, transformers; print('vllm:', vllm.__version__); print('torch:', torch.__version__); print('transformers:', transformers.__version__)" \
    || echo "versions: could not import vllm/torch/transformers"
  echo "model: $MODEL"; echo "awq_model: ${AWQ_MODEL:-none}"
  echo "stages: $STAGES"
  if [[ " $STAGES " == *" scratch "* ]]; then echo "scratch_rungs: $SCRATCH_RUNGS"; fi
  echo "gpu_cost_per_hour: ${GPU_COST:-unset}"
} | tee "$OUT/env.txt"

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

start_server() {  # start_server <label> <model> [extra vllm args...]
  local label=$1 model=$2; shift 2
  echo "== starting vLLM server [$label]: $model $*"
  vllm serve "$model" --port "$PORT" --dtype half --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --seed 0 "$@" > "logs/server_$label.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 900); do
    if curl -sf "localhost:$PORT/health" > /dev/null; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server [$label] died; tail of logs/server_$label.log:"; tail -40 "logs/server_$label.log"; exit 1
    fi
    sleep 1
  done
  echo "server [$label] did not become healthy in 15 min"; exit 1
}

sweep() {  # sweep <label> <experiment> <server-args-string> [extra sweep args...]
  local label=$1 exp=$2 sargs=$3; shift 3
  python -m vellum sweep --base-url "http://localhost:$PORT" --label "$label" \
    --experiment "$exp" --server-args "$sargs" --out "$OUT" "$@"
}

stage() {  # stage <name> <function>: a failure is logged and the run goes on to the report
  local status
  set +e  # not `( ... ) || warn`: that would silently switch off set -e inside the stage
  ( set -e; trap stop_server EXIT; "$2" )
  status=$?
  set -e
  if ((status)); then echo "WARNING: stage $1 failed (exit $status); continuing to the report" >&2; fi
}

# ---- 1. engine: HF Transformers vs vLLM, same fixed batches, no server ---------------
engine_stage() {
  python -m vellum offline --backend hf   --model "$MODEL" --label hf-fp16   --experiment engine \
    --batch-sizes $BATCH_SIZES --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --out "$OUT"
  python -m vellum offline --backend vllm --model "$MODEL" --label vllm-fp16 --experiment engine \
    --batch-sizes $BATCH_SIZES --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --out "$OUT"
}

# ---- 2. scratch: the from-scratch Mistral, one run per rung of its optimization ladder --
# Same batches and prompts as the engine stage, under the same experiment, so the rungs chart
# against HF and vLLM. The awq and triton rungs run on a checkpoint `vellum quantize` writes
# first, with vLLM's own AWQ build of the model as their yardstick; scripts/bench_int4.py times
# every 4-bit matmul path behind triton. A failing rung is logged and skipped.
scratch_stage() {
  local rung model quality=()
  if [[ " $SCRATCH_RUNGS " == *" awq "* || " $SCRATCH_RUNGS " == *" triton "* ]]; then
    if [[ $AWQ_QUALITY == 1 ]]; then quality=(--quality "$OUT/awq_quality.json"); fi
    python -m vellum quantize --model "$MODEL" --out weights/awq ${quality[@]+"${quality[@]}"} \
      || echo "WARNING: quantize failed; the 4-bit rungs will be skipped" >&2
    if [[ " $SCRATCH_RUNGS " == *" triton "* ]]; then
      python scripts/bench_int4.py --out "$OUT/int4_kernels.json" > logs/bench_int4.log 2>&1 \
        || echo "WARNING: the 4-bit kernel benchmark failed; see logs/bench_int4.log" >&2
    fi
    if [[ $AWQ_BASELINE == 1 && -n "${AWQ_MODEL:-}" ]]; then
      python -m vellum offline --backend vllm --model "$AWQ_MODEL" --label vllm-awq --experiment engine \
        --batch-sizes $BATCH_SIZES --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --out "$OUT" \
        || echo "WARNING: vllm-awq failed; continuing" >&2
    fi
  fi
  for rung in $SCRATCH_RUNGS; do
    model=$MODEL
    if [[ $rung == awq || $rung == triton ]]; then
      [[ -f weights/awq/quantization.json ]] || continue
      model=weights/awq
    fi
    python -m vellum offline --backend scratch --model "$model" --rung "$rung" --label "scratch-$rung" \
      --experiment engine --batch-sizes $BATCH_SIZES --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
      --out "$OUT" || echo "WARNING: scratch-$rung failed; continuing" >&2
  done
}

# ---- 3. quant: FP16 vs AWQ-INT4 served online, sweeping concurrency ------------------
# Prefix caching off so random prompts can't get accidental cache hits.
quant_stage() {
  start_server fp16 "$MODEL" --no-enable-prefix-caching
  sweep fp16 quant "$MODEL --dtype half --no-enable-prefix-caching" \
    --concurrency $CONCURRENCY --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN"
  stop_server
  start_server awq "$AWQ_MODEL" --no-enable-prefix-caching
  sweep awq quant "$AWQ_MODEL --dtype half --no-enable-prefix-caching" \
    --concurrency $CONCURRENCY --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN"
  stop_server
}

# ---- 4. prefix: same shared-system-prompt workload, cache off vs on ------------------
prefix_stage() {
  local mode flag pin=$((PREFIX_LEN + PREFIX_UNIQUE))
  for mode in off on; do
    flag=$([[ $mode == on ]] && echo --enable-prefix-caching || echo --no-enable-prefix-caching)
    start_server "prefix-$mode" "$MODEL" "$flag"
    sweep "prefix-$mode" prefix "$MODEL --dtype half $flag" \
      --concurrency 1 4 16 64 --input-len "$pin" --output-len "$OUTPUT_LEN" --shared-prefix-len "$PREFIX_LEN"
    stop_server
  done
}

for name in engine scratch quant prefix; do
  if [[ " $STAGES " == *" $name "* ]]; then stage "$name" "${name}_stage"; fi
done

# ---- report ----------------------------------------------------------------------------
python -m vellum plot "$OUT"
python -m vellum report "$OUT" ${GPU_COST:+--gpu-cost-per-hour "$GPU_COST"} > "$OUT/report.md"
{ echo '```'; cat "$OUT/env.txt"; echo '```'; echo; cat "$OUT/report.md"; } > "$OUT/report.tmp" && mv "$OUT/report.tmp" "$OUT/report.md"
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Done. Report: $OUT/report.md   Charts: $OUT/*.png   Bundle: $OUT.tar.gz"
