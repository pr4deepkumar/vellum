#!/usr/bin/env bash
# Run every experiment on a Linux + NVIDIA GPU box and write results/<gpu>/report.md.
#
#   GPU_COST=0.80 bash scripts/run_all.sh            # all stages
#   STAGES="quant prefix" GPU_COST=0.80 bash scripts/run_all.sh
#
# GPU_COST is what YOU pay for this GPU in USD/hr; it is only used for $/1M-token math.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
AWQ_MODEL=${AWQ_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}
STAGES=${STAGES:-"engine quant prefix"}
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
  echo "model: $MODEL"; echo "awq_model: $AWQ_MODEL"
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

# ---- 1. engine: HF Transformers vs vLLM, same fixed batches, no server ---------------
if [[ " $STAGES " == *" engine "* ]]; then
  python -m vellum offline --backend hf   --model "$MODEL" --label hf-fp16   --experiment engine \
    --batch-sizes $BATCH_SIZES --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --out "$OUT"
  python -m vellum offline --backend vllm --model "$MODEL" --label vllm-fp16 --experiment engine \
    --batch-sizes $BATCH_SIZES --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --out "$OUT"
fi

# ---- 2. quant: FP16 vs AWQ-INT4 served online, sweeping concurrency ------------------
# Prefix caching off so random prompts can't get accidental cache hits.
if [[ " $STAGES " == *" quant "* ]]; then
  start_server fp16 "$MODEL" --no-enable-prefix-caching
  sweep fp16 quant "$MODEL --dtype half --no-enable-prefix-caching" \
    --concurrency $CONCURRENCY --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN"
  stop_server
  start_server awq "$AWQ_MODEL" --no-enable-prefix-caching
  sweep awq quant "$AWQ_MODEL --dtype half --no-enable-prefix-caching" \
    --concurrency $CONCURRENCY --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN"
  stop_server
fi

# ---- 3. prefix: same shared-system-prompt workload, cache off vs on ------------------
if [[ " $STAGES " == *" prefix "* ]]; then
  PIN=$((PREFIX_LEN + PREFIX_UNIQUE))
  for mode in off on; do
    flag=$([[ $mode == on ]] && echo --enable-prefix-caching || echo --no-enable-prefix-caching)
    start_server "prefix-$mode" "$MODEL" "$flag"
    sweep "prefix-$mode" prefix "$MODEL --dtype half $flag" \
      --concurrency 1 4 16 64 --input-len "$PIN" --output-len "$OUTPUT_LEN" --shared-prefix-len "$PREFIX_LEN"
    stop_server
  done
fi

# ---- report ----------------------------------------------------------------------------
python -m vellum plot "$OUT"
python -m vellum report "$OUT" ${GPU_COST:+--gpu-cost-per-hour "$GPU_COST"} > "$OUT/report.md"
{ echo '```'; cat "$OUT/env.txt"; echo '```'; echo; cat "$OUT/report.md"; } > "$OUT/report.tmp" && mv "$OUT/report.tmp" "$OUT/report.md"
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Done. Report: $OUT/report.md   Charts: $OUT/*.png   Bundle: $OUT.tar.gz"
