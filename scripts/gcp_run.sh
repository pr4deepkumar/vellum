#!/usr/bin/env bash
# Run VELLUM v2 on a Google Cloud L4 VM, billed to your project (billing credits apply).
#
#   bash scripts/gcp_run.sh preflight   # read-only: login, project, APIs, L4 quota
#   bash scripts/gcp_run.sh             # up + wait: start the VM, wait, download results
#   bash scripts/gcp_run.sh up          # upload this checkout and start the VM (runs unattended)
#   bash scripts/gcp_run.sh status      # VM state and the last lines of its log
#   bash scripts/gcp_run.sh wait        # poll until the run finishes, then fetch
#   bash scripts/gcp_run.sh fetch       # download results/ and logs, delete the VM if still up
#   bash scripts/gcp_run.sh down        # delete the VM now
#
# The VM runs scripts/run_all.sh on this working tree as it is, uncommitted changes included,
# uploads results to gs://$BUCKET, then deletes itself. GCP also deletes it after MAX_HOURS
# whatever happens, so a hung run costs at most MAX_HOURS of the machine.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v gcloud > /dev/null || {
  echo "Install the Google Cloud CLI, then run 'gcloud init': https://cloud.google.com/sdk/docs/install" >&2
  exit 1
}
PROJECT=${PROJECT:-$(gcloud config get-value project 2> /dev/null)}
[[ -n "$PROJECT" ]] || { echo "No project set: run 'gcloud init' or pass PROJECT=<project-id>" >&2; exit 1; }
REGION=${REGION:-us-central1}  # home region: the results bucket and preflight's quota check
# L4s are often out of stock in a zone. `up` tries every L4 zone of these regions in order, and
# when all of them are taken it tries again every few minutes for RETRY_MINUTES. Set ZONES to
# pin an exact list instead.
REGIONS=${REGIONS:-"$REGION us-east1 us-east4 us-west1 us-west4 northamerica-northeast1 northamerica-northeast2"}
ZONES=${ZONES:-}
RETRY_MINUTES=${RETRY_MINUTES:-60}
RETRY_PAUSE=${RETRY_PAUSE:-180}  # seconds between passes over the zones
VM=${VM:-vellum-l4}
MACHINE=${MACHINE:-g2-standard-8}  # 1x L4 + 32 GB RAM; g2-standard-4's 16 GB is too little for the HF baseline
SPOT=${SPOT:-0}                    # 1: about 60% cheaper, but a preemption loses the run
MAX_HOURS=${MAX_HOURS:-5}          # the full run takes roughly 2-2.5 h
DISK_GB=${DISK_GB:-200}
IMAGE_FAMILY=${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}  # NVIDIA driver 580 for vLLM's CUDA 13 wheels
BUCKET=${BUCKET:-$PROJECT-vellum}
STATE=logs/gcp_run.env

# What the VM runs, handed to scripts/run_all.sh: every stage on Mistral 7B -- HF and vLLM,
# the from-scratch engine at every rung, and vLLM's AWQ and prefix caching under load.
MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
AWQ_MODEL=${AWQ_MODEL:-solidrust/Mistral-7B-Instruct-v0.3-AWQ}
STAGES=${STAGES:-"engine scratch quant prefix"}
SCRATCH_RUNGS=${SCRATCH_RUNGS:-"naive lastlogit nosync gqa sdpa chunked graph awq triton"}
# USD/hr for the $/1M-token math: g2-standard-8 list price in us-central1 (spot rates float monthly).
GPU_COST=${GPU_COST:-$([[ $SPOT == 1 ]] && echo 0.34 || echo 0.85)}
PIN_VLLM=${PIN_VLLM:-0.31.0}  # pinned for reproducible results (not VLLM_*: vLLM warns on unknown ones)
AWQ_QUALITY=${AWQ_QUALITY:-1}    # see scripts/run_all.sh
AWQ_BASELINE=${AWQ_BASELINE:-1}
OUT=${OUT:-}  # results folder on the VM; empty means results/<gpu name>

vm_zone() {
  gcloud compute instances list --project "$PROJECT" --filter="name=($VM)" --format="value(zone.basename())" 2> /dev/null
}

l4_zones() {  # every zone in $REGIONS that offers L4s, region by region
  local all region
  all=$(gcloud compute accelerator-types list --project "$PROJECT" --filter="name=nvidia-l4" \
    --format="value(zone.basename())" | sort)
  for region in $REGIONS; do grep "^$region-" <<< "$all" || true; done
}

load_state() {  # the run `up` started last: RUN, ZONE, BUCKET
  RUN=""
  # shellcheck source=/dev/null  # written by `up`
  if [[ -f $STATE ]]; then source "$STATE"; fi
  [[ -n $RUN ]] || { echo "No run recorded in $STATE; start one with 'up'" >&2; exit 1; }
}

marker() {  # prints DONE or FAILED once the VM has written one
  local m
  for m in DONE FAILED; do
    if gcloud storage ls "gs://$BUCKET/runs/$RUN/$m" > /dev/null 2>&1; then echo "$m"; return 0; fi
  done
  return 1
}

serial_tail() {  # last N lines the startup script printed to the VM's serial console
  gcloud compute instances get-serial-port-output "$VM" --zone "$1" --project "$PROJECT" 2> /dev/null \
    | grep -o 'startup-script: .*' | sed 's/^startup-script: //' | tail -n "$2" || true
}

quota() {  # quota <metric>: this project's limit, regional or (GPUS_ALL_REGIONS) global
  local where=(regions describe "$REGION")
  [[ $1 == GPUS_ALL_REGIONS ]] && where=(project-info describe)
  gcloud compute "${where[@]}" --project "$PROJECT" --flatten="quotas[]" \
    --format="csv[no-heading](quotas.metric,quotas.limit)" | awk -F, -v m="$1" '$1 == m { print $2 }'
}

preflight() {
  local account
  account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2> /dev/null | head -1)
  [[ -n "$account" ]] || { echo "Not logged in: run 'gcloud auth login'" >&2; return 1; }
  echo "project $PROJECT, region $REGION, $MACHINE$([[ $SPOT == 1 ]] && echo ' (spot)'), \$$GPU_COST/hr"
  local api missing=()
  for api in compute.googleapis.com storage.googleapis.com; do
    gcloud services list --project "$PROJECT" --enabled --filter="config.name=$api" \
      --format="value(config.name)" | grep -q . || missing+=("$api")
  done
  if ((${#missing[@]})); then
    echo "Not enabled yet: ${missing[*]} ('up' enables them; quota is checked after that)"
    return 0
  fi
  local metric=NVIDIA_L4_GPUS regional global
  [[ $SPOT == 1 ]] && metric=PREEMPTIBLE_NVIDIA_L4_GPUS
  regional=$(quota "$metric")
  global=$(quota GPUS_ALL_REGIONS)
  regional=${regional%.*} global=${global%.*}
  echo "quota: $metric in $REGION = ${regional:-0}, GPUS_ALL_REGIONS = ${global:-0}"
  local need=()
  ((${regional:-0} >= 1)) || need+=("\"$metric\" in $REGION")
  ((${global:-0} >= 1)) || need+=("\"GPUs (all regions)\" (GPUS_ALL_REGIONS)")
  if ((${#need[@]})); then
    {
      echo "Not enough GPU quota. Request a limit of 1 for each of these, then re-run preflight:"
      printf '  - %s\n' "${need[@]}"
      echo "at https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT"
      echo "(Projects on the \$300 free trial get no GPUs until upgraded to a paid account.)"
    } >&2
    return 1
  fi
  echo "preflight ok"
}

up() {
  [[ -z "$(vm_zone)" ]] || { echo "$VM already exists: use 'status', 'wait' or 'down'" >&2; exit 1; }
  gcloud services enable compute.googleapis.com storage.googleapis.com --project "$PROJECT"
  preflight
  if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" > /dev/null 2>&1; then
    gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
      --uniform-bucket-level-access
  fi
  # The VM runs as the default compute service account: let it read and write this bucket only.
  local number
  number=$(gcloud projects describe "$PROJECT" --format="value(projectNumber)")
  gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --role roles/storage.objectAdmin \
    --member "serviceAccount:$number-compute@developer.gserviceaccount.com" > /dev/null

  local run tmp f v
  run=$(date -u +%Y%m%d-%H%M%S)
  tmp=$(mktemp -d)
  # This checkout as git sees it (tracked + untracked, minus ignored files), without old results
  # or the v1/ archive.
  git ls-files --cached --others --exclude-standard | grep -v -e '^results/' -e '^v1/' \
    | while read -r f; do [[ -e $f ]] && echo "$f"; done > "$tmp/files"
  COPYFILE_DISABLE=1 tar czf "$tmp/src.tar.gz" -T "$tmp/files"  # no macOS ._ files
  for v in MODEL AWQ_MODEL STAGES SCRATCH_RUNGS GPU_COST PIN_VLLM AWQ_QUALITY AWQ_BASELINE OUT; do
    printf 'export %s=%q\n' "$v" "${!v}"
  done > "$tmp/run.env"
  gcloud storage cp "$tmp/src.tar.gz" "$tmp/run.env" "gs://$BUCKET/runs/$run/"
  rm -rf "$tmp"

  local args=(--project "$PROJECT" --machine-type "$MACHINE"
    --image-family "$IMAGE_FAMILY" --image-project deeplearning-platform-release
    --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced
    --maintenance-policy TERMINATE --scopes cloud-platform
    --max-run-duration "${MAX_HOURS}h" --instance-termination-action DELETE
    --metadata "install-nvidia-driver=True,vellum-bucket=$BUCKET,vellum-run=$run"
    --metadata-from-file startup-script=scripts/gcp_vm_startup.sh)
  [[ $SPOT == 1 ]] && args+=(--provisioning-model SPOT)
  local zone out zones=$ZONES deadline=$((SECONDS + RETRY_MINUTES * 60))
  [[ -n $zones ]] || zones=$(l4_zones | tr '\n' ' ')
  while :; do
    for zone in $zones; do
      if out=$(gcloud compute instances create "$VM" --zone "$zone" "${args[@]}" 2>&1); then
        mkdir -p logs
        printf 'RUN=%s\nZONE=%s\nBUCKET=%s\n' "$run" "$zone" "$BUCKET" > "$STATE"
        echo "Run $run started in $zone. The VM works unattended; 'status' to peek, 'wait' to collect."
        return 0
      fi
      if [[ $out != *ZONE_RESOURCE_POOL_EXHAUSTED* ]]; then  # not a stockout: retrying won't help
        echo "$out" >&2
        return 1
      fi
      echo "   $zone: no $MACHINE capacity"
    done
    if ((SECONDS >= deadline)); then
      echo "No L4 capacity in any of $zones for $RETRY_MINUTES min. Try later, or widen REGIONS." >&2
      return 1
    fi
    echo "== every zone is out of L4s; trying again in $((RETRY_PAUSE / 60)) min $((RETRY_PAUSE % 60)) s"
    sleep "$RETRY_PAUSE"
  done
}

status() {
  load_state
  local zone outcome
  zone=$(vm_zone)
  if [[ -n $zone ]]; then
    echo "$VM: $(gcloud compute instances describe "$VM" --zone "$zone" --project "$PROJECT" \
      --format='value(status)') in $zone, run $RUN"
    serial_tail "$zone" 8
  else
    echo "$VM: not running (run $RUN)"
  fi
  if outcome=$(marker); then echo "run $RUN finished: $outcome ('fetch' to download)"; fi
}

wait_run() {
  load_state
  echo "Waiting for run $RUN. Ctrl-C is safe: the VM keeps going; 'wait' or 'fetch' later."
  local zone line last=""
  until marker > /dev/null; do
    zone=$(vm_zone)
    if [[ -z $zone ]]; then
      marker > /dev/null && break  # it wrote the marker just before deleting itself
      echo "$VM is gone without finishing (spot preemption or the ${MAX_HOURS}h limit)" >&2
      break
    fi
    line=$(serial_tail "$zone" 1)
    if [[ -n $line && $line != "$last" ]]; then echo "  $(date +%H:%M)  $line"; last=$line; fi
    sleep 60
  done
  fetch
}

fetch() {
  load_state
  local outcome
  outcome=$(marker || echo UNFINISHED)
  mkdir -p results "logs/gcp-$RUN"
  gcloud storage cp -r "gs://$BUCKET/runs/$RUN/results/*" results/ \
    || echo "No results were uploaded for run $RUN"
  gcloud storage cp "gs://$BUCKET/runs/$RUN/run.log" "logs/gcp-$RUN/" 2> /dev/null || true
  gcloud storage cp -r "gs://$BUCKET/runs/$RUN/logs/*" "logs/gcp-$RUN/" 2> /dev/null || true
  down
  echo "Run $RUN: $outcome. Results in results/ (NVIDIA-L4/), log in logs/gcp-$RUN/run.log"
}

down() {
  local zone
  zone=$(vm_zone)
  if [[ -z $zone ]]; then echo "$VM: nothing to delete"; return 0; fi
  gcloud compute instances delete "$VM" --zone "$zone" --project "$PROJECT" --quiet \
    || echo "$VM is already being deleted"
}

case "${1:-all}" in
  preflight) preflight ;;
  up) up ;;
  status) status ;;
  wait) wait_run ;;
  fetch) fetch ;;
  down) down ;;
  all)
    up
    wait_run
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
