#!/bin/bash
# Startup script for the VM that scripts/gcp_run.sh creates; runs as root at boot.
# Pulls the uploaded checkout from GCS, installs vLLM + VELLUM (with its Mistral engine), runs
# scripts/run_all.sh, uploads results and logs, then deletes the VM it runs on.
set -uo pipefail
export HOME=/root

md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME=$(md name)
ZONE=$(md zone)
ZONE=${ZONE##*/}
DEST="gs://$(md attributes/vellum-bucket)/runs/$(md attributes/vellum-run)"
LOG=/var/log/vellum-run.log

if [[ -e /var/lib/vellum-ran ]]; then  # startup scripts run on every boot; the benchmark runs once
  echo "vellum already ran on this VM; powering off"
  poweroff
  exit 0
fi
touch /var/lib/vellum-ran

(
  set -e
  mkdir -p /opt/vellum && cd /opt/vellum
  gcloud storage cp "$DEST/run.env" .
  gcloud storage cp "$DEST/src.tar.gz" - | tar xz
  source ./run.env
  echo "== waiting for the NVIDIA driver"
  for _ in $(seq 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done
  nvidia-smi
  if ! command -v g++ > /dev/null; then
    # vLLM's FlashInfer sampler compiles CUDA kernels on first use, which needs a C++ compiler;
    # the Deep Learning VM image ships gcc without g++. Retry while first-boot apt holds the lock.
    echo "== installing g++"
    for _ in $(seq 30); do apt-get update -qq && break; sleep 10; done
    DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq g++
  fi
  echo "== installing uv, vLLM $PIN_VLLM and VELLUM"
  curl -LsSf https://astral.sh/uv/0.11.15/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  uv venv /opt/venv --python 3.12
  uv pip install --python /opt/venv/bin/python "vllm==$PIN_VLLM"
  uv pip install --python /opt/venv/bin/python -e ".[mistral]"
  export PATH="/opt/venv/bin:$PATH"
  echo "== running scripts/run_all.sh (STAGES=$STAGES)"
  bash scripts/run_all.sh
) 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

outcome=$([[ $status == 0 ]] && echo DONE || echo FAILED)
echo "== $outcome (exit $status); uploading to $DEST"
[[ -d /opt/vellum/results ]] && gcloud storage cp -r /opt/vellum/results/* "$DEST/results/"
[[ -d /opt/vellum/logs ]] && gcloud storage cp -r /opt/vellum/logs/* "$DEST/logs/"
gcloud storage cp "$LOG" "$DEST/run.log"
echo "$outcome" | gcloud storage cp - "$DEST/$outcome"  # last: gcp_run.sh fetches once this exists
echo "== deleting this VM"
gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || poweroff
