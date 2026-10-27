#!/bin/bash
# Launch SFT on AMD gfx906 (Vega 20) with the environment and overrides this
# architecture needs. See docs/ROCM_GFX906.md for the diagnosis behind each
# line. Prerequisites: run `python scripts/gpu_preflight.py` first and see READY.
set -euo pipefail
cd "$(dirname "$0")/.."

# Force every GEMM onto plain rocBLAS; hipblasLt ships no gfx906 kernels and
# the first optimizer step hangs the process if torch tries it first.
export TORCH_BLAS_PREFER_HIPBLASLT=0
export DISABLE_ADDMM_HIP_LT=1
# Not supported on this platform; silence the warning by not asking.
unset PYTORCH_HIP_ALLOC_CONF
# Pin to card 0 by default; override by exporting HIP_VISIBLE_DEVICES before this script.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-train/configs/qwen3-0.6b-gfx906.json}"
OUT="${OUT:-results/sft}"
mkdir -p "$OUT"

exec python -u train/sft.py \
  --config "$CONFIG" \
  --out "$OUT" \
  --set sft.args.warmup_steps=8 \
  --set sft.args.eval_strategy='"no"' \
  --set sft.args.logging_steps=5 \
  "$@"
