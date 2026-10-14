# Running this repo on AMD gfx906 (Vega 20)

Applies to Radeon VII, Radeon Pro Vega II, Instinct MI50 and MI60 — anything
whose `gcnArchName` starts with `gfx906`. Everything below was measured on a
2× Radeon Pro Vega II host under ROCm 7.2.3 on 2026-10-14. Nothing here is
estimated.

**Short version:** install `torch==2.4.1+rocm6.0`, train in `float16` (not
`bfloat16`), and turn TunableOp on for runs longer than a few minutes. Use
`train/configs/qwen3-0.6b-gfx906.json`, which already does the second part.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r train/requirements-rocm-gfx906.txt
python scripts/gpu_preflight.py        # must print READY
```

---

## 1. The problem: current PyTorch has no GEMM kernels for this card

A current ROCm wheel looks like it works. `torch.cuda.is_available()` returns
`True`, the device reports its real name and its 32 GiB, and tensors allocate
without complaint. Then the first matrix multiply kills the process:

```
rocBLAS error: Cannot read .../rocblas/library/TensileLibrary.dat
               for GPU arch : gfx906
List of available TensileLibrary Files :
  TensileLibrary_lazy_gfx908.dat
  TensileLibrary_lazy_gfx90a.dat
  TensileLibrary_lazy_gfx942.dat
  TensileLibrary_lazy_gfx1030.dat ... gfx1100, gfx1101, gfx1102, gfx1200, gfx1201
```

gfx906 is not in that list. AMD dropped Vega 20 from the rocBLAS Tensile builds
after ROCm 5.7, so the kernels that perform matrix multiplication are simply not
shipped any more. This is not a `bfloat16` problem — **every** dtype fails,
`float32` included.

Two things make this hard to diagnose:

- It is not an exception. The process aborts, so `try`/`except` cannot catch it
  and the traceback points at whatever line happened to do the first matmul.
- Installing the system ROCm does not help. On this host `/opt/rocm-7.2.3`
  carries the same architecture list and is missing gfx906 too. `llama.cpp`
  built with `-DGGML_HIP=ON` runs on this card perfectly well, which makes the
  card look fine — it uses hand-written HIP kernels and never calls rocBLAS
  Tensile.

`scripts/gpu_preflight.py` detects exactly this and prints the fix.

## 2. The fix: the last wheel that ships gfx906 kernels

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.0 torch==2.4.1+rocm6.0
```

`torch-2.4.1+rocm6.0` contains `TensileLibrary_lazy_gfx906.dat`. It is the
newest PyTorch release that does, and it satisfies this repo's own `torch>=2.4`
floor. Verified on this host: `float32`, `float16` and `bfloat16` matmuls all
return correct results on both cards.

That pin forces the rest of the stack down with it, because current
`transformers` wants `torch>=2.5`. The combination that works is pinned in
`train/requirements-rocm-gfx906.txt`: `transformers==4.56.2` and `trl==1.15.0`,
which are this repo's declared floors.

**Do not** set `HSA_OVERRIDE_GFX_VERSION` to make the card claim to be a gfx908
or gfx90a. It will stop the error and let kernels compiled for a different
instruction set run on silicon that does not implement them. Wrong numbers are
worse than a crash.

## 3. Tuning: float16, and TunableOp

gfx906 has **no bfloat16 hardware**. PyTorch emulates it, and the cost is
severe. Measured with a 4096×4096 matmul, 30 iterations after warm-up:

| dtype | default | `PYTORCH_TUNABLEOP_ENABLED=1` |
|---|---:|---:|
| float32 | 9.60 TFLOP/s | 9.90 TFLOP/s |
| **float16** | **10.33 TFLOP/s** | **17.84 TFLOP/s** |
| bfloat16 | 5.93 TFLOP/s | 6.18 TFLOP/s |

Two things follow.

**Train in float16.** bfloat16 is 43% slower than float16 here, and slower than
float32. Every stock config in `train/configs/` sets `bf16: true` because that
is the right default on NVIDIA and on newer CDNA parts; on this card it is the
worst of the three. `train/configs/qwen3-0.6b-gfx906.json` is the 24 GB config
with `model.dtype: "float16"` and `fp16: true` / `bf16: false` in both stages,
and nothing else changed. The trade is the usual one: float16 has a narrower
exponent range than bfloat16, so loss scaling matters — the HF `Trainer` handles
it when `fp16: true` is set, but watch the loss for NaNs on the first few steps.

**Turn TunableOp on for long runs.** It autotunes each GEMM shape against the
available rocBLAS algorithms and gives float16 a further 1.7×, for **3.0× over
the bfloat16 default**:

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

Honest caveats, because this one has a real cost:

- The tuning sweep is slow. The benchmark above took several minutes of wall
  clock before producing a number, all of it warm-up. It pays for itself across
  158 SFT steps; it does not pay for itself on a smoke test.
- Setting `PYTORCH_TUNABLEOP_FILENAME` alone did **not** leave a results file on
  disk in our run, so the warm-up appears to be paid again on every new process.
  `torch.cuda.tunable.write_file()` is the documented way to persist it; we did
  not verify that it works on this version, so budget for the warm-up unless you
  check it yourself.
- `expandable_segments` is a fragmentation fix, not a speed fix. It is here
  because GRPO alternates generation and training allocations for hundreds of
  steps, which is the pattern that fragments a caching allocator.

## 4. What is still not solved

- **vLLM does not support gfx906.** `grpo.args.use_vllm` must stay `false`, so
  GRPO generation runs through plain `transformers`, which is much slower. This
  is the binding constraint on whether a 300-step GRPO run is practical here,
  not VRAM.
- **No Flash Attention.** `model.attn_implementation` stays `"sdpa"`.
- **32 GiB per card, and no NVLink.** Two cards do not combine into 64 GiB.
