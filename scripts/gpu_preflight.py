#!/usr/bin/env python3
"""Check that this machine can actually run the training stack, before it tries.

The failure this exists for is silent and confusing: on an AMD card whose
architecture PyTorch no longer ships GEMM kernels for, `torch.cuda.is_available()`
returns True, the device reports its correct name and memory, tensors allocate
fine -- and then the first matrix multiply aborts the whole process with a
rocBLAS error about a missing TensileLibrary file. Nothing in that message says
"wrong torch build", so it reads like a bug in the training code.

Run this first:

    python scripts/gpu_preflight.py

Exit status is 0 when training can proceed, 1 when it cannot.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

# Architectures whose rocBLAS kernels were dropped from recent PyTorch ROCm
# wheels. The last wheel that ships gfx906 kernels is torch 2.4.1+rocm6.0.
KNOWN_DROPPED = {"gfx906": "torch==2.4.1+rocm6.0", "gfx900": "torch==2.4.1+rocm6.0"}

# The probe runs in a child process on purpose: a missing-kernel abort cannot be
# caught with try/except, it terminates the interpreter.
PROBE = textwrap.dedent(
    """
    import torch
    a = torch.randn(256, 256, dtype=torch.{dtype}, device="cuda:{index}")
    print(float((a @ a).float().sum()))
    """
)


def probe(index: int, dtype: str) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-c", PROBE.format(index=index, dtype=dtype)],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0, (r.stderr or "").strip()


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL  torch is not installed in this interpreter:", sys.executable)
        return 1

    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("FAIL  no GPU visible to torch. Training needs one; evaluation can run on CPU.")
        return 1

    ok = True
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        arch = getattr(props, "gcnArchName", "") or ""
        base = arch.split(":")[0]
        gb = props.total_memory / 1024**3
        print(f"\ndevice {i}: {props.name} [{arch or 'cuda'}] {gb:.1f} GiB")

        good, err = probe(i, "float32")
        if good:
            print("  float32 matmul  OK")
        else:
            ok = False
            print("  float32 matmul  FAILED -- this device cannot do GPU math with "
                  "this torch build")
            if "TensileLibrary" in err or "rocBLAS" in err:
                fix = KNOWN_DROPPED.get(base)
                print("  cause: the installed torch ships no rocBLAS GEMM kernels for this "
                      f"architecture ({base}).")
                if fix:
                    print("  fix:   pip install --index-url "
                          f"https://download.pytorch.org/whl/rocm6.0 {fix}")
                    print("         (see docs/ROCM_GFX906.md; "
                          "train/requirements-rocm-gfx906.txt pins it)")
                else:
                    print("  fix:   install a torch build whose rocBLAS includes "
                          "this architecture.")
                print("  do NOT set HSA_OVERRIDE_GFX_VERSION to impersonate another "
                      "card: it silences the error and can return wrong numbers.")
            elif err:
                print("  stderr:", err.splitlines()[-1][:200])
            continue

        for dtype in ("float16", "bfloat16"):
            good, _ = probe(i, dtype)
            note = ""
            if dtype == "bfloat16" and base in ("gfx906", "gfx900") and good:
                note = "  (emulated on this card -- prefer float16; see docs/ROCM_GFX906.md)"
            print(f"  {dtype} matmul {'OK' if good else 'FAILED'}{note}")

    alloc = os.environ.get("PYTORCH_HIP_ALLOC_CONF", "")
    if "expandable_segments" not in alloc:
        print("\nhint: export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True "
              "reduces fragmentation on long GRPO runs.")

    print("\n" + ("READY" if ok else "NOT READY"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
