# Why attention experiment 012 changed output hashes

The proven difference is floating-point contraction. In the **archived benchmark binaries**, control 006 rounds the multiplication and addition separately during attention V accumulation. Experiment 012 explicitly uses fused multiply-add, which rounds once. This affects every attention position, not just the first pair.

Both archived manifests record GCC 13.3.0 and these shared flags:

```text
-D_GNU_SOURCE -O3 -std=c11 -Wall -Wextra -march=native
```

Control adds `-DPF_NO_X86_MOE`; experiment 012 adds `-DPF_X86_ATTN`.

Evidence from `objdump -d -M intel --disassemble=ph_b`:

| Archived run | Instruction evidence | Arithmetic |
|---|---|---|
| `20260912T024125.686657Z-006-control-repeat` | `5cd0: vmulps ymm0,ymm1,[rcx-0x200]`; `5cd8: vaddps ymm0,ymm0,[rbx]` | Separate product and sum rounding |
| `20260912T024432.210211Z-012-attention-registers` | `5cb7: vfmadd231ps ymm9,ymm0,[rax-0x200]` | Single fused rounding |
| Session 3 baseline 032 | `6e60: vmulps ymm0,ymm1,[rcx-0x200]`; `6e68: vaddps ymm0,ymm0,[r13]` | Separate product and sum rounding |

The same pattern repeats across all eight output vectors. Both paths use scalar division by the same softmax denominator and increasing position order. Sources and full build commands are preserved in each run's `source/pf.c` and `manifest.json`. The existing correctness archive records five changed complete-phase/layer hashes for run 012 despite its isolated attention tests passing.

The previously saved `/tmp/pf-attention-base.asm` and `/tmp/pf-attention-candidate.asm` are from **production executables**, not those archived benchmark binaries. Their V accumulation uses FMA in both cases. Thus their apparent equivalence did not explain the benchmark mismatch. Build flags for these temporary production executables were not recorded in the assembly files. A compiler option query reports `fp-contract=fast` both with and without `-std=c11`; a dialect-default explanation alone is unsupported. The precise compiler decision responsible for contraction differing between translation units remains unknown.

A source-only diagnostic preserved in [session3-attention.patch](session3-attention.patch) against the session-3 baseline uses a register-resident multiply/add loop with function-local contraction disabled, gated by `PF_X86_ATTN_ROUNDED`. It was not compiled, timed, or integrated. Matching the benchmark's separately rounded arithmetic would not establish equivalence to the production executable's fused arithmetic.

Decision: retain the existing attention implementation. Reconsider only with a production-built output comparison and an explicit arithmetic policy; a numerical-policy change also needs model-level validation, which is still unavailable. No production source or benchmark workload was changed for this investigation.
