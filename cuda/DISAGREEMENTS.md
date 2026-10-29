# Per-document disagreements of the final build vs the CPU oracle

**Final build:** `final/engine` = `stack_fdm@17370154d25f`. Its output is identical to `host_moe_attn_f16`.

**Comparison:** check.txt (1,534 docs) against the CPU bf16-emu oracle (`experiments/E1/oracle_bf16emu`).

**Source data:**
- `experiments/E23/gpu_run5/check/stack_fdm.vs_bf16emu.json`, produced by `bench/compare.py`;
- the noise-floor reports in `experiments/E1/reports/`.

Per-doc data is in `final/disagreements.json`.

## Summary

| quantity | value |
|---|---|
| docs with span or masked-text disagreement | **27 / 1,534** (98.24% agreement) |
| mean / max \|Δlogprob\| | 0.210 / 16.39. The max sits in very negative tail classes, as for every valid build |
| argmax flips (tokens) | 112 of 353,505 |
| **largest oracle top-1/top-2 margin of any flip** | **2.12 nats** |
| largest flip margin between two *valid CPU builds* (bf16-emu vs fp32, `E1/reports/A`) | 3.23 nats |

**Classification of the 27 disagreeing docs:**

| class | docs | meaning |
|---|---|---|
| floor-explained | 23 | The same doc also disagrees in a measured noise-floor comparison (any of the three below). |
| Viterbi-only near-tie | 3 (813, 959, 1297) | No argmax flip anywhere in the doc. Only the constrained Viterbi path differs, from tiny log-prob differences. |
| near-tie by margin | 1 (851) | One flip with a top-2 margin of 0.089 nats. |
| **unexplained** | **0** | |

The three noise-floor comparisons:
- **A:** CPU bf16-emu vs CPU fp32 (numerics choice), 37 docs.
- **E:** CPU bf16-emu scalar vs AVX2 (accumulation order only), 16 docs.
- **D:** GPU `--batch` 32k vs default, 6 docs.

**Where the disagreements fall.** 8 of the 27 docs are long multi-window documents (1500–1519, 8.7k–68k chars). All 8 are in floor set A, and 7 are also in E or D. The rest are short docs.

**Every flip is a near-tie.** No flip has a margin beyond what two correct CPU builds of pf already produce between themselves.

## Context
- The initial CUDA port disagreed on 16 docs (98.96%); 10 of those are among these 27.
- The upstream reference GPU implementation (openai/privacy-filter, PyTorch+Triton) agrees with CPU pf on only **96.35%** of these docs, and is not even deterministic run to run (`experiments/E2/README.md`).
- A deliberately wrong build (one layer skipped) scores 45.31% (`experiments/E1`).

## Per-doc table

| doc | kind | flips | max margin | in A | in E | in D | class |
|---|---|---|---|---|---|---|---|
| 139 | short | 1 | 0.355 | ✓ | | | floor |
| 515 | short | 1 | 0.675 | ✓ | | | floor |
| 638 | short | 1 | 0.801 | ✓ | | | floor |
| 731 | short | 1 | 0.195 | ✓ | | | floor |
| 813 | short | 0 | — | | | | Viterbi-only |
| 823 | short | 0 | — | ✓ | | | floor |
| 851 | short | 1 | 0.089 | | | | near-tie |
| 856 | short | 1 | 0.095 | ✓ | ✓ | | floor |
| 959 | short | 0 | — | | | | Viterbi-only |
| 1020 | short | 0 | — | ✓ | | | floor |
| 1026 | short | 0 | — | | ✓ | | floor |
| 1111 | short | 0 | — | | ✓ | | floor |
| 1125 | short | 2 | 0.174 | ✓ | | | floor |
| 1168 | short | 3 | 0.295 | ✓ | ✓ | | floor |
| 1263 | short | 0 | — | ✓ | ✓ | | floor |
| 1297 | short | 0 | — | | | | Viterbi-only |
| 1333 | short | 0 | — | ✓ | | | floor |
| 1367 | short | 4 | 1.318 | ✓ | | | floor |
| 1445 | short | 0 | — | ✓ | ✓ | | floor |
| 1500 | long | 8 | 0.650 | ✓ | ✓ | ✓ | floor |
| 1503 | long | 0 | — | ✓ | | | floor |
| 1505 | long | 8 | 0.786 | ✓ | ✓ | | floor |
| 1509 | long | 11 | 1.490 | ✓ | ✓ | ✓ | floor |
| 1510 | long | 4 | 1.275 | ✓ | | | floor |
| 1515 | long | 7 | 0.166 | ✓ | | | floor |
| 1516 | long | 4 | 0.191 | ✓ | ✓ | | floor |
| 1519 | long | 4 | 0.574 | ✓ | ✓ | ✓ | floor |
