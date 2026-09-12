# Cloud hillclimb log

Upstream: `ab3332870f689c3dc3451212431ecafd9238b1f9`.
Target: ChatGPT Work Linux x86-64, Intel Xeon Platinum 8272CL, GCC 13.3,
9 visible CPUs, cgroup budget 8 CPUs / 20 GiB. No AVX-512 BF16 instruction.

All experiments use deterministic synthetic weights unless explicitly marked
otherwise. Kernel speedups are not model end-to-end speedups or evidence of
unchanged PII labels. No approximation or expert skipping is enabled.

## 000 — setup / portability

- Upstream `make` fails on Linux: feature declarations and math constants
  hidden by strict C11; inappropriate x86 `-mcpu=native`; pthread flag absent.
  Full compiler output is under `setup/upstream-make.*`.
- Fixed architecture-specific flags, feature declarations, pthread linking,
  and monotonic timing. Successful build output: `setup/linux-make.*`.
- Added validated `PF_THREADS=1..64`, affinity and cgroup-v2-root quota aware
  automatic pool size; single-thread/single-task dispatch avoids pool barriers.
  Thread creation failures now fail explicitly instead of hanging.
- Direct GitHub clone could not connect through the network proxy. Sources
  were fetched through the authorized GitHub connector at the upstream SHA;
  the local initial commit is a source snapshot. Published commits are based
  on the real upstream commit and preserve its other files, including pf_mb.
- Hugging Face checkpoint request failed with `curl: (28) Proxy CONNECT
  aborted due to timeout`. Full-model/reference accuracy and end-to-end
  throughput are blocked until the checkpoint/tokenizer are available.

Each measured run will retain source snapshots, compiler and command details,
host facts, raw repetitions, resource counters, correctness metrics, and raw
stdout/stderr. Results are appended, including rejected experiments.

## Measured iterations

Timings below are medians in milliseconds. Serial and parallel rows are different
configurations; use matching thread counts for algorithm speedups. Each link
contains all repetitions and evidence, including rejected attempts.

| Run | Decision | Workers | 16-item expert | Packed256 layer | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| [001-upstream-native](results/20260912T023842.420521Z-001-upstream-native) | baseline | 1 | 1.964 | — | Upstream x86 algorithm. Original GNU feature-macro build repair only. |
| [002-x86-blocked-kt32](results/20260912T023916.578198Z-002-x86-blocked-kt32) | accepted | 1 | 0.686 | — | Blocked AVX512 expert kernel materially improves grouped work; continue validation. |
| [003-x86-blocked-kt16](results/20260912T023940.378601Z-003-x86-blocked-kt16) | rejected | 1 | 0.721 | — | KT16 local gains did not translate into a repeatable complete-layer improvement (see008). |
| [004-x86-blocked-kt64](results/20260912T024008.569919Z-004-x86-blocked-kt64) | rejected | 1 | 0.662 | — | KT64 loses streaming-weight performance despite a small cached-kernel gain. |
| [005-x86-avx2-kt32](results/20260912T024036.709751Z-005-x86-avx2-kt32) | rejected | 1 | 0.925 | — | AVX2 blocked path slower than AVX512 on this CPU; keep AVX2 as portable fallback. |
| [006-control-repeat](results/20260912T024125.686657Z-006-control-repeat) | control | 1 | 2.263 | 223.954 | Repeated original algorithm under final harness, serial dispatch. |
| [007-kt32-extended](results/20260912T024139.307729Z-007-kt32-extended) | accepted | 1 | 0.652 | 130.790 | KT32 complete-layer confirmation; retained tile size. |
| [008-kt16-extended](results/20260912T024209.770032Z-008-kt16-extended) | rejected | 1 | 0.697 | 132.303 | KT16 complete-layer timing slightly worse than KT32; no robust gain. |
| [009-threads8-kt16](results/20260912T024312.652753Z-009-threads8-kt16) | measurement | 8 | 0.731 | 20.832 | Eight-worker KT16 experiment establishes thread scaling. |
| [010-threads4-kt16](results/20260912T024356.861766Z-010-threads4-kt16) | rejected | 4 | 0.690 | 46.627 | Four workers reduce throughput relative to eight. |
| [011-threads2-kt16](results/20260912T024412.425191Z-011-threads2-kt16) | rejected | 2 | 0.810 | 76.004 | Two workers reduce throughput relative to eight. |
| [012-attention-registers](results/20260912T024432.210211Z-012-attention-registers) | rejected | 1 | 0.695 | 147.918 | Attention-only isolated bitwise tests passed but complete-phase output hashes changed; overall timing improvement inconsistent. Removed from production. |
| [013-control-threads8](results/20260912T024504.053951Z-013-control-threads8) | control | 8 | 1.755 | 41.771 | Original expert algorithm with eight workers, repeated9 samples. |
| [014-final-threads8](results/20260912T024555.543247Z-014-final-threads8) | accepted | 8 | 0.660 | 26.077 | Final KT32 blocked kernel with eight workers; identical complete-phase hashes to control. |
| [015-threads9](results/20260912T024710.193519Z-015-threads9) | rejected | 9 | 0.864 | 28.422 | Nine workers did not improve full-layer timing over eight and exceed quota. |
| [016-final-repeat](results/20260912T024735.940018Z-016-final-repeat) | accepted | 8 | 0.666 | 26.575 | Independent repeat of final configuration. |
| [017-control-repeat-threads8](results/20260912T024838.865216Z-017-control-repeat-threads8) | control | 8 | 2.083 | 35.876 | Independent repeated eight-worker control after final candidate. |
| [018-sparse-fast-path](results/20260912T025038.223913Z-018-sparse-fast-path) | rejected | 8 | 0.670 | 27.458 | Direct singleton dispatch adds complexity without consistent measured gain; short-layer timing regressed in this trial. Restored exact source from014/016. |

## Session 1 comparison

Both configurations use eight workers and the final harness. Pooled medians
use 18 samples across two independent runs for each variant: controls013/017,
final014/016. These are synthetic kernel/layer measurements, not whole-model
speedups. The final source SHA256 is `c4466e70fc32b2814d360d9151d119cbfef4c9c0901c0ce375624cf69152dba8`.

| Workload | Control ms | Final ms | Speedup |
| --- | ---: | ---: | ---: |
| One cached single-token expert | 0.120 | 0.140 | 0.86× |
| One expert with16assignments | 1.990 | 0.662 | 3.00× |
| 32experts ×4assignments | 5.911 | 3.737 | 1.58× |
| 128experts ×1assignment | 7.885 | 7.883 | 1.00× |
| 32-token synthetic layer | 9.446 | 7.536 | 1.25× |
| 256-token packed synthetic layer | 41.049 | 26.135 | 1.57× |

The small cached singleton case is slower in the final samples; the optimization
primarily benefits grouped expert work. Streaming singleton work is comparable.
The explicit singleton rewrite in018 did not provide a consistent overall win,
so it was rejected. Cloud scheduling noise remains visible in raw sample ranges;
these measurements are specific to this VM and synthetic routing distribution.

## Correctness and limitations

- All18runs passed independent numerical reference checks. Final014/016 have
  identical output hashes to control006 for all13cases, including complete
  synthetic layers, with no failed reference values. Exact standalone x86
  kernel comparisons and pool publication checks are retained under
  `correctness/`, `setup/`, and `validation/`.
- Attention012 is explicitly rejected despite passing the error tolerance:
  complete-phase output hashes changed. Its source remains in its run snapshot.
- Early manifests with `child_maxrss_kib` contain the compiler-inclusive child
  high-water value; use C samples' `max_rss_kib` for the measured process.
  Early numeric seed labels and projection-output weight footprint metadata were
  corrected during harness setup. RNG/data and default timed paths were unchanged;
  frozen harnesses retain the original evidence. All final comparisons use the
  same corrected harness.
- Model/tokenizer download was blocked by the network proxy. No checkpoint PII
  mask equivalence, eight-layer throughput, cold start, HTTP latency, or real
  multilingual corpus accuracy claim is made. `model_bench.py` is ready for an
  explicitly supplied test corpus and real checkpoint.
- Automatic review initially rejected a wider upload containing an optional
  generated text corpus with contact/credential examples. The corpus was removed
  and the upload narrowed to source and numerical synthetic-kernel evidence;
  the narrowed upload was accepted. No model input documents are archived.

## Continue the hillclimb

Run `make bench-cloud` for the current implementation and `make bench-report`
to refresh the offline chart. For a control, use:

```sh
PF_THREADS=8 python3 bench/run.py --label control --parallel --profile extended \
  --cflags '-D_GNU_SOURCE -O3 -std=c11 -march=native -DPF_NO_X86_MOE -DPF_NO_PROJ4 -DPF_NO_WIDE_AXPY'
```

Change one factor, record a new uniquely labeled run, compare outputs and timing,
then append a `decision.json` next to the immutable evidence. Preserve rejected
runs. Commit and push all new numerical evidence after each iteration. When
checkpoint access is available, complete model-level regression before treating
these synthetic results as deployment evidence. Next meaningful experiments
are trained-router distributions, longer packed inputs and model-resident
request latency; no background benchmark process remains running.

## Session 2: wider singleton and shared-input projections

Runs 019–031 continue from the previously delivered implementation. All use eight workers and the extended synthetic suite. Runs 027–030 use the same current harness, with the exact old source as control, in control/candidate/candidate/control order. Every run preserves complete raw records, resource counters and source, including rejected variants.

| Run | Decision | Packed layer ms | Reason |
| --- | --- | ---: | --- |
| [019](results/20260912T031003.208609Z-019-session2-baseline) | control | 28.347 | Fresh previous-version baseline. |
| [020](results/20260912T031031.542015Z-020-wide-singleton32) | rejected | 24.956 | 32-lane singleton candidate; 64 lanes performed better. |
| [021](results/20260912T031059.126002Z-021-wide-singleton64) | accepted | 22.515 | 64-lane singleton candidate reduced cached singleton time; retained in final combined source. |
| [022](results/20260912T031131.159682Z-022-pair-remainder) | rejected | 27.284 | Paired MoE remainder did not show consistent overall benefit. |
| [023](results/20260912T031337.893594Z-023-four-row-projection) | superseded | 24.652 | Four-row projection improved projection time; eight rows selected next. |
| [024](results/20260912T031425.923076Z-024-eight-row-projection) | accepted | 20.865 | Eight-row projection improved output projection and packed phase B; retained. |
| [025](results/20260912T031453.233495Z-025-combined-candidate) | rejected | 20.613 | Combined candidate included paired remainder; simpler version selected after comparison. |
| [026](results/20260912T031550.242682Z-026-combined-without-pair) | accepted | 22.635 | Wide singleton plus eight-row projection, without paired remainder. |
| [027](results/20260912T031625.855581Z-027-previous-version-control) | control | 24.777 | Exact previously delivered source with current harness; forward comparison. |
| [028](results/20260912T031658.936623Z-028-session2-final) | accepted | 21.870 | Final default-on 64-lane singleton and eight-row projection. |
| [029](results/20260912T031734.802691Z-029-session2-final-repeat) | accepted | 25.034 | Independent repeat of final selected source. |
| [030](results/20260912T031908.225658Z-030-previous-version-repeat) | control | 28.787 | Exact previous source repeated after candidate to expose scheduling drift. |
| [031](results/20260912T031933.552084Z-031-packed-weight-tile) | rejected | 21.577 | Weight preconversion improved isolated expert wall time but packed-layer CPU cost rose to 198.8 ms versus retained runs 163.0/174.9 ms. Wall results were noisy; added complexity rejected. |

Final comparisons pool 22 samples per case and variant from controls 027/030 and candidates 028/029. Scheduling noise remains substantial: these are measured medians, not confidence bounds or whole-model throughput claims. CPU time is process CPU per invocation, including all workers.

| Workload | Previous wall ms | New wall ms | Wall speedup | Previous CPU ms | New CPU ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| moe_e1_m1 | 0.128 | 0.119 | 1.08× | 0.128 | 0.119 |
| moe_e1_m4 | 0.218 | 0.207 | 1.06× | 0.218 | 0.206 |
| moe_e1_m16 | 0.661 | 0.640 | 1.03× | 0.661 | 0.640 |
| moe_e32_m1 | 2.162 | 2.156 | 1.00× | 13.448 | 13.067 |
| moe_e128_m1 | 7.911 | 7.589 | 1.04× | 57.350 | 54.397 |
| moe_e32_m4 | 3.757 | 3.776 | 0.99× | 24.621 | 24.454 |
| projection_qkv | 0.096 | 0.065 | 1.47× | 0.096 | 0.065 |
| projection_out | 0.064 | 0.038 | 1.66× | 0.064 | 0.038 |
| phase_b_t32 | 0.627 | 0.599 | 1.05× | 3.654 | 2.832 |
| phase_b_t256 | 6.388 | 5.437 | 1.17× | 46.775 | 39.201 |
| phase_b_packed_t256_s32 | 3.874 | 2.885 | 1.34× | 28.466 | 20.916 |
| layer_t32 | 8.435 | 8.215 | 1.03× | 56.489 | 54.065 |
| layer_packed_t256_s32 | 27.365 | 22.987 | 1.19× | 203.022 | 167.386 |

The singleton change widens independent output accumulators to 64 lanes while preserving each output’s FMA order. The projection change shares input loads across eight rows while retaining the original two-chain dot-product reduction exactly. Both are AVX-512 guarded; existing AVX2 and ARM paths remain available. `PF_NO_WIDE_AXPY` and `PF_NO_PROJ4` disable the respective additions; `PF_PROJ_ROWS` selects 1–8 rows and `PF_AXPY_NV` selects 1–8 groups of 16 output lanes. No paired-remainder or weight-packing experiment remains in the selected source.

Selected source SHA256: `971913470e097088096e8e1419fb3e03777f61d1163e5d16119df2b323c37f98`. All 13 cases in every session-2 trial match run 019 bitwise, with no failed reference values or unstable sample hashes; see `cross-run-checks-session2.json`. The projection tests additionally exercise input/output tails, bias, residual addition and output guards against an independent scalar FMA oracle. Full-model validation remains blocked by unavailable checkpoint downloads.

Session-2 validation: all 31 build, smoke, bitwise correctness, sanitizer and pool commands passed in `validation/verify-20260912T032225Z-63f1b94d/`. Native and sanitized projection checks each compared 5,404 outputs with zero bitwise mismatches.

Final review fixed projection-test skip handling on x86 hosts without AVX-512. A focused regression using the real AVX2 and native binaries passed six assertions, including rejection of malformed or unauthorized skips; see `validation/projection-skip-20260912T040329Z/`. Review also independently reproduced every wall and CPU median in `comparison-session2.json`.
