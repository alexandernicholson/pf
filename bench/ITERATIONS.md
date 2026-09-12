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
  --cflags '-D_GNU_SOURCE -O3 -std=c11 -march=native -DPF_NO_X86_MOE -DPF_NO_PROJ4 -DPF_NO_WIDE_AXPY -DPF_NO_MOE_TRIPLE'
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

## Session 3: broader workloads and a targeted remainder kernel

Runs 032–045 explore caller participation, larger register tiles, software prefetch, larger expert chunks, compiler vector-width preference and a three-token remainder. The attention investigation is preserved in `reviews/session3-attention.md` and its unmeasured diagnostic patch. Raw evidence was checkpointed during the session; rejected code remains in each measured run snapshot.

The new `diverse` profile appends nine cases to the original 13: expert batches of 3/7/15, uneven streaming occupancy, layers of 1/8 tokens, packed 512 tokens and full 512-token attention/layers. Original hashes are unchanged. New routing records contain every expert occupancy and chunk distribution, outside timed intervals. This remains synthetic, single-layer evidence with no trained-router or full-checkpoint validation.

| Run | Decision | Packed 256 layer ms | Reason |
| --- | --- | ---: | --- |
| [032](results/20260912T040946.531681Z-032-session3-baseline) | control | 21.236 | Fresh session-3 current-source control. |
| [033](results/20260912T041029.014841Z-033-caller-pool) | rejected | 24.492 | Caller participation did not improve the extended suite; consider separately on new short-input cases. |
| [034](results/20260912T041118.376258Z-034-diverse-control) | control | 22.666 | Broader 22-case suite; all original 13 hashes preserved. |
| [035](results/20260912T041233.497231Z-035-moe8x32) | rejected | 22.035 | 8x32 tile did not establish a broader layer win; targeted triple remainder is simpler and shows repeatable irregular-batch gains. |
| [036](results/20260912T041313.870196Z-036-moe6x64) | rejected | 19.897 | Six-token tile had mixed layer results and no reliable win in the irregular 15-item case; triple remainder selected instead. |
| [037](results/20260912T041458.134638Z-037-prefetch4) | rejected | 21.680 | Software prefetch had mixed timings and regressed uneven streaming expert work; no consistent broader benefit. |
| [038](results/20260912T041540.822203Z-038-chunks32) | rejected | 22.500 | 32-item chunks reduced dispatch count but increased both CPU and wall time in 512-token screens; default 16 retained. |
| [039](results/20260912T041631.603788Z-039-prefer512) | rejected | 24.261 | Preferring512bit compiler vectors helped some cases but regressed others; no consistent broad benefit. |
| [040](results/20260912T041737.772133Z-040-triple-tail) | accepted | 20.750 | Three-token remainder screening candidate; exact outputs retained and gain confirmed by final repeats 043/044. |
| [041](results/20260912T041850.448867Z-041-caller-diverse) | rejected | 20.896 | Caller participation repeat on diverse profile did not establish a short-input or broader-layer win. |
| [042](results/20260912T041925.797943Z-042-final-control) | control | 17.764 | Previous delivered source, current diverse harness; forward control. |
| [043](results/20260912T042022.508141Z-043-final-triple) | accepted | 20.452 | Final default-on AVX512 triple remainder; all 22 hashes match. |
| [044](results/20260912T042102.400443Z-044-final-triple-repeat) | accepted | 18.277 | Independent repeat of final source. |
| [045](results/20260912T042159.251248Z-045-final-control-repeat) | control | 24.663 | Previous source repeated after candidates; scheduling drift is visible on unchanged paths. |

The retained change shares BF16 weight loads/conversion across three remaining tokens after four-token blocks. It preserves each output’s increasing-k FMA order and activates by default only for the existing AVX512 expert path. `PF_NO_MOE_TRIPLE` disables it; `PF_MOE_TRIPLE` explicitly enables AVX2 experimentation. No caller-pool, larger tile, prefetch, larger chunk or compiler preference change was retained.

Final source SHA256: `5f870a795b83e632aa74a39a1d019865707558848dbc00eec639ee8d9e2f10e1`. All measured session-3 cases match their baseline hashes, with no failed reference values or unstable samples (`cross-run-checks-session3.json`).

### Repeated comparison

Controls 042/045 and candidates 043/044 each contribute 18 raw repetitions per case, in control/candidate/candidate/control order. Reproduce these calculations with:

```sh
python3 bench/compare.py --control 042 045 --candidate 043 044 \
  --output bench/comparison-session3.json
```

| Workload | Control wall ms | Candidate wall ms | Wall speedup | CPU speedup |
| --- | ---: | ---: | ---: | ---: |
| moe_e1_m1 | 0.120 | 0.117 | 1.02× | 1.02× |
| moe_e1_m4 | 0.221 | 0.210 | 1.06× | 1.06× |
| moe_e1_m16 | 0.695 | 0.659 | 1.05× | 1.05× |
| moe_e32_m1 | 2.190 | 2.016 | 1.09× | 1.09× |
| moe_e128_m1 | 7.554 | 7.422 | 1.02× | 1.02× |
| moe_e32_m4 | 3.879 | 3.749 | 1.03× | 0.97× |
| projection_qkv | 0.068 | 0.065 | 1.05× | 1.05× |
| projection_out | 0.039 | 0.046 | 0.86× | 0.86× |
| phase_b_t32 | 0.570 | 0.491 | 1.16× | 1.03× |
| phase_b_t256 | 5.269 | 4.883 | 1.08× | 1.05× |
| phase_b_packed_t256_s32 | 3.032 | 3.110 | 0.98× | 0.98× |
| layer_t32 | 8.933 | 6.871 | 1.30× | 1.22× |
| layer_packed_t256_s32 | 21.777 | 19.855 | 1.10× | 1.10× |
| moe_e1_m3 | 0.265 | 0.174 | 1.52× | 1.52× |
| moe_e1_m7 | 0.440 | 0.312 | 1.41× | 1.41× |
| moe_e1_m15 | 0.715 | 0.594 | 1.20× | 1.20× |
| moe_e32_uneven_m1_31 | 6.851 | 6.615 | 1.04× | 1.05× |
| layer_t1 | 0.627 | 0.479 | 1.31× | 1.21× |
| layer_t8 | 2.935 | 2.249 | 1.31× | 1.17× |
| layer_packed_t512_s64 | 41.026 | 34.573 | 1.19× | 1.18× |
| phase_b_t512 | 11.124 | 10.357 | 1.07× | 1.07× |
| layer_t512 | 41.621 | 37.392 | 1.11× | 1.11× |

The defensible gain is the targeted irregular expert kernel: 3/7/15 assignments improve by 1.52×/1.41×/1.20× in pooled medians, and both candidate runs beat both control medians in those cases. Broader pooled layer medians look better, but cannot cleanly separate the change from host/scheduling drift: the one-token layer does not exercise the triple path yet moves substantially, and unchanged output projection regresses in the pooled comparison. No reliable overall-layer speedup is claimed for this round. All outliers and regressions remain included; no samples were discarded. CPU time is aggregate process CPU divided by each repetition’s invocation count, not elapsed wall time or unnormalized batch CPU.

Final validation passed all 31 build, smoke, correctness, sanitizer and pool commands in `validation/verify-20260912T042312Z-48c0abd4/`. Native/AVX2/sanitized expert tests each passed 30 cases, including three-token remainders after multiple full blocks and a partial reduction tile. Native and sanitized projection tests each passed 5,404 exact comparisons.
