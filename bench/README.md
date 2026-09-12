# Cloud CPU benchmark harness

`run.py` builds a frozen copy of `pf.c` and includes its actual static inference
functions in `kernel_bench.c`. No model download is needed. These are synthetic
kernel and layer measurements, **not model accuracy or end-to-end privacy-filter
throughput measurements**.

`make check` runs the portable build and correctness checks, saving commands,
compiler output and test output under `bench/validation/`. The final cloud
validation passed native, AVX2/FMA, generic scalar and AVX2-without-FMA production
builds with `-Werror`. Native, AVX2 and AddressSanitizer/UndefinedBehaviorSanitizer
expert tests each passed 25 cases with zero bitwise mismatches. Pool checks cover
repeated dispatch and invalid settings. LeakSanitizer scanning is unsupported
in this container; address/UB instrumentation remains enabled. The earlier
test-only warning and leak-scanner failures are retained in their original logs.

```sh
python3 bench/run.py --label baseline
python3 bench/run.py --label candidate --note 'Describe the one change under test'
python3 bench/run.py --label parallel --parallel --profile extended
PF_THREADS=4 python3 bench/run.py --label threads-4 --parallel --profile extended
python3 bench/run.py --label upstream --source /path/to/upstream-pf.c
```

Default flags are `-D_GNU_SOURCE -O3 -std=c11 -Wall -Wextra -march=native` and
`-lm -pthread`. Pass `--cc`, `--cflags`, and `--ldflags` explicitly when comparing
compilers or ISA options. `-D_GNU_SOURCE` is required to compile the upstream
Linux source correctly. `PF_THREADS` only affects source revisions that implement
that setting; it is always captured as provenance. Serial invocation is the
default. `--parallel` uses the source revision's real `pfor` implementation.

The default profile contains:

| Cases | What runs |
| --- | --- |
| `moe_e1_m1`, `moe_e1_m4`, `moe_e1_m16` | Actual `ph_c`, one expert receiving 1, 4 or 16 assignments |
| `moe_e32_m1`, `moe_e128_m1`, `moe_e32_m4` | Actual `ph_c`, streaming 32 or 128 distinct experts |
| `projection_qkv`, `projection_out` | Production dot kernel over full 640→1152 and 896→640 projections |
| `phase_b_t32`, `phase_b_t256` | Actual attention, output projection, residual, RMS norm and router |
| `phase_b_packed_t256_s32` | The same phase, 256 tokens split into eight independent 32-token documents |

`--profile extended` adds `layer_t32` and `layer_packed_t256_s32`: one complete
synthetic layer using actual `ph_a`, `ph_b` and `ph_c`, with routing and chunking
matching `forward`. All 128 experts are independently populated, including the
300 MiB of BF16 expert matrices. This is one layer's working set, not the full
eight-layer model. The synthetic weights are reproducible uniform BF16 values;
the router distribution is not claimed to match trained model behavior.

Each case performs a correctness run, two calibration calls, then five raw timed
repetitions. Calibration targets 25 ms per repetition, with at least one whole
call. Change these using `--target-ms` and `--reps`. Slow or heavily parallel
configurations can exceed the target because calls are indivisible. Setup and
reference calculations are outside timing. Phase B and layer measurements
include the small reset copy needed for deterministic repeated residual updates.

Correctness uses independently written double precision references. Projection
outputs and small expert blocks are checked in full. Streaming workloads check
all output rows and all items of the first, middle and final expert; attention
checks the first, middle and final token's attention values, residual, norm,
router choices and probabilities. Layer cases also check sampled expert results
against their actual phase inputs. Tolerance is `2e-5 + 2e-4 * abs(reference)`.
This is component numerical validation; it does not establish model-level
classification parity. Exact output hashes must remain stable across repetitions
of a run. Different ISA reduction orders may legitimately change hashes between
runs, so inter-run numerical validation uses the reference errors rather than
requiring equal hashes.

Results are append-only directories under `bench/results/<UTC>-<label>/`:

- `manifest.json`: status, arguments, compiler, source hashes, Git state, timing
  process metadata and summary.
- `stdout.jsonl`, `stderr.txt`: complete benchmark output, including failed runs.
- `records.json`: every parsed JSON record and unknown metric without filtering.
- `summary.json`: median, mean, extrema, spread and sample count per case.
- `host-before.json`, `host-after.json`: CPU, affinity, memory, load, cgroup limits,
  throttling counters, and relevant thread/compiler environment settings.
- `source/`, `source.patch`, `compile.stdout`, `compile.stderr`: exact source
  snapshots and compiler evidence. The executable is locally retained and Git
  ignored; the captured sources and command rebuild it.

Per-repetition records retain monotonic wall time, process CPU time, user/system
time, context switches, page faults, process maximum RSS, and output hashes.
`cumulative_children_maxrss_kib` in the manifest can include the compiler and is
explicitly not the benchmark's own RSS; use each C sample's `max_rss_kib`.
Raw observations allow later charts of metrics not selected for the initial
summary. A failed build, timeout, malformed record, failed reference check, or
unstable output leaves its evidence directory and returns failure.

`python3 bench/report.py` regenerates the standalone `report.html` chart and
`metrics.csv.gz` from the raw archive. Select a case and any captured numeric
metric; runs with different thread/compiler settings remain separate series.
The chart embeds its data and needs no network connection. The uncompressed
CSV is a rebuildable local convenience; the compressed CSV retains every row.

## Optional checkpoint validation

When the real checkpoint and tokenizer are available, use:

```sh
python3 bench/model_bench.py --baseline ./pf-baseline --candidate ./pf \
  --model ./model --corpus /path/to/test-corpus.jsonl
```

The corpus must contain JSONL objects with `id` and single-line `text` fields.
This runner archives its input and raw outputs; use inputs suitable for the
intended results destination. It checks tokens, masks, spans, logprobs, and
packed-versus-separate behavior before timing. CLI timing includes startup;
warmup does not guarantee page-cache residency. It fails explicitly when
required model files are missing. No model-level run was possible in the
initial cloud experiment. The initial optional generated text corpus was
removed; the numeric kernel archive contains no model input documents.

The runner uses `/tmp/pf-cloud-benchmark.lock` to serialize benchmark processes.
Avoid other CPU-intensive work while timing; an advisory lock cannot prevent
unrelated workloads or host contention. Compare identical harness hashes,
flags, profiles, affinity and thread settings, and repeat promising results to
separate changes from cloud-machine noise. Keep rejected experiments and their
reason alongside accepted iterations. Commit the source, result JSON and raw
evidence to the repository to preserve them beyond the current machine.
