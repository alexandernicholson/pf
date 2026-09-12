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
