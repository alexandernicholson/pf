#!/usr/bin/env python3
"""Exercise the actual verification runner against real kernels and bad skip records."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
verify_path = ROOT / "bench/verify.py"
spec = importlib.util.spec_from_file_location("pf_verify", verify_path)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
summary = {"status": "running", "scope": "Projection verification skip-handling regression; deliberately rejected commands are retained",
           "commands": [], "skipped_commands": [], "assertions": [],
           "source_sha256": {p: verify.digest(ROOT/p) for p in ("pf.c", "bench/verify.py", "bench/test_projection.c")}}
verify.save(out / "summary.json", summary)
(out / "verify.py.snapshot").write_bytes(verify_path.read_bytes())
# Extract the unchanged nested runner so the test exercises its production logic.
module = ast.parse(verify_path.read_text())
main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
runner = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "run")
namespace = dict(vars(verify), ROOT=ROOT, out=out, summary=summary, args=SimpleNamespace(timeout=30))
exec(compile(ast.Module(body=[runner], type_ignores=[]), str(verify_path), "exec"), namespace)
run = namespace["run"]
reason = "projection kernel unavailable"

def check(name, command, allowed_reason, expected_pass, expected_skip):
    rejected = False
    try:
        run(name, command, skip_reason=allowed_reason)
    except RuntimeError:
        rejected = True
    result = summary["commands"][-1]
    assert rejected == (not expected_pass), (name, rejected)
    assert result["passed"] == expected_pass, name
    assert result["skipped"] == expected_skip, name
    summary["assertions"].append({"name": name, "expected_pass": expected_pass,
                                  "expected_skip": expected_skip, "matched": True})

try:
    with tempfile.TemporaryDirectory(prefix="pf-projection-skip-") as tmp:
        for name, flags in (("avx2", ["-march=x86-64", "-mavx2", "-mfma"]), ("native", ["-march=native"])):
            binary = Path(tmp) / name
            run("build-"+name, ["cc", "-O3", "-std=c11", "-pthread", *flags, "bench/test_projection.c", "-lm", "-o", binary])
            if name == "avx2":
                check("avx2-allowed-skip", [binary], reason, True, True)
                check("avx2-skip-not-authorized", [binary], None, False, False)
            else:
                check("native-real-pass", [binary], reason, True, False)
        def python_exit(text, code):
            return [sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(int(sys.argv[2]))", text, str(code)]
        check("malformed-skip", python_exit("not JSON", 77), reason, False, False)
        check("wrong-reason", python_exit(json.dumps({"status":"skip", "reason":"unexpected"}),77), reason, False, False)
        check("wrong-exit", python_exit(json.dumps({"status":"skip", "reason":reason}),1), reason, False, False)
    assert summary["skipped_commands"] == ["avx2-allowed-skip"]
    summary["status"] = "passed"
except Exception as exc:
    summary.update(status="failed", error=repr(exc))
    raise
finally:
    summary["script_sha256"] = verify.digest(Path(__file__))
    verify.save(out / "summary.json", summary)
    print(json.dumps({"status":summary["status"], "assertions":len(summary["assertions"]), "skipped_commands":summary["skipped_commands"]}))
