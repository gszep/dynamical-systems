"""Single sequential Julia invocation, owned process-group cleanup, no setup.

Usage: python3 run_bounded.py --julia PATH --record FILE -- SCRIPT [ARGS...]
Optional --expected-exit 1 --expected-message TEXT checks a negative route.
No installation, no background subprocesses, timeout 270s including import.
This launcher records raw exit status; expecting a negative result never
turns its scientific gate into a passing result.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--julia", required=True)
p.add_argument("--record", required=True)
p.add_argument("--expected-exit", type=int, default=0)
p.add_argument("--expected-message", default="")
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ["--"] else a.command
start = time.monotonic()
cwd = Path(__file__).resolve().parent
record = Path(a.record).resolve()
record.parent.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(JULIA_DEPOT_PATH=str(cwd.parents[1] / ".julia-depot"),
           JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           JULIA_PKG_PRECOMPILE_AUTO="0")
cmd = [a.julia, "--startup-file=no", "-L", str(cwd / "cache_only.jl"),
       "--project=" + str(cwd / "pkgenv")] + command
setup_done = time.monotonic()
timed_out = False
with record.with_suffix(".log").open("w+") as log:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=270)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
    finally:
        # Never leave invocation-owned descendants after success or failure.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    end = time.monotonic()
    log.seek(0)
    output = log.read()
matched = (not timed_out and proc.returncode == a.expected_exit
           and a.expected_message in output)
record.write_text(json.dumps(dict(command=cmd, raw_exit=proc.returncode,
    expected_exit=a.expected_exit, expected_message=a.expected_message,
    expectation_matched=matched, timed_out=timed_out, queue_seconds=0.0,
    queue_note="No launcher queue; earlier human resource hold not timed here",
    setup_seconds=setup_done-start, execution_seconds=end-setup_done,
    total_seconds=end-start, no_package_setup=True), indent=2) + "\n")
print(output[-2400:])
print(f"raw_exit={proc.returncode} expectation_matched={matched} total={end-start:.2f}s")
raise SystemExit(0 if matched else 1)
