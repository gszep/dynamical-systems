#!/usr/bin/env python3
"""Bounded local validation; outputs are invocation-owned, never overwritten.

No package installation is performed. Run only after the package setup owner
has finished. Logs and timings remain under .tools/runs for failure diagnosis.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "01-equilibria": ("tests.jl", "main.jl"),
    "02-cycles": ("tests.jl", "main.jl"),
    "03-patterns": ("scripts/tests.jl", "scripts/main.jl"),
}


def drain(process):
    """Terminate only this invocation's process group, then reap its leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Descendants may outlive the leader, so kill the group even if wait returned.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", type=Path,
                        default=ROOT / ".tools/julia-1.10.11/bin/julia")
    parser.add_argument("--stages", nargs="+", choices=STAGES,
                        default=list(STAGES))
    parser.add_argument("--phase", choices=("tests", "smoke", "all"), default="all")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not 0 < args.timeout <= 1800:
        parser.error("timeout must be in (0, 1800] seconds per process")
    if len(args.stages) != len(set(args.stages)):
        parser.error("duplicate stages would share mutable fixtures")
    julia = args.julia.resolve(strict=True)
    start = time.monotonic()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / ".tools/runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    active, lock, stop = set(), threading.Lock(), threading.Event()

    def cancel(signum, frame):
        stop.set()
        with lock:
            for process in list(active):
                drain(process)

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    setup_seconds = time.monotonic() - start
    submitted = time.monotonic()

    def stage(name):
        began = time.monotonic()
        folder = run_dir / name
        folder.mkdir()
        env = os.environ.copy()
        env.update(JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", JULIA_NUM_PRECOMPILE_TASKS="1",
                   JULIA_DEPOT_PATH=str(ROOT / ".julia-depot" / name))
        records = []
        for phase, script in zip(("tests", "smoke"), STAGES[name]):
            if args.phase not in ("all", phase):
                continue
            command = [str(julia), "--startup-file=no", "--threads=1",
                       "--project=" + str(ROOT / "experiments" / name),
                       str(ROOT / "experiments" / name / script)]
            if phase == "smoke":
                command += ["--out-dir", str(folder / "artifacts")]
            tick = time.monotonic()
            record = {"phase": phase, "command": command, "timeout": False}
            try:
                with (folder / (phase + ".log")).open("w") as log:
                    with lock:
                        if stop.is_set():
                            record.update(returncode=130, cancelled=True)
                            records.append(record)
                            break
                        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        active.add(process)
                    try:
                        record["returncode"] = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        drain(process)
                        record.update(returncode=124, timeout=True)
                    finally:
                        # No child processes are expected, but own the entire group.
                        drain(process)
                        with lock:
                            active.discard(process)
            except OSError as error:
                record.update(returncode=127, error=str(error))
            record["execution_seconds"] = time.monotonic() - tick
            records.append(record)
            print(name, phase, record, flush=True)
        return {"stage": name, "queue_seconds": began - submitted,
                "total_seconds": time.monotonic() - began, "processes": records}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(stage, args.stages))
    summary = {"run_id": stamp, "setup_seconds": setup_seconds,
               "source_head_at_start": source_head,
               "source_status_at_start": source_status,
               "total_seconds": time.monotonic() - start,
               "budget": {"max_processes": args.jobs, "threads_per_process": 1},
               "cancelled": stop.is_set(), "stages": results}
    (run_dir / "timings.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Evidence:", run_dir, flush=True)
    return int(stop.is_set() or any(
        r["returncode"] != 0 for s in results for r in s["processes"]))


if __name__ == "__main__":
    raise SystemExit(main())
