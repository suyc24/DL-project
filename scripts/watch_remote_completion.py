#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_REMOTE = "root@connect.nmb1.seetacloud.com"
DEFAULT_PORT = "39314"
DEFAULT_REMOTE_DIR = "/root/fhis-verification-signal"
DEFAULT_DONE_FILE = "data/generated_traces.jsonl"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def ssh_cmd(args: argparse.Namespace, remote_script: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"UserKnownHostsFile={args.known_hosts}",
        "-p",
        args.port,
        args.remote,
        remote_script,
    ]


def remote_status(args: argparse.Namespace) -> tuple[bool, str]:
    script = f"""
set +e
cd {args.remote_dir}
active=$(ps -eo pid,cmd | grep -E 'python -m fhis|VLLM::EngineCore|run_generate|run_hidden|run_probe|run_full' | grep -v grep || true)
done_file={args.done_file}
if [ -n "$active" ]; then
  echo "ACTIVE"
  echo "$active"
elif [ -s "$done_file" ]; then
  echo "DONE"
  wc -l "$done_file"
else
  echo "WAITING"
  ls -lh data logs 2>/dev/null | tail -n 40
fi
"""
    completed = run(ssh_cmd(args, script), check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        return False, output
    return output.startswith("DONE"), output


def pull_artifacts(args: argparse.Namespace) -> None:
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    remote_base = f"{args.remote}:{args.remote_dir}"
    run(
        [
            "scp",
            "-r",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={args.known_hosts}",
            "-P",
            args.port,
            f"{remote_base}/data/.",
            "data/",
        ]
    )
    for directory in ("logs", "results", "figures"):
        run(
            [
                "scp",
                "-r",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                f"UserKnownHostsFile={args.known_hosts}",
                "-P",
                args.port,
                f"{remote_base}/{directory}/.",
                f"{directory}/",
            ],
            check=False,
        )


def run_labeling(args: argparse.Namespace) -> None:
    if not args.label:
        print("labeling skipped: --label was not set", flush=True)
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("labeling skipped: OPENAI_API_KEY is not set", flush=True)
        return
    cmd = [
        sys.executable,
        "-m",
        "fhis.label_with_openai",
        "--config",
        args.config,
        "--resume",
    ]
    if args.label_limit is not None:
        cmd.extend(["--limit", str(args.label_limit)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for the remote FHIS run to finish, pull artifacts, and optionally label traces."
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--done-file", default=DEFAULT_DONE_FILE)
    parser.add_argument("--known-hosts", default="/tmp/fhis_known_hosts")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--timeout-minutes", type=int, default=240)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--label-limit", type=int, default=None)
    args = parser.parse_args()

    deadline = time.time() + args.timeout_minutes * 60
    while True:
        done, status = remote_status(args)
        print(status.rstrip(), flush=True)
        if done:
            pull_artifacts(args)
            run_labeling(args)
            print("watch complete", flush=True)
            return
        if time.time() >= deadline:
            raise TimeoutError(f"remote experiment did not finish within {args.timeout_minutes} minutes")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
