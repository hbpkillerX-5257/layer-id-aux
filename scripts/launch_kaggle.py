#!/usr/bin/env python3
"""Push one private GPU kernel per distinct Kaggle account.

Tokens are read from ~/.kaggle. Files that resolve to the same username share
one weekly GPU budget, so only the first token for that user is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

TOKEN_DIR = Path.home() / ".kaggle"
TOKEN_FILES = [
    "access_token",
    "access_token_1",
    "access_token_2",
    "access_token_3",
    "access_token_4",
    "access_token_5",
    "access_token_6",
]
REPO = "https://github.com/hbpkillerX-5257/layer-id-aux.git"
MACHINE = "NvidiaTeslaP100"
TIMEOUT_SECONDS = "5400"

# One run per distinct account. Seeds are paired on the comparison that matters.
RUNS = [
    {"slug": "conf-noise-brier-l1-s1", "mode": "noise_brier", "lam": 1.0, "seed": 1},
    {"slug": "conf-noise-brier-l1-s2", "mode": "noise_brier", "lam": 1.0, "seed": 2},
    {"slug": "conf-noise-ce-s1", "mode": "noise_ce", "lam": 1.0, "seed": 1},
    {"slug": "conf-noise-ce-s2", "mode": "noise_ce", "lam": 1.0, "seed": 2},
]


def load_accounts() -> list[dict[str, str]]:
    accounts: list[dict[str, str]] = []
    seen: set[str] = set()
    for name in TOKEN_FILES:
        path = TOKEN_DIR / name
        if not path.exists():
            continue
        token = path.read_text().strip()
        if not token:
            continue
        request = urllib.request.Request(
            "https://www.kaggle.com/api/v1/hello",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            username = json.loads(response.read().decode())["userName"]
        if username in seen:
            print(f"skip {name}: {username} already has a token", flush=True)
            continue
        seen.add(username)
        accounts.append({"username": username, "token": token, "file": name})
    return accounts


def quota_text(token: str) -> str:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    result = subprocess.run(["kaggle", "quota"], capture_output=True, text=True, env=env)
    text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(text)
    return text


def remaining_gpu_hours(quota: str) -> float | None:
    for line in quota.splitlines():
        if "GPU" not in line:
            continue
        # Table rows look like: GPU  1.20h  28.80h  30.00h  <refresh>
        parts = line.split()
        for part in parts:
            if part.endswith("h") and part[:-1].replace(".", "", 1).isdigit():
                # first hours field is used, second is remaining
                pass
        hours = [float(part[:-1]) for part in parts if part.endswith("h")]
        if len(hours) >= 2:
            return hours[1]
    return None


def kernel_script(sha: str, run: dict) -> str:
    cmd = [
        "__PYTHON__",
        "-m",
        "src.train",
        "--mode",
        run["mode"],
        "--lam",
        str(run["lam"]),
        "--seed",
        str(run["seed"]),
        "--steps",
        "4000",
        "--out-dir",
        "/kaggle/working/results",
        "--data-dir",
        "/kaggle/working/data",
    ]
    return f"""import subprocess
import sys

repo = {REPO!r}
dest = "/kaggle/working/layer-id-aux"
expected = {sha!r}
subprocess.check_call(["git", "clone", "--depth", "1", repo, dest])
sha = subprocess.check_output(["git", "-C", dest, "rev-parse", "HEAD"], text=True).strip()
print("checked out", sha, flush=True)
if sha != expected:
    raise SystemExit(f"sha mismatch: {{sha}} != {{expected}}")
cmd = {cmd!r}
cmd[0] = sys.executable
print("running", cmd, flush=True)
raise SystemExit(subprocess.call(cmd, cwd=dest))
"""


def write_kernel(folder: Path, username: str, sha: str, run: dict) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    slug = run["slug"]
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": MACHINE,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (folder / "run.py").write_text(kernel_script(sha, run))


def push_kernel(folder: Path, token: str) -> None:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(folder), "-t", TIMEOUT_SECONDS],
        check=True,
        env=env,
    )


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota-only", action="store_true")
    parser.add_argument("--min-gpu-hours", type=float, default=1.5)
    args = parser.parse_args()

    accounts = load_accounts()
    print(f"{len(accounts)} distinct accounts", flush=True)
    usable = []
    for account in accounts:
        text = quota_text(account["token"])
        remaining = remaining_gpu_hours(text)
        print(f"\n{account['username']} ({account['file']}) remaining_gpu_h={remaining}", flush=True)
        print(text, flush=True)
        account["remaining"] = remaining
        if remaining is None or remaining >= args.min_gpu_hours:
            usable.append(account)
        else:
            print(f"skip {account['username']}: under {args.min_gpu_hours}h GPU left", flush=True)

    if args.quota_only:
        return
    if len(usable) < len(RUNS):
        raise SystemExit(
            f"need {len(RUNS)} accounts with GPU budget, found {len(usable)}. "
            "Refusing to stack two runs on one account."
        )

    sha = git_sha()
    print(f"pinning {sha}", flush=True)
    root = Path("/tmp/layer-id-aux-kernels")
    root.mkdir(parents=True, exist_ok=True)
    for account, run in zip(usable, RUNS):
        folder = root / run["slug"]
        write_kernel(folder, account["username"], sha, run)
        print(f"push {account['username']}/{run['slug']}", flush=True)
        push_kernel(folder, account["token"])
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
