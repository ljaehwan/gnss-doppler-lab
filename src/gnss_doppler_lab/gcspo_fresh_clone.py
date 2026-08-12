"""Fresh-clone isolation helpers for GCSPO verification."""
from __future__ import annotations

from pathlib import Path
import subprocess


def _run(command, *, cwd=None):
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def clone_exact(repo_url, commit, destination):
    if len(commit) != 40: raise ValueError("exact 40-hex target commit is required")
    target = Path(destination)
    if target.exists(): raise ValueError("fresh clone destination already exists")
    _run(["git", "clone", "--no-checkout", "--", str(repo_url), str(target)])
    _run(["git", "checkout", "--detach", commit], cwd=target)
    head = _run(["git", "rev-parse", "HEAD"], cwd=target)
    status = _run(["git", "status", "--porcelain"], cwd=target)
    if head != commit or status: raise ValueError("fresh clone checkout is not exact and clean")
    return {"root": str(target.resolve()), "head": head, "status": status,
            "repo_url": str(repo_url), "tracked_only": True}
