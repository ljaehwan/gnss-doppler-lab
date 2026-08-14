"""Round-5 signed witness index and parity reconstruction."""
from __future__ import annotations

import json
from pathlib import Path

from .gcspo_provenance import compare_full_a5_runs, verify_witnessed_runs


def verify_round5_a5(root: str | Path):
    artifact = Path(root).resolve()
    repo = artifact.parents[1]
    index_path = artifact / "round5_a5_provenance.json"
    if not index_path.is_file():
        raise ValueError("round-5 signed A5 provenance index is absent")
    index = json.loads(index_path.read_text())
    if index.get("schema") != "gnss-doppler-lab.gcspo-stage0.round5-a5-provenance-index.v1":
        raise ValueError("round-5 signed A5 provenance index schema mismatch")
    source_commit = index.get("source_commit")
    challenge_path = index.get("challenge_path")
    evidence = index.get("evidence_roots")
    if (not isinstance(source_commit, str) or len(source_commit) != 40 or
            not isinstance(challenge_path, str) or
            not isinstance(evidence, list) or len(evidence) != 3):
        raise ValueError("round-5 signed A5 provenance index is incomplete")
    roots = []
    for relative in evidence:
        path = (repo / relative).resolve(strict=True)
        if repo not in path.parents:
            raise ValueError("round-5 signed A5 evidence path escapes repository")
        roots.append(path)
    verified = verify_witnessed_runs(
        roots, repo_root=repo, source_commit=source_commit,
        challenge_path=challenge_path)
    parity = compare_full_a5_runs(verified)
    parity_path = (repo / index.get("parity_report", "")).resolve(strict=True)
    if repo not in parity_path.parents or json.loads(parity_path.read_text()) != parity:
        raise ValueError("round-5 signed A5 parity report differs from reconstruction")
    return {"status": "PASS", "witnessed": verified, "parity": parity}
