#!/home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python -I
"""Isolated, sealed-runtime launcher for the frozen GCSPO Stage-0 R1 completion."""
import sys
if not sys.flags.isolated:
    raise SystemExit("R1 protected launcher must be executed directly under its -I shebang")

import fcntl
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

sys.dont_write_bytecode = True
_GIT = "/usr/bin/git"
_ACTIVE = "GCSPO_R1_SNAPSHOT_ACTIVE"
_FD_MAP = "GCSPO_R1_SEALED_FD_MAP"


def _reviewed_snapshot(root: Path) -> tuple[Path, str]:
    wrapper = subprocess.run([_GIT, "-C", str(root), "rev-parse", "HEAD"], check=True,
                             text=True, capture_output=True).stdout.strip()
    target = subprocess.run([_GIT, "-C", str(root), "rev-parse", f"{wrapper}^"], check=True,
                            text=True, capture_output=True).stdout.strip()
    archive = subprocess.run([_GIT, "-C", str(root), "archive", "--format=tar", target],
                             check=True, capture_output=True).stdout
    snapshot = Path(tempfile.mkdtemp(prefix=f"gcspo-r1-reviewed-{target[:12]}-"))
    snapshot.chmod(0o700)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        base = snapshot.resolve()
        for member in bundle.getmembers():
            destination = (snapshot / member.name).resolve()
            if not destination.is_relative_to(base) or not (member.isdir() or member.isfile()):
                raise ValueError(f"unsafe reviewed archive member: {member.name}")
        bundle.extractall(snapshot)
    return snapshot, target


def _git_blob(root: Path, target: str, relative: str) -> bytes:
    return subprocess.run([_GIT, "-C", str(root), "show", f"{target}:{relative}"],
                          check=True, capture_output=True).stdout


def _seal_snapshot(snapshot: Path, root: Path, target: str) -> dict[str, int]:
    descriptors: dict[str, int] = {}
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        payload = path.read_bytes()
        if payload != _git_blob(root, target, relative):
            raise RuntimeError(f"reviewed snapshot differs from target Git blob: {relative}")
        descriptor = os.memfd_create(f"gcspo-r1:{relative}", os.MFD_ALLOW_SEALING)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        os.set_inheritable(descriptor, True)
        descriptors[relative] = descriptor
    return descriptors


def _validated_sealed_state(root: Path, target: str) -> dict[str, int]:
    actual_target = subprocess.run([_GIT, "-C", str(root), "rev-parse", "HEAD^"], check=True,
                                   text=True, capture_output=True).stdout.strip()
    if target != actual_target:
        raise RuntimeError("sealed runtime target does not match authorization-wrapper parent")
    raw = json.loads(os.environ[_FD_MAP])
    descriptors = {str(key): int(value) for key, value in raw.items()}
    expected = set(subprocess.run(
        [_GIT, "-C", str(root), "ls-tree", "-r", "--name-only", target], check=True,
        text=True, capture_output=True).stdout.splitlines())
    if set(descriptors) != expected:
        raise RuntimeError("sealed runtime exact Git path inventory mismatch")
    required = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    for relative, descriptor in descriptors.items():
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != required:
            raise RuntimeError(f"runtime descriptor is not fully sealed: {relative}")
        size = os.fstat(descriptor).st_size
        payload = os.pread(descriptor, size, 0)
        if len(payload) != size or payload != _git_blob(root, target, relative):
            raise RuntimeError(f"sealed runtime Git identity mismatch: {relative}")
    return descriptors


class _SealedLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, relative: str, descriptor: int, origin: str, package: bool):
        self.fullname, self.relative, self.descriptor = fullname, relative, descriptor
        self.origin, self.package = origin, package

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        source = os.read(self.descriptor, os.fstat(self.descriptor).st_size)
        module.__file__ = self.origin
        if self.package:
            module.__path__ = [str(Path(self.origin).parent)]
        exec(compile(source, self.origin, "exec"), module.__dict__)


class _SealedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, snapshot: Path, descriptors: dict[str, int]):
        self.snapshot, self.descriptors = snapshot, descriptors

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gnss_doppler_lab":
            relative, package = "src/gnss_doppler_lab/__init__.py", True
        elif fullname.startswith("gnss_doppler_lab."):
            relative = "src/" + fullname.replace(".", "/") + ".py"
            package = False
        else:
            return None
        descriptor = self.descriptors.get(relative)
        if descriptor is None:
            return None
        origin = str(self.snapshot / relative)
        loader = _SealedLoader(fullname, relative, descriptor, origin, package)
        return importlib.util.spec_from_loader(fullname, loader, origin=origin, is_package=package)


def _activate_sealed_imports(snapshot: Path) -> None:
    raw = json.loads(os.environ[_FD_MAP])
    descriptors = {str(key): int(value) for key, value in raw.items()}
    sys.meta_path.insert(0, _SealedFinder(snapshot, descriptors))


def main() -> int:
    active = os.environ.get(_ACTIVE)
    if not active:
        root = Path(os.environ.get("GCSPO_R1_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve(strict=True)
        snapshot, target = _reviewed_snapshot(root)
        descriptors = _seal_snapshot(snapshot, root, target)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.update({
            "GCSPO_R1_REPO_ROOT": str(root),
            "GCSPO_R1_RUNTIME_SNAPSHOT": str(snapshot),
            _ACTIVE: target,
            _FD_MAP: json.dumps(descriptors, sort_keys=True, separators=(",", ":")),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        launcher_fd = descriptors["scripts/run_gcspo_stage0_r1.py"]
        os.execve(sys.executable,
                  [sys.executable, "-I", f"/proc/self/fd/{launcher_fd}"], environment)
    root = Path(os.environ["GCSPO_R1_REPO_ROOT"]).resolve(strict=True)
    snapshot = Path(os.environ["GCSPO_R1_RUNTIME_SNAPSHOT"]).absolute()
    descriptors = _validated_sealed_state(root, active)
    _activate_sealed_imports(snapshot)
    source = snapshot / "src"
    sys.path[:] = [str(source)] + [entry for entry in sys.path
                                  if not Path(entry or os.getcwd()).resolve().is_relative_to(snapshot)]
    from gnss_doppler_lab.gcspo_r1_runner import main as runner_main
    return runner_main()


if __name__ == "__main__":
    raise SystemExit(main())
