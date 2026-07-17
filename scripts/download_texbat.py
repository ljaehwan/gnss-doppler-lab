#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FILES = {
    "The Texas Spoofing Test Battery_1.pdf": {"file_id": "638e098ebae2f1393c118965", "md5": "842ef8922b56e21ac8a8d5a962b53f5e", "size": 9032074, "subdir": "docs"},
    "cleanDynamic.bin": {"file_id": "638e5e7331dc39a3feb6e778", "md5": "95726eb42e0599c9cec38b81d78119f9", "size": 44291850240, "subdir": "raw"},
    "cleanStatic.bin": {"file_id": "638e667f31dc39a3feb6e779", "md5": "fcba7a32fd45e3293397559db185651d", "size": 48016392192, "subdir": "raw"},
    "ds1.bin": {"file_id": "638eb89031dc39a3feb6e787", "md5": "fc8819beb0b7590f612328a9b704675d", "size": 46170898432, "subdir": "raw"},
    "ds2.bin": {"file_id": "638ec3f831dc39a3feb6e78a", "md5": "51e0d0d8853141a608a4dcd0e97fdbc2", "size": 45717913600, "subdir": "raw"},
    "ds3.bin": {"file_id": "638f05ff8c39720bda8331be", "md5": "5405f1fb4f24d02ff0a4baa9534ecf7d", "size": 45785022464, "subdir": "raw"},
    "ds4.bin": {"file_id": "638f05ff8c39720bda8331bd", "md5": "1BEFFC0DEE6CD72972CEC55BA10EFE1D", "size": 12821987328, "subdir": "raw"},
    "ds5.bin": {"file_id": "638f05ff8c39720bda8331bc", "md5": "C71F31FF1C935FFB4C6577BE1EA850C5", "size": 41825599488, "subdir": "raw"},
    "ds6.bin": {"file_id": "638f05ff8c39720bda8331bb", "md5": "5C2712E93CF105009A11937C229321A3", "size": 41825599488, "subdir": "raw"},
    "ds7.bin": {"file_id": "638f2e699f9d8f05bc7f1167", "md5": "BA6A33A536F8FF60B98A2DE3289610B5", "size": 27059290112, "subdir": "raw"},
    "ds8.bin": {"file_id": "638f2e699f9d8f05bc7f1166", "md5": "35f672b90c8f792bc3daa01495fe52e4", "size": 47000141168, "subdir": "raw"},
    "readme.txt": {"file_id": "638e098ebae2f1393c118963", "md5": "a865248b745640f07a09629be05a88f0", "size": 78, "subdir": "docs"},
    "texbat_ds7_and_ds8.pdf": {"file_id": "638e098ebae2f1393c118964", "md5": "3df897cb0aaf2680ef41a99bd5305097", "size": 93170, "subdir": "docs"},
}
SCIDB_BASE = "https://china.scidb.cn/download?fileId="
UT_BASE_URLS = ["https://rnl-data.ae.utexas.edu/datastore/texbat", "https://radionavlab.ae.utexas.edu/datastore/texbat"]

def digest_file(path: Path, algo: str, block_size: int = 1024 * 1024 * 8) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def run_curl(url: str, dst: Path, connect_timeout: int, max_time: int) -> subprocess.CompletedProcess[str]:
    tmp = dst.with_suffix(dst.suffix + ".part")
    cmd = ["curl", "-L", "--fail", "--continue-at", "-", "--connect-timeout", str(connect_timeout), "-A", "Mozilla/5.0", "-o", str(tmp), url]
    if max_time > 0:
        cmd[4:4] = ["--max-time", str(max_time)]
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def urls_for(name: str, source: str) -> list[str]:
    meta = FILES[name]
    urls: list[str] = []
    if source in {"scidb", "auto"}:
        urls.append(SCIDB_BASE + meta["file_id"])
    if source in {"ut", "auto"}:
        urls.extend(f"{base}/{name}" for base in UT_BASE_URLS)
    return urls

def main() -> int:
    ap = argparse.ArgumentParser(description="Download public TEXBAT files into the VM-local dataset staging area. Raw .bin files are git-ignored.")
    ap.add_argument("--root", default="data/external/texbat")
    ap.add_argument("--files", nargs="*", default=["readme.txt", "texbat_ds7_and_ds8.pdf"], choices=sorted(FILES))
    ap.add_argument("--source", choices=["auto", "scidb", "ut"], default="scidb")
    ap.add_argument("--connect-timeout", type=int, default=30)
    ap.add_argument("--max-time", type=int, default=0, help="curl max time per file; 0 disables limit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    checksums = root / "checksums"
    manifests = root / "manifests"
    checksums.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    total_requested = sum(int(FILES[name]["size"]) for name in args.files)
    usage = shutil.disk_usage(root)
    print(f"root={root.resolve()}")
    print(f"requested_bytes={total_requested}")
    print(f"free_bytes={usage.free}")
    if total_requested > usage.free:
        print("WARNING requested files exceed current free space; download a smaller subset.")
    results = []
    for name in args.files:
        meta = FILES[name]
        out_dir = root / str(meta["subdir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / name
        expected_md5 = str(meta["md5"]).lower()
        if dst.exists() and dst.stat().st_size == int(meta["size"]):
            got_md5 = digest_file(dst, "md5").lower()
            status = "existing" if got_md5 == expected_md5 else "md5_mismatch"
            print(f"{status.upper()} {name} bytes={dst.stat().st_size} md5={got_md5}")
            results.append({"file": name, "status": status, "bytes": dst.stat().st_size, "md5": got_md5})
            continue
        if args.dry_run:
            print(f"DRY-RUN {name} -> {dst} size={meta['size']} md5={meta['md5']}")
            results.append({"file": name, "status": "dry_run", "bytes_expected": meta["size"], "md5_expected": meta["md5"]})
            continue
        ok = False
        last_output = ""
        for url in urls_for(name, args.source):
            print(f"DOWNLOAD {name} from {url}")
            proc = run_curl(url, dst, args.connect_timeout, args.max_time)
            last_output = proc.stdout[-4000:]
            part = dst.with_suffix(dst.suffix + ".part")
            if proc.returncode == 0 and part.exists() and part.stat().st_size > 0:
                part.rename(dst)
                got_md5 = digest_file(dst, "md5").lower()
                got_sha256 = digest_file(dst, "sha256")
                md5_ok = got_md5 == expected_md5
                (checksums / f"{name}.md5").write_text(f"{got_md5}  {name}\n", encoding="utf-8")
                (checksums / f"{name}.sha256").write_text(f"{got_sha256}  {name}\n", encoding="utf-8")
                print(f"DONE {name} bytes={dst.stat().st_size} md5={got_md5} md5_ok={md5_ok}")
                results.append({"file": name, "status": "downloaded" if md5_ok else "md5_mismatch", "source_url": url, "bytes": dst.stat().st_size, "md5": got_md5, "sha256": got_sha256})
                ok = md5_ok
                break
            print(f"FAILED {name} rc={proc.returncode}")
        if not ok:
            results.append({"file": name, "status": "failed", "last_output": last_output})
    report = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "source": args.source, "results": results}
    out = manifests / "download_status.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if all(r["status"] in {"existing", "downloaded", "dry_run"} for r in results) else 2

if __name__ == "__main__":
    sys.exit(main())
