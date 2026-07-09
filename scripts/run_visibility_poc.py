from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.pipeline import run_visibility_pipeline


if __name__ == "__main__":
    result = run_visibility_pipeline(
        config_path=REPO_ROOT / "configs" / "seoul_yesterday_13_15.yaml",
        output_root=REPO_ROOT / "artifacts",
    )
    print(f"output_dir={result.output_dir}")
    print(f"records_written={result.records_written}")
    print(f"visible_satellite_count={result.visible_satellite_count}")
