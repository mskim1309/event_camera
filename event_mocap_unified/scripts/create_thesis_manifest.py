#!/usr/bin/env python3
"""Create a provenance manifest for the cited thesis experiments.

The historical data lack a capture manifest.  This script at least makes the
new simulation/post-processing claims independently auditable: every cited
input, configuration, script, and summary has a SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def record(path: Path, root: Path) -> Dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def existing(paths: Iterable[Path]) -> Iterable[Path]:
    return (path for path in paths if path.exists())


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    results = project_root / "thesis_results"
    output = results / "experiment_manifest.json"

    inputs = [
        workspace_root / "기시설1_2021-19515_김민성_3차제출.docx",
        project_root / "config/marker/multi_led_marker.json",
        project_root / "config/marker/multi_led_v2.json",
        project_root / "config/calibration_1017.json",
        workspace_root / "event_mocap/final/pose/pose_log_1119_stop_camera.csv",
        workspace_root / "event_mocap/final/pose/pose_log_1119_stop_camera_bundle.csv",
        workspace_root / "event_mocap/final/pose/pose_log_1119_rotate_camera_bundle.csv",
    ]
    scripts = [
        project_root / "scripts/evaluate_temporal_pose_filter.py",
        project_root / "scripts/run_thesis_pose_experiments.py",
        project_root / "scripts/run_thesis_experiments.py",
        project_root / "scripts/characterize_stationary_pose_noise.py",
        project_root / "scripts/make_thesis_extension_docx.py",
    ]
    summaries = [
        results / "final_pixel_noise_05/summary.json",
        results / "final_timestamp_jitter_1000/summary.json",
        results / "final_combined_realistic/summary.json",
        results / "real_1119_stop_baseline/summary.json",
        results / "real_1119_stop_bundle/summary.json",
        results / "noise_1119_stop_baseline/summary.json",
        results / "noise_1119_stop_bundle/summary.json",
    ]
    document_outputs = [
        results / "논문_4차_추가_초안.md",
        results / "논문_4차_실측노이즈_보충.md",
        results / "기시설1_2021-19515_김민성_4차_실측노이즈포함.docx",
    ]
    manifest = {
        "purpose": "Provenance for event-camera pose-filter thesis experiments",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(project_root),
        "inputs": [record(path, workspace_root) for path in existing(inputs)],
        "scripts": [record(path, workspace_root) for path in existing(scripts)],
        "cited_result_summaries": [record(path, workspace_root) for path in existing(summaries)],
        "thesis_outputs": [record(path, workspace_root) for path in existing(document_outputs)],
        "known_limitations": [
            "Historical pose CSVs lack per-LED 2D observations, tracker status, reprojection residuals, and capture manifests.",
            "No external 6-DoF ground truth is retained for historical dynamic logs.",
            "Current workspace build lacks MetavisionSDK; RAW replay is not part of this manifest.",
        ],
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
