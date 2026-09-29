#!/usr/bin/env python3
"""Analyze one C++ runtime R0/R1 event-RAW replay without re-filtering it.

R0 is the raw PnP pose CSV. R1 is the temporal diagnostics CSV written by
``event_mocap_unified`` with ``--temporal_filter=true``.  The script treats
the R1 ``raw_*`` fields as the PnP measurement and the ``output_*`` fields as
the already-computed causal output; it never runs a second Python filter.

Without external ground truth, the report labels improvements as output
continuity, static-like repeatability, or high-frequency variation only.  It
does not report absolute pose accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-12


@dataclass
class Pose:
    timestamp_us: int
    position_mm: np.ndarray
    rvec: np.ndarray

    def matrix(self) -> np.ndarray:
        matrix, _ = cv2.Rodrigues(self.rvec.reshape(3, 1))
        return matrix


@dataclass
class TemporalRow:
    timestamp_us: int
    output_source: str
    raw_available: bool
    output_available: bool
    raw_pose: Optional[Pose]
    output_pose: Optional[Pose]
    translation_accepted: bool
    rotation_accepted: bool
    reset: bool
    reinitialized: bool
    dt_s: float
    translation_innovation_mm: float
    translation_mahalanobis2: float
    rotation_innovation_deg: float


def _bool(row: Dict[str, str], name: str) -> bool:
    value = row.get(name, "")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _float(row: Dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise ValueError(f"missing {name}")
    return float(value)


def _optional_pose(row: Dict[str, str], prefix: str, available: bool) -> Optional[Pose]:
    if not available:
        return None
    return Pose(
        timestamp_us=int(_float(row, "timestamp")),
        position_mm=np.asarray(
            [_float(row, f"{prefix}_x"), _float(row, f"{prefix}_y"), _float(row, f"{prefix}_z")],
            dtype=np.float64,
        ),
        rvec=np.asarray(
            [
                _float(row, f"{prefix}_rvec_x"),
                _float(row, f"{prefix}_rvec_y"),
                _float(row, f"{prefix}_rvec_z"),
            ],
            dtype=np.float64,
        ),
    )


def load_temporal_csv(path: Path) -> List[TemporalRow]:
    expected = {
        "timestamp",
        "output_source",
        "raw_measurement_available",
        "output_pose_available",
        "raw_x",
        "raw_rvec_x",
        "output_x",
        "output_rvec_x",
        "translation_accepted",
        "rotation_accepted",
        "reset",
        "reinitialized_from_measurement",
        "dt_s",
        "translation_innovation_mm",
        "translation_mahalanobis2",
        "rotation_innovation_deg",
    }
    rows: List[TemporalRow] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = expected - fields
        if missing:
            raise ValueError(f"{path} is missing temporal diagnostics columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                raw_available = _bool(row, "raw_measurement_available")
                output_available = _bool(row, "output_pose_available")
                rows.append(
                    TemporalRow(
                        timestamp_us=int(_float(row, "timestamp")),
                        output_source=row["output_source"],
                        raw_available=raw_available,
                        output_available=output_available,
                        raw_pose=_optional_pose(row, "raw", raw_available),
                        output_pose=_optional_pose(row, "output", output_available),
                        translation_accepted=_bool(row, "translation_accepted"),
                        rotation_accepted=_bool(row, "rotation_accepted"),
                        reset=_bool(row, "reset"),
                        reinitialized=_bool(row, "reinitialized_from_measurement"),
                        dt_s=_float(row, "dt_s"),
                        translation_innovation_mm=_float(row, "translation_innovation_mm"),
                        translation_mahalanobis2=_float(row, "translation_mahalanobis2"),
                        rotation_innovation_deg=_float(row, "rotation_innovation_deg"),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    if any(next_row.timestamp_us < row.timestamp_us for row, next_row in zip(rows, rows[1:])):
        raise ValueError(f"{path} timestamps must be non-decreasing")
    return rows


def load_raw_pose_csv(path: Path) -> List[Pose]:
    poses: List[Pose] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp", "x", "y", "z", "rvec_x", "rvec_y", "rvec_z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing raw pose columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = int(float(row["timestamp"]))
                poses.append(Pose(
                    timestamp_us=timestamp,
                    position_mm=np.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64),
                    rvec=np.asarray(
                        [float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])], dtype=np.float64
                    ),
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not poses:
        raise ValueError(f"{path} contains no pose rows")
    if any(next_pose.timestamp_us < pose.timestamp_us for pose, next_pose in zip(poses, poses[1:])):
        raise ValueError(f"{path} timestamps must be non-decreasing")
    return poses


def poses_to_arrays(poses: Sequence[Pose]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray([pose.timestamp_us for pose in poses], dtype=np.int64)
    positions = np.vstack([pose.position_mm for pose in poses])
    rvecs = np.vstack([pose.rvec for pose in poses])
    matrices = np.stack([pose.matrix() for pose in poses])
    return timestamps, positions, rvecs, matrices


def rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    rvec, _ = cv2.Rodrigues(relative)
    return float(np.linalg.norm(rvec.reshape(3)) * 180.0 / math.pi)


def exp_so3(tangent: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.asarray(tangent, dtype=np.float64).reshape(3, 1))
    return matrix


def log_so3(matrix: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(matrix)
    return rvec.reshape(3)


def rotation_medoid(matrices: np.ndarray) -> np.ndarray:
    if len(matrices) == 0:
        raise ValueError("cannot compute a rotation medoid of no matrices")
    # Exact O(n²) medoid is affordable for the 1.25-s static-like candidates.
    best_index = 0
    best_cost = math.inf
    for index, candidate in enumerate(matrices):
        cost = float(np.median([rotation_distance_deg(candidate, matrix) for matrix in matrices]))
        if cost < best_cost:
            best_index, best_cost = index, cost
    return matrices[best_index]


def distribution(values: Iterable[float], unit: str) -> Dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    if len(data) == 0:
        return {"samples": 0, "unit": unit}
    return {
        "samples": int(len(data)),
        "unit": unit,
        "rms": float(math.sqrt(np.mean(data * data))),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def repeatability_metrics(positions: np.ndarray, matrices: np.ndarray) -> Dict[str, Any]:
    center = np.median(positions, axis=0)
    radial = np.linalg.norm(positions - center, axis=1)
    reference_rotation = rotation_medoid(matrices)
    rotation_deviation = [rotation_distance_deg(reference_rotation, matrix) for matrix in matrices]
    return {
        "position_center_mm": [float(value) for value in center],
        "position_radial": distribution(radial, "mm"),
        "rotation_geodesic": distribution(rotation_deviation, "deg"),
    }


def local_linear_metrics(
    timestamps_us: np.ndarray,
    positions: np.ndarray,
    matrices: np.ndarray,
    max_gap_us: int,
) -> Dict[str, Any]:
    position_errors: List[float] = []
    rotation_errors: List[float] = []
    for index in range(1, len(timestamps_us) - 1):
        before = int(timestamps_us[index] - timestamps_us[index - 1])
        after = int(timestamps_us[index + 1] - timestamps_us[index])
        if before <= 0 or after <= 0 or before > max_gap_us or after > max_gap_us:
            continue
        weight = before / float(before + after)
        position_interpolated = (1.0 - weight) * positions[index - 1] + weight * positions[index + 1]
        position_errors.append(float(np.linalg.norm(positions[index] - position_interpolated)))
        relative = matrices[index - 1].T @ matrices[index + 1]
        rotation_interpolated = matrices[index - 1] @ exp_so3(weight * log_so3(relative))
        rotation_errors.append(rotation_distance_deg(rotation_interpolated, matrices[index]))
    return {
        "definition": "middle sample distance from timestamp-weighted endpoint interpolation",
        "max_adjacent_gap_us": int(max_gap_us),
        "position": distribution(position_errors, "mm"),
        "rotation": distribution(rotation_errors, "deg"),
    }


def local_linear_position_errors(
    timestamps_us: np.ndarray,
    positions: np.ndarray,
    max_gap_us: int,
) -> List[float]:
    errors: List[float] = []
    for index in range(1, len(timestamps_us) - 1):
        before = int(timestamps_us[index] - timestamps_us[index - 1])
        after = int(timestamps_us[index + 1] - timestamps_us[index])
        if before <= 0 or after <= 0 or before > max_gap_us or after > max_gap_us:
            continue
        weight = before / float(before + after)
        interpolated = (1.0 - weight) * positions[index - 1] + weight * positions[index + 1]
        errors.append(float(np.linalg.norm(positions[index] - interpolated)))
    return errors


def _linear_speed_mm_s(timestamps_us: np.ndarray, positions: np.ndarray) -> float:
    seconds = (timestamps_us - timestamps_us[0]).astype(np.float64) * 1e-6
    design = np.column_stack((np.ones_like(seconds), seconds))
    coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
    return float(np.linalg.norm(coefficients[1]))


def _bin_radial_p95(positions: np.ndarray) -> float:
    center = np.median(positions, axis=0)
    return float(np.percentile(np.linalg.norm(positions - center, axis=1), 95))


def find_static_like_candidates(
    timestamps_us: np.ndarray,
    positions: np.ndarray,
    matrices: np.ndarray,
    window_us: int,
    bin_us: int,
    max_gap_us: int,
    max_trend_speed_mm_s: float,
    max_rotation_speed_deg_s: float,
    max_bin_radial_p95_mm: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    origin = int(timestamps_us[0])
    stop = int(timestamps_us[-1] - window_us)
    for requested_start in range(origin, stop + 1, bin_us):
        requested_end = requested_start + window_us
        indices = np.flatnonzero((timestamps_us >= requested_start) & (timestamps_us <= requested_end))
        if len(indices) < 20:
            continue
        start, end = int(indices[0]), int(indices[-1])
        ts = timestamps_us[start : end + 1]
        if ts[-1] - ts[0] < int(0.98 * window_us) or np.max(np.diff(ts)) > max_gap_us:
            continue
        pos = positions[start : end + 1]
        mats = matrices[start : end + 1]
        bin_p95s: List[float] = []
        bins_valid = True
        for bin_start in range(requested_start, requested_end, bin_us):
            bin_indices = np.flatnonzero((ts >= bin_start) & (ts < bin_start + bin_us))
            if len(bin_indices) < 10:
                bins_valid = False
                break
            bin_p95s.append(_bin_radial_p95(pos[bin_indices]))
        if not bins_valid:
            continue
        duration_s = float((ts[-1] - ts[0]) * 1e-6)
        trend_speed = _linear_speed_mm_s(ts, pos)
        rotation_speed = rotation_distance_deg(mats[0], mats[-1]) / max(duration_s, EPS)
        maximum_bin_p95 = max(bin_p95s)
        passed = (
            trend_speed <= max_trend_speed_mm_s
            and rotation_speed <= max_rotation_speed_deg_s
            and maximum_bin_p95 <= max_bin_radial_p95_mm
        )
        candidates.append(
            {
                "start_timestamp_us": int(ts[0]),
                "end_timestamp_us": int(ts[-1]),
                "duration_s": duration_s,
                "samples": int(len(ts)),
                "max_timestamp_gap_us": int(np.max(np.diff(ts))),
                "linear_trend_speed_mm_s": trend_speed,
                "endpoint_rotation_speed_deg_s": rotation_speed,
                "max_bin_radial_p95_mm": maximum_bin_p95,
                "passed": passed,
            }
        )
    return candidates


def max_prediction_run(rows: Sequence[TemporalRow]) -> Dict[str, Any]:
    run_start: Optional[int] = None
    run_end: Optional[int] = None
    run_count = 0
    best: Tuple[int, int, int] = (0, 0, 0)
    for row in rows:
        if row.output_source == "prediction":
            if run_start is None:
                run_start = row.timestamp_us
            run_end = row.timestamp_us
            run_count += 1
            continue
        if run_start is not None and run_end is not None and run_count > best[0]:
            best = (run_count, run_start, run_end)
        run_start, run_end, run_count = None, None, 0
    if run_start is not None and run_end is not None and run_count > best[0]:
        best = (run_count, run_start, run_end)
    return {
        "frames": int(best[0]),
        "start_timestamp_us": int(best[1]) if best[0] else None,
        "end_timestamp_us": int(best[2]) if best[0] else None,
        "span_ms": float((best[2] - best[1]) * 1e-3) if best[0] else 0.0,
    }


def paired_raw_consistency(r0_poses: Sequence[Pose], rows: Sequence[TemporalRow]) -> Dict[str, Any]:
    r1_poses = [row.raw_pose for row in rows if row.raw_available and row.raw_pose is not None]
    timestamp_mismatches = 0
    numerical_mismatches = 0
    max_abs_difference = 0.0
    for r0_pose, r1_pose in zip(r0_poses, r1_poses):
        if r0_pose.timestamp_us != r1_pose.timestamp_us:
            timestamp_mismatches += 1
        difference = np.concatenate((r1_pose.position_mm - r0_pose.position_mm, r1_pose.rvec - r0_pose.rvec))
        max_abs_difference = max(max_abs_difference, float(np.max(np.abs(difference))))
        if not np.allclose(difference, 0.0, atol=1e-9, rtol=0.0):
            numerical_mismatches += 1
    r0_extra_rows = max(0, len(r0_poses) - len(r1_poses))
    r1_extra_rows = max(0, len(r1_poses) - len(r0_poses))
    return {
        "comparison_order": "stable row order; duplicate timestamps are allowed",
        "r0_pose_rows": int(len(r0_poses)),
        "r1_raw_measurement_rows": int(len(r1_poses)),
        "timestamp_mismatches": int(timestamp_mismatches),
        "numerical_mismatches": int(numerical_mismatches),
        "r0_extra_rows": int(r0_extra_rows),
        "r1_extra_rows": int(r1_extra_rows),
        "missing_from_r0": int(r1_extra_rows),
        "extra_in_r0": int(r0_extra_rows),
        "maximum_absolute_component_difference": max_abs_difference,
        "identical": r0_extra_rows == 0 and r1_extra_rows == 0 and timestamp_mismatches == 0 and numerical_mismatches == 0,
    }


def _percent_change(before: float, after: float) -> Optional[float]:
    if not math.isfinite(before) or abs(before) < EPS:
        return None
    return 100.0 * (after - before) / before


def static_pair_summary(
    candidate: Dict[str, Any],
    timestamps_us: np.ndarray,
    raw_positions: np.ndarray,
    raw_matrices: np.ndarray,
    output_positions: np.ndarray,
    output_matrices: np.ndarray,
) -> Dict[str, Any]:
    mask = (timestamps_us >= candidate["start_timestamp_us"]) & (timestamps_us <= candidate["end_timestamp_us"])
    raw_metrics = repeatability_metrics(raw_positions[mask], raw_matrices[mask])
    output_metrics = repeatability_metrics(output_positions[mask], output_matrices[mask])
    raw_position_rms = raw_metrics["position_radial"]["rms"]
    output_position_rms = output_metrics["position_radial"]["rms"]
    raw_rotation_rms = raw_metrics["rotation_geodesic"]["rms"]
    output_rotation_rms = output_metrics["rotation_geodesic"]["rms"]
    return {
        "candidate": candidate,
        "raw": raw_metrics,
        "filtered": output_metrics,
        "percent_change": {
            "position_radial_rms": _percent_change(raw_position_rms, output_position_rms),
            "position_radial_p95": _percent_change(
                raw_metrics["position_radial"]["p95"], output_metrics["position_radial"]["p95"]
            ),
            "rotation_geodesic_rms": _percent_change(raw_rotation_rms, output_rotation_rms),
            "rotation_geodesic_p95": _percent_change(
                raw_metrics["rotation_geodesic"]["p95"], output_metrics["rotation_geodesic"]["p95"]
            ),
        },
    }


def plot_report(
    output_path: Path,
    timestamps_us: np.ndarray,
    raw_positions: np.ndarray,
    output_positions: np.ndarray,
    local_raw_mm: Sequence[float],
    local_output_mm: Sequence[float],
    selected: Optional[Dict[str, Any]],
    source_counts: Counter,
) -> None:
    seconds = (timestamps_us - timestamps_us[0]) * 1e-6
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ("#1b9e77", "#d95f02", "#7570b3")
    labels = ("x", "y", "z")
    for axis, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(seconds, raw_positions[:, axis], color=color, alpha=0.22, lw=0.7)
        axes[0, 0].plot(seconds, output_positions[:, axis], color=color, alpha=0.9, lw=1.0, label=f"filtered {label}")
    if selected:
        start = (selected["start_timestamp_us"] - timestamps_us[0]) * 1e-6
        end = (selected["end_timestamp_us"] - timestamps_us[0]) * 1e-6
        axes[0, 0].axvspan(start, end, color="#e6ab02", alpha=0.18, label="static-like candidate")
    axes[0, 0].set_title("Paired raw PnP and filtered position")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("camera centre [mm]")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    if selected:
        mask = (timestamps_us >= selected["start_timestamp_us"]) & (timestamps_us <= selected["end_timestamp_us"])
        local_seconds = seconds[mask]
        for axis, (label, color) in enumerate(zip(labels, colors)):
            axes[0, 1].plot(local_seconds, raw_positions[mask, axis], color=color, alpha=0.35, lw=0.8)
            axes[0, 1].plot(local_seconds, output_positions[mask, axis], color=color, lw=1.1, label=label)
        axes[0, 1].set_title("Static-like candidate: output position")
        axes[0, 1].set_xlabel("time [s]")
        axes[0, 1].set_ylabel("camera centre [mm]")
        axes[0, 1].legend(ncol=3, fontsize=8)
        axes[0, 1].grid(alpha=0.25)
    else:
        axes[0, 1].text(0.5, 0.5, "No static-like candidate passed the configured criteria", ha="center", va="center")
        axes[0, 1].set_axis_off()

    bins = np.geomspace(1e-3, max(max(local_raw_mm, default=1.0), max(local_output_mm, default=1.0)), 60)
    axes[1, 0].hist(local_raw_mm, bins=bins, density=True, alpha=0.55, label="raw PnP")
    axes[1, 0].hist(local_output_mm, bins=bins, density=True, alpha=0.55, label="filtered")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Local-linear position residual")
    axes[1, 0].set_xlabel("residual [mm]")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    names = ("filtered_measurement", "prediction", "none")
    values = [source_counts.get(name, 0) for name in names]
    axes[1, 1].bar(names, values, color=("#1b9e77", "#7570b3", "#d95f02"))
    axes[1, 1].set_title("Runtime output source")
    axes[1, 1].set_ylabel("callback frames")
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    continuity = summary["continuity"]
    consistency = summary["raw_pnp_pairing"]
    dynamic = summary["dynamic_local_linear"]
    lines = ["# Runtime RAW R0/R1 분석", ""]
    lines += [f"- R0 raw pose: `{summary['raw_pose_csv']}`", f"- R1 temporal diagnostics: `{summary['temporal_csv']}`", ""]
    lines += ["## R0/R1 pairing", ""]
    lines += [
        f"- R0 raw pose {consistency['r0_pose_rows']:,}개와 R1 raw measurement "
        f"{consistency['r1_raw_measurement_rows']:,}개를 timestamp별로 대조했다.",
        f"- numerical mismatch: {consistency['numerical_mismatches']:,}, missing/extra: "
        f"{consistency['missing_from_r0']:,}/{consistency['extra_in_r0']:,}.",
        "",
    ]
    lines += ["## Continuity", ""]
    lines += [
        "| callback | raw measurement | output pose | prediction | no output |",
        "|---:|---:|---:|---:|---:|",
        f"| {continuity['callbacks']:,} | {continuity['raw_measurements']:,} | "
        f"{continuity['output_poses']:,} | {continuity['output_source_counts'].get('prediction', 0):,} | "
        f"{continuity['output_source_counts'].get('none', 0):,} |",
        "",
        "prediction은 raw PnP measurement가 아니라 bounded causal model output이므로 jitter/accuracy 표에서 제외한다.",
        "",
    ]
    lines += ["## Dynamic high-frequency variation", ""]
    lines += [
        "동일 timestamp의 raw-measurement/filtered-measurement 쌍에서만, 가운데 pose와 timestamp 보간 endpoint 사이의 거리를 계산했다.",
        "",
        "| method | position RMS (mm) | position P95 (mm) | rotation RMS (deg) | rotation P95 (deg) |",
        "|---|---:|---:|---:|---:|",
        f"| raw PnP | {dynamic['raw']['position']['rms']:.4f} | {dynamic['raw']['position']['p95']:.4f} | "
        f"{dynamic['raw']['rotation']['rms']:.4f} | {dynamic['raw']['rotation']['p95']:.4f} |",
        f"| filtered | {dynamic['filtered']['position']['rms']:.4f} | {dynamic['filtered']['position']['p95']:.4f} | "
        f"{dynamic['filtered']['rotation']['rms']:.4f} | {dynamic['filtered']['rotation']['p95']:.4f} |",
        "",
    ]
    static_result = summary.get("static_like_primary")
    if static_result:
        candidate = static_result["candidate"]
        lines += ["## Static-like interval", ""]
        lines += [
            f"선정 구간: `{candidate['start_timestamp_us']}–{candidate['end_timestamp_us']} us`, "
            f"{candidate['samples']:,} paired samples, {candidate['duration_s']:.3f} s.",
            "선정은 raw PnP 측정만으로 한 사전 규칙 기반 screen이며, 외부 기준으로 정지 상태를 증명한 것은 아니다.",
            "",
            "| method | position radial RMS (mm) | position radial P95 (mm) | SO(3) RMS (deg) | SO(3) P95 (deg) |",
            "|---|---:|---:|---:|---:|",
            f"| raw PnP | {static_result['raw']['position_radial']['rms']:.4f} | "
            f"{static_result['raw']['position_radial']['p95']:.4f} | "
            f"{static_result['raw']['rotation_geodesic']['rms']:.4f} | "
            f"{static_result['raw']['rotation_geodesic']['p95']:.4f} |",
            f"| filtered | {static_result['filtered']['position_radial']['rms']:.4f} | "
            f"{static_result['filtered']['position_radial']['p95']:.4f} | "
            f"{static_result['filtered']['rotation_geodesic']['rms']:.4f} | "
            f"{static_result['filtered']['rotation_geodesic']['p95']:.4f} |",
            "",
        ]
    else:
        lines += ["## Static-like interval", "", "Configured selection criteria를 만족하는 구간이 없어 static repeatability를 보고하지 않는다.", ""]
    lines += [
        "## 해석 한계",
        "",
        "외부 6-DoF ground truth가 없으므로 이 보고서는 absolute position/orientation accuracy, latency, 또는 물리적 노이즈 원인을 주장하지 않는다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pose-csv", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-window-s", type=float, default=1.25)
    parser.add_argument("--static-bin-s", type=float, default=0.25)
    parser.add_argument("--static-max-trend-speed-mm-s", type=float, default=50.0)
    parser.add_argument("--static-max-rotation-speed-deg-s", type=float, default=3.0)
    parser.add_argument("--static-max-bin-radial-p95-mm", type=float, default=30.0)
    parser.add_argument("--max-local-gap-ms", type=float, default=10.0)
    args = parser.parse_args()

    rows = load_temporal_csv(args.temporal_csv)
    raw_csv = load_raw_pose_csv(args.raw_pose_csv)
    source_counts: Counter = Counter(row.output_source for row in rows)
    consistency = paired_raw_consistency(raw_csv, rows)
    if not consistency["identical"]:
        raise ValueError("R0 raw pose CSV and R1 raw-measurement fields are not identical; comparison is invalid")
    paired_rows = [
        row
        for row in rows
        if row.raw_available
        and row.output_available
        and row.output_source == "filtered_measurement"
        and row.raw_pose is not None
        and row.output_pose is not None
    ]
    if len(paired_rows) < 3:
        raise ValueError("fewer than three paired raw/filter measurements")
    raw_poses = [row.raw_pose for row in paired_rows if row.raw_pose is not None]
    output_poses = [row.output_pose for row in paired_rows if row.output_pose is not None]
    timestamps, raw_positions, _, raw_matrices = poses_to_arrays(raw_poses)
    output_timestamps, output_positions, _, output_matrices = poses_to_arrays(output_poses)
    if not np.array_equal(timestamps, output_timestamps):
        raise ValueError("paired raw and filtered timestamps differ")

    max_gap_us = int(round(args.max_local_gap_ms * 1000.0))
    dynamic_raw = local_linear_metrics(timestamps, raw_positions, raw_matrices, max_gap_us)
    dynamic_filtered = local_linear_metrics(timestamps, output_positions, output_matrices, max_gap_us)
    raw_plot_errors = local_linear_position_errors(timestamps, raw_positions, max_gap_us)
    filtered_plot_errors = local_linear_position_errors(timestamps, output_positions, max_gap_us)
    raw_to_output = distribution(np.linalg.norm(raw_positions - output_positions, axis=1), "mm")
    raw_to_output_rotation = distribution(
        [rotation_distance_deg(raw_matrix, output_matrix) for raw_matrix, output_matrix in zip(raw_matrices, output_matrices)],
        "deg",
    )
    candidates = find_static_like_candidates(
        timestamps,
        raw_positions,
        raw_matrices,
        window_us=int(round(args.static_window_s * 1e6)),
        bin_us=int(round(args.static_bin_s * 1e6)),
        max_gap_us=max_gap_us,
        max_trend_speed_mm_s=args.static_max_trend_speed_mm_s,
        max_rotation_speed_deg_s=args.static_max_rotation_speed_deg_s,
        max_bin_radial_p95_mm=args.static_max_bin_radial_p95_mm,
    )
    passing = [candidate for candidate in candidates if candidate["passed"]]
    primary = passing[0] if passing else None
    static_summary = (
        static_pair_summary(primary, timestamps, raw_positions, raw_matrices, output_positions, output_matrices)
        if primary
        else None
    )

    raw_measurement_rows = [row for row in rows if row.raw_available]
    continuity = {
        "callbacks": int(len(rows)),
        "raw_measurements": int(len(raw_measurement_rows)),
        "output_poses": int(sum(row.output_available for row in rows)),
        "output_source_counts": {name: int(count) for name, count in sorted(source_counts.items())},
        "translation_accepted": int(sum(row.translation_accepted for row in raw_measurement_rows)),
        "translation_rejected": int(sum(not row.translation_accepted for row in raw_measurement_rows)),
        "rotation_accepted": int(sum(row.rotation_accepted for row in raw_measurement_rows)),
        "rotation_rejected": int(sum(not row.rotation_accepted for row in raw_measurement_rows)),
        "reset_events": int(sum(row.reset for row in rows)),
        "reinitialization_events": int(sum(row.reinitialized for row in rows)),
        "max_prediction_run": max_prediction_run(rows),
    }
    summary: Dict[str, Any] = {
        "raw_pose_csv": str(args.raw_pose_csv),
        "temporal_csv": str(args.temporal_csv),
        "pairing_policy": "stable row order plus timestamp equality; paired metrics require raw_measurement_available=1, output_source=filtered_measurement, and output_pose_available=1",
        "raw_pnp_pairing": consistency,
        "continuity": continuity,
        "dynamic_local_linear": {"raw": dynamic_raw, "filtered": dynamic_filtered},
        "raw_to_filtered_position_change": raw_to_output,
        "raw_to_filtered_rotation_change": raw_to_output_rotation,
        "static_like_selection": {
            "window_s": args.static_window_s,
            "bin_s": args.static_bin_s,
            "max_trend_speed_mm_s": args.static_max_trend_speed_mm_s,
            "max_rotation_speed_deg_s": args.static_max_rotation_speed_deg_s,
            "max_bin_radial_p95_mm": args.static_max_bin_radial_p95_mm,
            "candidates_passing": int(len(passing)),
            "candidates": candidates,
        },
        "static_like_primary": static_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "summary.md", summary)
    plot_report(
        args.output_dir / "runtime_pair_comparison.png",
        timestamps,
        raw_positions,
        output_positions,
        raw_plot_errors,
        filtered_plot_errors,
        primary,
        source_counts,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
