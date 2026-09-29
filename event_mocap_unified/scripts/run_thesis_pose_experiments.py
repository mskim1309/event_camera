#!/usr/bin/env python3
"""Production-matched synthetic experiment for the active-marker thesis.

Unlike a purely framewise PnP comparison, this script reproduces the current
production estimator's important behaviour: after the first observation, the
previous *raw* PnP pose is supplied as the iterative PnP initial estimate.
The proposed robust temporal filter is then evaluated as a separate causal
post-estimation layer.  Keeping those two state mechanisms separate makes the
comparison fair: both methods receive exactly the same noisy 2D LED points.

It imports numerical/CSV/plotting helpers from evaluate_temporal_pose_filter.py.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import evaluate_temporal_pose_filter as base


def simulate_with_production_warm_start(args: argparse.Namespace) -> Tuple[base.PoseSeries, base.PoseSeries, Dict[str, Any]]:
    """Project asynchronous noisy LED observations and recover them like production PnP."""

    marker, K, distortion = base._load_marker_and_camera(args.marker_json, args.calibration_json)
    rng = np.random.default_rng(args.seed)
    sample_count = int(round(args.duration_s * args.rate_hz))
    timestamps_us = np.arange(sample_count, dtype=np.float64) * (1e6 / args.rate_hz)

    gt_position = np.zeros((sample_count, 3), dtype=np.float64)
    gt_rvec = np.zeros((sample_count, 3), dtype=np.float64)
    estimated_position: List[np.ndarray] = []
    estimated_rvec: List[np.ndarray] = []
    per_led_correlated_noise = np.zeros((len(marker), 2), dtype=np.float64)
    previous_rvec: Optional[np.ndarray] = None
    previous_tvec: Optional[np.ndarray] = None
    ransac_failures = 0
    pnp_failures = 0

    for frame_index, timestamp_us in enumerate(timestamps_us):
        nominal_time_s = timestamp_us * 1e-6
        center_gt, rvec_gt, _ = base._look_at_pose(nominal_time_s, args.trajectory)
        gt_position[frame_index] = center_gt
        gt_rvec[frame_index] = rvec_gt

        image_points = []
        for led_index, object_point in enumerate(marker):
            # Each LED is physically observed at a different time, which is
            # the timestamp-mismatch mechanism studied in the thesis.
            observation_time_s = nominal_time_s + rng.uniform(
                -args.timestamp_jitter_us, args.timestamp_jitter_us
            ) * 1e-6
            observation_time_s = float(np.clip(observation_time_s, 0.0, args.duration_s))
            _, rvec_obs, tvec_obs = base._look_at_pose(observation_time_s, args.trajectory)
            projected, _ = cv2.projectPoints(
                object_point.reshape(1, 3),
                rvec_obs.reshape(3, 1),
                tvec_obs.reshape(3, 1),
                K,
                distortion,
            )

            rho = float(np.clip(args.correlated_noise_rho, 0.0, 0.9999))
            per_led_correlated_noise[led_index] = (
                rho * per_led_correlated_noise[led_index]
                + math.sqrt(1.0 - rho * rho) * rng.normal(0.0, args.correlated_noise_px, size=2)
            )
            perturbation = rng.normal(0.0, args.pixel_noise_px, size=2) + per_led_correlated_noise[led_index]
            if rng.random() < args.outlier_probability:
                perturbation += rng.normal(0.0, args.outlier_std_px, size=2)
            image_points.append(projected.reshape(2) + perturbation)

        image_points_np = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
        ransac_ok, ransac_rvec, ransac_tvec, _ = cv2.solvePnPRansac(
            marker,
            image_points_np,
            K,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=100,
            reprojectionError=args.ransac_threshold_px,
            confidence=0.99,
        )
        if not ransac_ok:
            ransac_failures += 1

        if previous_rvec is None or previous_tvec is None:
            # Current unified PoseEstimator subsequently executes ITERATIVE
            # PnP without a guess on the first frame, even after RANSAC.
            pnp_ok, rvec, tvec = cv2.solvePnP(marker, image_points_np, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            # This is the production warm start, not the proposed filter.
            pnp_ok, rvec, tvec = cv2.solvePnP(
                marker,
                image_points_np,
                K,
                distortion,
                previous_rvec.reshape(3, 1).copy(),
                previous_tvec.reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not pnp_ok:
            pnp_failures += 1
            if estimated_position:
                estimated_position.append(estimated_position[-1].copy())
                estimated_rvec.append(estimated_rvec[-1].copy())
            else:
                estimated_position.append(center_gt.copy())
                estimated_rvec.append(rvec_gt.copy())
            continue

        R_wc = base.rotvec_to_matrix(rvec.reshape(3))
        camera_center = -R_wc.T @ tvec.reshape(3)
        estimated_position.append(camera_center)
        estimated_rvec.append(rvec.reshape(3))
        previous_rvec = rvec.reshape(3).copy()
        previous_tvec = tvec.reshape(3).copy()

    raw = base.PoseSeries(timestamps_us, np.asarray(estimated_position), np.asarray(estimated_rvec))
    ground_truth = base.PoseSeries(timestamps_us, gt_position, gt_rvec)
    metadata: Dict[str, Any] = {
        "estimator": "iterative PnP with previous raw-pose initialisation",
        "ransac_failures": ransac_failures,
        "pnp_failures": pnp_failures,
        "marker_count": int(len(marker)),
        "simulation": {
            "trajectory": args.trajectory,
            "duration_s": args.duration_s,
            "rate_hz": args.rate_hz,
            "pixel_noise_px": args.pixel_noise_px,
            "timestamp_jitter_us": args.timestamp_jitter_us,
            "correlated_noise_px": args.correlated_noise_px,
            "correlated_noise_rho": args.correlated_noise_rho,
            "outlier_probability": args.outlier_probability,
            "outlier_std_px": args.outlier_std_px,
            "seed": args.seed,
        },
    }
    return raw, ground_truth, metadata


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, ground_truth, metadata = simulate_with_production_warm_start(args)
    filter_config = base._filter_config_from_args(args)
    filtered, diagnostics = base.filter_series(raw, filter_config)
    summary: Dict[str, Any] = {
        "experiment_type": "production_matched_simulation",
        "metadata": metadata,
        "filter_config": asdict(filter_config),
        "metrics": {
            "production_warmstart_pnp": base.ground_truth_metrics(raw, ground_truth, args.outlier_position_threshold_mm),
            "robust_temporal_filter": base.ground_truth_metrics(filtered, ground_truth, args.outlier_position_threshold_mm),
        },
        "filter_diagnostics": {
            "translation_rejection_rate_percent": float(100.0 * np.mean([not step.translation_accepted for step in diagnostics])),
            "rotation_rejection_rate_percent": float(100.0 * np.mean([not step.rotation_accepted for step in diagnostics])),
        },
    }
    raw_diagnostics = [base.FilterStep(p, rv, True, True, 0.0, 0.0) for p, rv in zip(raw.position_mm, raw.rvec)]
    base.write_pose_csv(args.output_dir / "production_warmstart_pnp.csv", raw, raw_diagnostics)
    base._save_ground_truth_csv(args.output_dir / "ground_truth.csv", ground_truth)
    base.write_pose_csv(args.output_dir / "robust_temporal_filter.csv", filtered, diagnostics)
    base.write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(base._metric_markdown(summary), encoding="utf-8")
    base._plot_simulation(args.output_dir, ground_truth, raw, filtered)
    print(json.dumps(base._json_ready(summary["metrics"]), ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marker-json", type=Path, default=Path("config/marker/multi_led_marker.json"))
    parser.add_argument("--calibration-json", type=Path, default=Path("config/calibration_1017.json"))
    parser.add_argument("--trajectory", choices=("circular", "linear", "sinusoidal"), default="sinusoidal")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--pixel-noise-px", type=float, default=0.5)
    parser.add_argument("--timestamp-jitter-us", type=float, default=1000.0)
    parser.add_argument("--correlated-noise-px", type=float, default=0.0)
    parser.add_argument("--correlated-noise-rho", type=float, default=0.85)
    parser.add_argument("--outlier-probability", type=float, default=0.0)
    parser.add_argument("--outlier-std-px", type=float, default=8.0)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--outlier-position-threshold-mm", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20260929)
    base.add_filter_arguments(parser)
    # Safe defaults for the SO(3) low-pass update.  A small hard gate caused
    # drift after planar-PnP ambiguities in the validation sweep.
    parser.set_defaults(orientation_gate_deg=180.0, orientation_hard_gate_deg=360.0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.marker_json = args.marker_json.resolve()
    args.calibration_json = args.calibration_json.resolve()
    run(args)


if __name__ == "__main__":
    main()
