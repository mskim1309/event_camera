#!/usr/bin/env python3
"""Synthetic, known-ground-truth comparison of OpenCV PnP-RANSAC solvers.

This experiment is deliberately independent of camera playback: it projects the
same marker geometry and calibration used by the tracker, adds repeatable image
noise, and checks the pose returned by solvePnPRansac against the supplied
ground truth.  It is useful for detecting a solver/initialization failure that
would otherwise be mistaken for temporal jitter.

Example (the marker JSON must be the configuration used for the recording):
  python3 scripts/benchmark_pnp_solver_stability.py \
    --calibration config/calibration_1017.json --marker-json config/6marker.json \
    --translation-mm 0,0,1500 --rotation-rvec 0,0,0 --noise-px 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


SOLVER_FLAGS = {
    "iterative": "SOLVEPNP_ITERATIVE",
    "epnp": "SOLVEPNP_EPNP",
    "sqpnp": "SOLVEPNP_SQPNP",
    "p3p": "SOLVEPNP_P3P",
    "ap3p": "SOLVEPNP_AP3P",
}


def comma_vector(value: str) -> np.ndarray:
    try:
        result = np.asarray([float(component) for component in value.split(",")], dtype=np.float64)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be three comma-separated numbers") from error
    if result.shape != (3,):
        raise argparse.ArgumentTypeError("must contain exactly three numbers")
    return result.reshape(3, 1)


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if "K" not in config or "D" not in config:
        raise ValueError(f"{path}: expected calibration keys 'K' and 'D'")
    camera_matrix = np.asarray(config["K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(config["D"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, distortion


def load_marker_points(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    entries = config.get("active marker")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected an 'active marker' list")

    points: list[list[float]] = []
    for entry in entries:
        # The original tracking JSON stores each point inside a `led` object.
        # Supporting a flat entry keeps the benchmark usable with exported
        # configuration variants without changing the actual point values.
        led = entry.get("led", entry)
        xyz = led.get("xyz")
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError(f"{path}: every marker LED must have a three-value xyz")
        points.append(xyz)
    if len(points) < 4:
        raise ValueError(f"{path}: at least four marker points are required")
    return np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)


def rotation_error_degrees(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    r_estimate, _ = cv2.Rodrigues(estimate)
    r_ground_truth, _ = cv2.Rodrigues(ground_truth)
    cosine = (np.trace(r_estimate @ r_ground_truth.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values), level)) if values else float("nan")


def run_solver(
    solver_name: str,
    object_points: np.ndarray,
    ideal_image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    ground_truth_rvec: np.ndarray,
    ground_truth_tvec: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float | int | str]:
    constant_name = SOLVER_FLAGS[solver_name]
    if not hasattr(cv2, constant_name):
        return {"solver": solver_name, "status": f"unavailable ({constant_name})"}
    flag = getattr(cv2, constant_name)
    rng = np.random.default_rng(seed)
    attempted = returned = invalid_consensus = 0
    max_errors: list[float] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []

    for _ in range(args.trials):
        attempted += 1
        noisy_points = ideal_image_points + rng.normal(0.0, args.noise_px, ideal_image_points.shape)
        try:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                object_points,
                noisy_points,
                camera_matrix,
                distortion,
                iterationsCount=args.iterations,
                reprojectionError=args.reprojection_threshold_px,
                confidence=args.confidence,
                flags=flag,
            )
        except cv2.error:
            continue
        if not ok or rvec is None or tvec is None:
            continue
        returned += 1
        reprojection, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        residuals = np.linalg.norm((reprojection - noisy_points).reshape(-1, 2), axis=1)
        max_error = float(np.max(residuals))
        max_errors.append(max_error)
        if not math.isfinite(max_error) or max_error > args.reprojection_threshold_px:
            invalid_consensus += 1
        position_errors.append(float(np.linalg.norm(tvec - ground_truth_tvec)))
        orientation_errors.append(rotation_error_degrees(rvec, ground_truth_rvec))

    return {
        "solver": solver_name,
        "status": "ok",
        "returned": returned,
        "attempted": attempted,
        "invalid_consensus": invalid_consensus,
        "median_max_residual_px": percentile(max_errors, 50),
        "p95_max_residual_px": percentile(max_errors, 95),
        "median_position_error_mm": percentile(position_errors, 50),
        "p95_position_error_mm": percentile(position_errors, 95),
        "median_rotation_error_deg": percentile(orientation_errors, 50),
        "p95_rotation_error_deg": percentile(orientation_errors, 95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--marker-json", type=Path, required=True)
    parser.add_argument("--translation-mm", type=comma_vector, default=comma_vector("0,0,1500"))
    parser.add_argument("--rotation-rvec", type=comma_vector, default=comma_vector("0,0,0"))
    parser.add_argument("--noise-px", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--reprojection-threshold-px", type=float, default=4.0)
    parser.add_argument("--solvers", default="iterative,epnp,sqpnp")
    args = parser.parse_args()

    if args.trials <= 0 or args.noise_px < 0.0 or args.iterations <= 0:
        parser.error("trials and iterations must be positive; noise-px must be non-negative")
    requested_solvers = [name.strip().lower() for name in args.solvers.split(",") if name.strip()]
    unknown = [name for name in requested_solvers if name not in SOLVER_FLAGS]
    if unknown:
        parser.error(f"unknown solver(s): {', '.join(unknown)}")

    try:
        camera_matrix, distortion = load_intrinsics(args.calibration)
        object_points = load_marker_points(args.marker_json)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2

    ideal_image_points, _ = cv2.projectPoints(
        object_points, args.rotation_rvec, args.translation_mm, camera_matrix, distortion
    )
    centered = object_points.reshape(-1, 3) - object_points.reshape(-1, 3).mean(axis=0)
    geometry_rank = int(np.linalg.matrix_rank(centered))
    print(
        f"opencv={cv2.__version__} points={len(object_points)} geometry_rank={geometry_rank} "
        f"noise_px={args.noise_px:g} trials={args.trials}"
    )
    print(
        "solver,status,returned/attempted,invalid_consensus,median_max_residual_px,"
        "p95_max_residual_px,median_position_error_mm,p95_position_error_mm,"
        "median_rotation_error_deg,p95_rotation_error_deg"
    )
    for solver in requested_solvers:
        report = run_solver(
            solver,
            object_points,
            ideal_image_points,
            camera_matrix,
            distortion,
            args.rotation_rvec,
            args.translation_mm,
            args,
            args.seed,
        )
        if report["status"] != "ok":
            print(f"{solver},{report['status']},-,-,-,-,-,-,-,-")
            continue
        print(
            f"{report['solver']},ok,{report['returned']}/{report['attempted']},"
            f"{report['invalid_consensus']},"
            f"{report['median_max_residual_px']:.6g},{report['p95_max_residual_px']:.6g},"
            f"{report['median_position_error_mm']:.6g},{report['p95_position_error_mm']:.6g},"
            f"{report['median_rotation_error_deg']:.6g},{report['p95_rotation_error_deg']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
