#!/usr/bin/env python3
"""Reproducible pose-filter experiments for the event-marker thesis.

This tool deliberately works on CSV pose logs as well as a synthetic 2D
marker-projection experiment.  It makes two validation claims separable:

* ``simulate`` has known 6-DoF ground truth and therefore reports accuracy.
* ``real`` never invents unavailable ground truth.  For stationary recordings
  it reports pose precision (jitter about a fixed pose); for moving recordings
  it reports high-frequency trajectory variation only.

The proposed online filter is a robust constant-velocity Kalman filter for
the camera centre plus an SO(3) complementary filter for rotation.  It is
causal: no future sample is used.  Large innovations are down-weighted or
rejected instead of being copied into the output trajectory.

Only Python's standard library, NumPy, OpenCV, and Matplotlib are required;
Pandas is intentionally not required so the script runs in the lab setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


@dataclass
class PoseSeries:
    """A pose time series in the convention stored by the input CSV.

    ``rvec`` is treated as an arbitrary, internally consistent Rodrigues
    convention.  The temporal SO(3) filter only uses relative rotations, so
    this also works for old logs that store camera-to-world rather than
    world-to-camera rotations.
    """

    timestamp_us: np.ndarray
    position_mm: np.ndarray
    rvec: np.ndarray

    def __len__(self) -> int:
        return int(self.timestamp_us.shape[0])

    def subset(self, mask: np.ndarray) -> "PoseSeries":
        return PoseSeries(self.timestamp_us[mask], self.position_mm[mask], self.rvec[mask])


@dataclass
class TemporalFilterConfig:
    """Tunable parameters of the online robust temporal filter.

    The values are in millimetres and seconds.  ``measurement_std_mm`` should
    be selected from a stationary validation sequence or a held-out simulator
    sweep, not tuned against a dynamic test trajectory.
    """

    process_accel_std_mm_s2: float = 2500.0
    measurement_std_mm: float = 45.0
    gate_chi2: float = 16.27  # chi-square(3), approximately 99.9 percentile
    hard_gate_chi2: float = 81.0
    orientation_alpha: float = 0.35
    orientation_beta: float = 0.025
    orientation_gate_deg: float = 20.0
    orientation_hard_gate_deg: float = 70.0
    reset_gap_s: float = 0.25


@dataclass
class FilterStep:
    position_mm: np.ndarray
    rvec: np.ndarray
    translation_accepted: bool
    rotation_accepted: bool
    innovation_mm: float
    innovation_deg: float


def _as_float(row: Dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            return float(row[name])
    raise KeyError(f"missing required column; expected one of {names}")


def _rotation_from_row(row: Dict[str, str]) -> np.ndarray:
    rvec_names = ("rvec_x", "rvec_y", "rvec_z")
    if all(name in row and row[name] not in ("", None) for name in rvec_names):
        return np.array([float(row[name]) for name in rvec_names], dtype=np.float64)

    matrix_names = ("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22")
    if all(name in row and row[name] not in ("", None) for name in matrix_names):
        matrix = np.array([float(row[name]) for name in matrix_names], dtype=np.float64).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(matrix)
        return rvec.reshape(3)

    raise KeyError("missing rvec_x/rvec_y/rvec_z or 3x3 rotation-matrix columns")


def load_pose_csv(path: Path) -> PoseSeries:
    """Load either unified/final pose CSV conventions without Pandas."""

    timestamps: List[float] = []
    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path} does not have a CSV header")
        for row_number, row in enumerate(reader, start=2):
            try:
                ts = _as_float(row, "timestamp_us", "timestamp", "ts")
                pos = np.array([_as_float(row, "x"), _as_float(row, "y"), _as_float(row, "z")], dtype=np.float64)
                rvec = _rotation_from_row(row)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: {exc}") from exc
            if np.isfinite(ts) and np.all(np.isfinite(pos)) and np.all(np.isfinite(rvec)):
                timestamps.append(ts)
                positions.append(pos)
                rotations.append(rvec)

    if not timestamps:
        raise ValueError(f"{path} contains no finite pose rows")
    order = np.argsort(np.asarray(timestamps, dtype=np.float64), kind="stable")
    return PoseSeries(
        np.asarray(timestamps, dtype=np.float64)[order],
        np.asarray(positions, dtype=np.float64)[order],
        np.asarray(rotations, dtype=np.float64)[order],
    )


def write_pose_csv(path: Path, series: PoseSeries, diagnostics: Sequence[FilterStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_us",
                "x",
                "y",
                "z",
                "rvec_x",
                "rvec_y",
                "rvec_z",
                "translation_accepted",
                "rotation_accepted",
                "translation_innovation_mm",
                "rotation_innovation_deg",
            ]
        )
        for ts, p, rv, diag in zip(series.timestamp_us, series.position_mm, series.rvec, diagnostics):
            writer.writerow(
                [
                    f"{ts:.0f}",
                    *[f"{value:.9g}" for value in p],
                    *[f"{value:.9g}" for value in rv],
                    int(diag.translation_accepted),
                    int(diag.rotation_accepted),
                    f"{diag.innovation_mm:.9g}",
                    f"{diag.innovation_deg:.9g}",
                ]
            )


def rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return matrix


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3)


def exp_so3(tangent: np.ndarray) -> np.ndarray:
    return rotvec_to_matrix(tangent)


def log_so3(matrix: np.ndarray) -> np.ndarray:
    return matrix_to_rotvec(matrix)


def rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(log_so3(left.T @ right)) * 180.0 / math.pi)


class RobustTemporalPoseFilter:
    """Causal SE(3) filter: CV Kalman translation + robust SO(3) update."""

    def __init__(self, config: TemporalFilterConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._last_ts_us = 0.0
        self._x = np.zeros(6, dtype=np.float64)  # [camera center, velocity]
        self._P = np.eye(6, dtype=np.float64)
        self._R = np.eye(3, dtype=np.float64)
        self._omega = np.zeros(3, dtype=np.float64)
        self._consecutive_rejections = 0

    def _initialize(self, ts_us: float, position: np.ndarray, rotation: np.ndarray) -> FilterStep:
        measurement_std = max(self.config.measurement_std_mm, 1e-3)
        self._initialized = True
        self._last_ts_us = ts_us
        self._x[:3] = position
        self._x[3:] = 0.0
        self._P = np.diag(
            [measurement_std**2] * 3
            + [max(10.0 * measurement_std, 100.0) ** 2] * 3
        )
        self._R = rotation
        self._omega[:] = 0.0
        self._consecutive_rejections = 0
        return FilterStep(position.copy(), matrix_to_rotvec(rotation), True, True, 0.0, 0.0)

    def update(self, timestamp_us: float, position_mm: np.ndarray, rvec: np.ndarray) -> FilterStep:
        measurement_rotation = rotvec_to_matrix(rvec)
        if not self._initialized:
            return self._initialize(timestamp_us, position_mm, measurement_rotation)

        dt = (timestamp_us - self._last_ts_us) * 1e-6
        if dt <= 0.0 or dt > self.config.reset_gap_s:
            return self._initialize(timestamp_us, position_mm, measurement_rotation)
        self._last_ts_us = timestamp_us

        # Constant-velocity Kalman prediction with white acceleration process noise.
        identity3 = np.eye(3, dtype=np.float64)
        F = np.block([[identity3, dt * identity3], [np.zeros((3, 3)), identity3]])
        acceleration_variance = max(self.config.process_accel_std_mm_s2, 1e-3) ** 2
        Q = acceleration_variance * np.block(
            [
                [(dt**4 / 4.0) * identity3, (dt**3 / 2.0) * identity3],
                [(dt**3 / 2.0) * identity3, (dt**2) * identity3],
            ]
        )
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._P = 0.5 * (self._P + self._P.T)

        H = np.zeros((3, 6), dtype=np.float64)
        H[:, :3] = identity3
        innovation = position_mm - H @ self._x
        measurement_variance = max(self.config.measurement_std_mm, 1e-3) ** 2
        Rm = measurement_variance * identity3
        S = H @ self._P @ H.T + Rm
        try:
            mahalanobis2 = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            mahalanobis2 = float("inf")
        innovation_norm = float(np.linalg.norm(innovation))

        translation_accepted = mahalanobis2 <= self.config.hard_gate_chi2
        if translation_accepted:
            # Huber-like down-weighting: large but plausible innovations get a
            # larger observation covariance instead of an abrupt hard clip.
            inflation = max(1.0, mahalanobis2 / max(self.config.gate_chi2, EPS))
            S_effective = H @ self._P @ H.T + inflation * Rm
            K = self._P @ H.T @ np.linalg.inv(S_effective)
            self._x = self._x + K @ innovation
            IKH = np.eye(6) - K @ H
            self._P = IKH @ self._P @ IKH.T + K @ (inflation * Rm) @ K.T
            self._P = 0.5 * (self._P + self._P.T)
            self._consecutive_rejections = 0
        else:
            self._consecutive_rejections += 1

        # In the same spirit, predict an angular velocity and use a robust
        # relative-rotation update.  This avoids averaging Rodrigues vectors
        # directly, which is invalid around the pi branch cut.
        predicted_rotation = self._R @ exp_so3(self._omega * dt)
        orientation_innovation = log_so3(predicted_rotation.T @ measurement_rotation)
        orientation_innovation_deg = float(np.linalg.norm(orientation_innovation) * 180.0 / math.pi)
        rotation_accepted = orientation_innovation_deg <= self.config.orientation_hard_gate_deg
        if rotation_accepted:
            attenuation = min(
                1.0,
                max(self.config.orientation_gate_deg, 1e-3) / max(orientation_innovation_deg, 1e-3),
            )
            alpha = self.config.orientation_alpha * attenuation
            beta = self.config.orientation_beta * attenuation
            self._R = predicted_rotation @ exp_so3(alpha * orientation_innovation)
            self._omega = self._omega + (beta / max(dt, 1e-4)) * orientation_innovation
        else:
            self._R = predicted_rotation

        # If a long burst was rejected, reset at the current measured pose. It
        # prevents the predictor from remaining detached after a real motion.
        if self._consecutive_rejections >= 4:
            return self._initialize(timestamp_us, position_mm, measurement_rotation)

        return FilterStep(
            self._x[:3].copy(),
            matrix_to_rotvec(self._R),
            translation_accepted,
            rotation_accepted,
            innovation_norm,
            orientation_innovation_deg,
        )


def filter_series(series: PoseSeries, config: TemporalFilterConfig) -> Tuple[PoseSeries, List[FilterStep]]:
    filter_ = RobustTemporalPoseFilter(config)
    steps = [filter_.update(ts, p, rv) for ts, p, rv in zip(series.timestamp_us, series.position_mm, series.rvec)]
    return (
        PoseSeries(
            series.timestamp_us.copy(),
            np.asarray([step.position_mm for step in steps], dtype=np.float64),
            np.asarray([step.rvec for step in steps], dtype=np.float64),
        ),
        steps,
    )


def _rotation_medoids(rotations: np.ndarray) -> np.ndarray:
    """Return a robust representative rotation without averaging rvec values."""

    matrices = [rotvec_to_matrix(rv) for rv in rotations]
    count = len(matrices)
    if count == 1:
        return matrices[0]
    candidates = np.linspace(0, count - 1, min(count, 200), dtype=int)
    best_index = int(candidates[0])
    best_cost = float("inf")
    for candidate in candidates:
        distances = [rotation_distance_deg(matrices[candidate], matrix) for matrix in matrices]
        cost = float(np.median(distances))
        if cost < best_cost:
            best_cost = cost
            best_index = int(candidate)
    return matrices[best_index]


def _time_steps_s(timestamps_us: np.ndarray) -> np.ndarray:
    if len(timestamps_us) < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(timestamps_us) * 1e-6


def stationary_metrics(series: PoseSeries) -> Dict[str, Any]:
    center = np.median(series.position_mm, axis=0)
    radial = np.linalg.norm(series.position_mm - center, axis=1)
    reference_rotation = _rotation_medoids(series.rvec)
    rotation_deviation = np.asarray(
        [rotation_distance_deg(reference_rotation, rotvec_to_matrix(rv)) for rv in series.rvec], dtype=np.float64
    )
    steps = np.linalg.norm(np.diff(series.position_mm, axis=0), axis=1) if len(series) > 1 else np.empty(0)
    return {
        "samples": len(series),
        "duration_s": float((series.timestamp_us[-1] - series.timestamp_us[0]) * 1e-6) if len(series) > 1 else 0.0,
        "median_position_mm": [float(value) for value in center],
        "position_radial_rms_mm": float(math.sqrt(np.mean(radial**2))),
        "position_radial_median_mm": float(np.median(radial)),
        "position_radial_p95_mm": float(np.percentile(radial, 95)),
        "position_component_std_mm": [float(value) for value in np.std(series.position_mm, axis=0)],
        "orientation_rms_deg": float(math.sqrt(np.mean(rotation_deviation**2))),
        "orientation_median_deg": float(np.median(rotation_deviation)),
        "orientation_p95_deg": float(np.percentile(rotation_deviation, 95)),
        "median_step_mm": float(np.median(steps)) if len(steps) else 0.0,
        "step_p95_mm": float(np.percentile(steps, 95)) if len(steps) else 0.0,
    }


def motion_metrics(series: PoseSeries) -> Dict[str, Any]:
    """Smoothness-only metrics for real moving data with no reference truth."""

    timestamps = series.timestamp_us
    position = series.position_mm
    if len(series) < 3:
        return {"samples": len(series)}
    median_dt = float(np.median(_time_steps_s(timestamps)))
    # A 100 ms moving average approximates the low-frequency trajectory.  The
    # residual is deliberately labeled jitter, not accuracy, without ground truth.
    window = max(3, int(round(0.10 / max(median_dt, 1e-4))))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(position, ((window // 2, window // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)], axis=1)
    high_frequency = np.linalg.norm(position - smooth, axis=1)
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    displacement = float(np.linalg.norm(position[-1] - position[0]))
    return {
        "samples": len(series),
        "duration_s": float((timestamps[-1] - timestamps[0]) * 1e-6),
        "median_dt_ms": median_dt * 1e3,
        "smoothing_window_samples": int(window),
        "high_frequency_position_rms_mm": float(math.sqrt(np.mean(high_frequency**2))),
        "high_frequency_position_p95_mm": float(np.percentile(high_frequency, 95)),
        "median_step_mm": float(np.median(steps)),
        "step_p95_mm": float(np.percentile(steps, 95)),
        "endpoint_displacement_mm": displacement,
    }


def ground_truth_metrics(estimate: PoseSeries, ground_truth: PoseSeries, outlier_threshold_mm: float = 50.0) -> Dict[str, Any]:
    if len(estimate) != len(ground_truth) or not np.allclose(estimate.timestamp_us, ground_truth.timestamp_us):
        raise ValueError("estimate and ground truth must have equal, matching timestamps")
    position_error = np.linalg.norm(estimate.position_mm - ground_truth.position_mm, axis=1)
    rotation_error = np.asarray(
        [
            rotation_distance_deg(rotvec_to_matrix(est), rotvec_to_matrix(gt))
            for est, gt in zip(estimate.rvec, ground_truth.rvec)
        ],
        dtype=np.float64,
    )
    return {
        "samples": len(estimate),
        "position_rmse_mm": float(math.sqrt(np.mean(position_error**2))),
        "position_median_mm": float(np.median(position_error)),
        "position_p95_mm": float(np.percentile(position_error, 95)),
        "rotation_rmse_deg": float(math.sqrt(np.mean(rotation_error**2))),
        "rotation_median_deg": float(np.median(rotation_error)),
        "rotation_p95_deg": float(np.percentile(rotation_error, 95)),
        "position_outlier_rate_percent": float(100.0 * np.mean(position_error > outlier_threshold_mm)),
    }


def _load_marker_and_camera(marker_path: Path, calibration_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    marker_doc = json.loads(marker_path.read_text(encoding="utf-8"))
    points = []
    for node in marker_doc["active marker"]:
        points.append(node["led"]["xyz"])
    object_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    calibration_doc = json.loads(calibration_path.read_text(encoding="utf-8"))
    K_values = np.asarray(calibration_doc["K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(calibration_doc.get("D", []), dtype=np.float64).reshape(-1, 1)
    return object_points, K_values, distortion


def _look_at_pose(t_s: float, trajectory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera center, world-to-camera rvec, world-to-camera tvec."""

    if trajectory == "circular":
        radius, height, frequency = 200.0, 500.0, 0.5
        center = np.array(
            [radius * math.cos(2.0 * math.pi * frequency * t_s), radius * math.sin(2.0 * math.pi * frequency * t_s), height],
            dtype=np.float64,
        )
    elif trajectory == "sinusoidal":
        center = np.array(
            [
                130.0 * math.sin(2.0 * math.pi * 0.35 * t_s),
                80.0 * math.sin(2.0 * math.pi * 0.50 * t_s),
                500.0 + 40.0 * math.sin(2.0 * math.pi * 0.20 * t_s),
            ],
            dtype=np.float64,
        )
    elif trajectory == "linear":
        center = np.array([80.0 * (t_s - 5.0), 0.0, 500.0], dtype=np.float64)
    else:
        raise ValueError(f"unknown trajectory {trajectory}")

    forward = -center / max(float(np.linalg.norm(center)), EPS)
    up_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), EPS)
    down = np.cross(forward, right)
    down /= max(float(np.linalg.norm(down)), EPS)
    R_cw = np.column_stack([right, down, forward])
    R_wc = R_cw.T
    tvec = -R_wc @ center
    return center, matrix_to_rotvec(R_wc), tvec


def _simulate_series(args: argparse.Namespace) -> Tuple[PoseSeries, PoseSeries, Dict[str, Any]]:
    marker, K, distortion = _load_marker_and_camera(args.marker_json, args.calibration_json)
    rng = np.random.default_rng(args.seed)
    count = int(round(args.duration_s * args.rate_hz))
    timestamps = np.arange(count, dtype=np.float64) * (1e6 / args.rate_hz)
    gt_position = np.zeros((count, 3), dtype=np.float64)
    gt_rvec = np.zeros((count, 3), dtype=np.float64)
    estimated_position: List[np.ndarray] = []
    estimated_rvec: List[np.ndarray] = []
    correlated_noise = np.zeros((len(marker), 2), dtype=np.float64)
    pnp_failures = 0

    for frame, timestamp_us in enumerate(timestamps):
        time_s = timestamp_us * 1e-6
        center, rvec_gt, _ = _look_at_pose(time_s, args.trajectory)
        gt_position[frame] = center
        gt_rvec[frame] = rvec_gt

        image_points = []
        for marker_index, point in enumerate(marker):
            jitter_s = rng.uniform(-args.timestamp_jitter_us, args.timestamp_jitter_us) * 1e-6
            _, rvec_observation, tvec_observation = _look_at_pose(
                float(np.clip(time_s + jitter_s, 0.0, args.duration_s)), args.trajectory
            )
            projection, _ = cv2.projectPoints(
                point.reshape(1, 3), rvec_observation.reshape(3, 1), tvec_observation.reshape(3, 1), K, distortion
            )
            rho = float(np.clip(args.correlated_noise_rho, 0.0, 0.9999))
            correlated_noise[marker_index] = (
                rho * correlated_noise[marker_index]
                + math.sqrt(max(1.0 - rho * rho, 0.0)) * rng.normal(0.0, args.correlated_noise_px, size=2)
            )
            noise = rng.normal(0.0, args.pixel_noise_px, size=2) + correlated_noise[marker_index]
            if rng.random() < args.outlier_probability:
                noise += rng.normal(0.0, args.outlier_std_px, size=2)
            image_points.append(projection.reshape(2) + noise)

        image_points_np = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            marker,
            image_points_np,
            K,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=100,
            reprojectionError=args.ransac_threshold_px,
            confidence=0.99,
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(marker, image_points_np, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            pnp_failures += 1
            # Keep timestamps aligned.  The known GT placeholder is replaced by
            # the previous available estimate, which is explicit in metadata.
            if estimated_position:
                estimated_position.append(estimated_position[-1].copy())
                estimated_rvec.append(estimated_rvec[-1].copy())
            else:
                estimated_position.append(center.copy())
                estimated_rvec.append(rvec_gt.copy())
            continue
        R_wc = rotvec_to_matrix(rvec.reshape(3))
        camera_center = -R_wc.T @ tvec.reshape(3)
        estimated_position.append(camera_center)
        estimated_rvec.append(rvec.reshape(3))

    raw = PoseSeries(timestamps, np.asarray(estimated_position), np.asarray(estimated_rvec))
    ground_truth = PoseSeries(timestamps, gt_position, gt_rvec)
    metadata = {
        "pnp_failures": int(pnp_failures),
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


def _save_ground_truth_csv(path: Path, series: PoseSeries) -> None:
    empty = [FilterStep(p, rv, True, True, 0.0, 0.0) for p, rv in zip(series.position_mm, series.rvec)]
    write_pose_csv(path, series, empty)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(content), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _plot_simulation(output_dir: Path, ground_truth: PoseSeries, raw: PoseSeries, filtered: PoseSeries) -> None:
    seconds = ground_truth.timestamp_us * 1e-6
    raw_error = np.linalg.norm(raw.position_mm - ground_truth.position_mm, axis=1)
    filter_error = np.linalg.norm(filtered.position_mm - ground_truth.position_mm, axis=1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ground_truth.position_mm[:, 0], ground_truth.position_mm[:, 1], "k-", lw=2, label="ground truth")
    axes[0].plot(raw.position_mm[:, 0], raw.position_mm[:, 1], color="#d95f02", alpha=0.55, label="framewise PnP")
    axes[0].plot(filtered.position_mm[:, 0], filtered.position_mm[:, 1], color="#1b9e77", lw=1.5, label="robust temporal filter")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("Camera-center trajectory")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(seconds, raw_error, color="#d95f02", alpha=0.7, label="framewise PnP")
    axes[1].plot(seconds, filter_error, color="#1b9e77", lw=1.5, label="robust temporal filter")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position error [mm]")
    axes[1].set_title("Known-ground-truth error")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "simulation_comparison.png", dpi=180)
    plt.close(figure)


def _plot_real(output_dir: Path, raw: PoseSeries, filtered: PoseSeries, title: str) -> None:
    seconds = (raw.timestamp_us - raw.timestamp_us[0]) * 1e-6
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = ("x", "y", "z")
    for axis, label in enumerate(labels):
        axes[0, 0].plot(seconds, raw.position_mm[:, axis], alpha=0.55, label=f"raw {label}")
        axes[0, 1].plot(seconds, filtered.position_mm[:, axis], alpha=0.8, label=f"filtered {label}")
    axes[0, 0].set_title("Raw camera-center coordinates")
    axes[0, 1].set_title("Filtered camera-center coordinates")
    for panel in (axes[0, 0], axes[0, 1]):
        panel.set_xlabel("time [s]")
        panel.set_ylabel("position [mm]")
        panel.legend(ncol=3, fontsize=8)
        panel.grid(alpha=0.25)

    axes[1, 0].plot(raw.position_mm[:, 0], raw.position_mm[:, 2], color="#d95f02", alpha=0.65, label="raw")
    axes[1, 0].plot(filtered.position_mm[:, 0], filtered.position_mm[:, 2], color="#1b9e77", alpha=0.8, label="filtered")
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("z [mm]")
    axes[1, 0].set_title("x-z trajectory")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    raw_steps = np.r_[0.0, np.linalg.norm(np.diff(raw.position_mm, axis=0), axis=1)]
    filtered_steps = np.r_[0.0, np.linalg.norm(np.diff(filtered.position_mm, axis=0), axis=1)]
    axes[1, 1].plot(seconds, raw_steps, color="#d95f02", alpha=0.65, label="raw")
    axes[1, 1].plot(seconds, filtered_steps, color="#1b9e77", alpha=0.8, label="filtered")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("inter-sample displacement [mm]")
    axes[1, 1].set_title("Frame-to-frame variation")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_dir / "real_pose_comparison.png", dpi=180)
    plt.close(figure)


def _metric_markdown(summary: Dict[str, Any]) -> str:
    """A compact Korean thesis-lab-note; values stay machine-readable in JSON."""

    lines = ["# 시간적 포즈 필터 실험 요약", ""]
    kind = summary.get("experiment_type", "")
    if kind == "simulation":
        lines += [
            "이 결과는 알려진 ground truth가 있는 합성 2D 투영 실험의 정확도 평가이다.",
            "실데이터의 절대 정확도 주장에는 사용하지 않는다.",
            "",
            "| 방법 | 위치 RMSE (mm) | 회전 RMSE (deg) | 위치 이상치율 (%) |",
            "|---|---:|---:|---:|",
        ]
        for name, metric in summary["metrics"].items():
            lines.append(
                f"| {name} | {metric['position_rmse_mm']:.3f} | {metric['rotation_rmse_deg']:.3f} | "
                f"{metric['position_outlier_rate_percent']:.2f} |"
            )
    else:
        stationary = summary.get("sequence_type") == "stationary"
        if stationary:
            lines += [
                "정지 실험에서는 실제 카메라 포즈가 상수이므로, 중앙값 주변 산포를 정밀도(jitter)로 평가한다.",
                "절대 위치 바이어스는 외부 기준 장비가 없으므로 별도로 주장하지 않는다.",
                "",
                "| 방법 | 위치 RMS jitter (mm) | 위치 P95 (mm) | 회전 RMS jitter (deg) |",
                "|---|---:|---:|---:|",
            ]
            for name, metric in summary["metrics"].items():
                lines.append(
                    f"| {name} | {metric['position_radial_rms_mm']:.3f} | {metric['position_radial_p95_mm']:.3f} | "
                    f"{metric['orientation_rms_deg']:.3f} |"
                )
        else:
            lines += [
                "동적 실험에는 외부 ground truth가 없으므로, 아래 수치는 정확도가 아니라 고주파 위치 변동(jitter) 지표이다.",
                "",
                "| 방법 | 고주파 RMS (mm) | 고주파 P95 (mm) | step P95 (mm) |",
                "|---|---:|---:|---:|",
            ]
            for name, metric in summary["metrics"].items():
                lines.append(
                    f"| {name} | {metric['high_frequency_position_rms_mm']:.3f} | "
                    f"{metric['high_frequency_position_p95_mm']:.3f} | {metric['step_p95_mm']:.3f} |"
                )
    lines += ["", "필터 설정은 `summary.json`에 함께 기록되어 재현 가능하다.", ""]
    return "\n".join(lines)


def _filter_config_from_args(args: argparse.Namespace) -> TemporalFilterConfig:
    return TemporalFilterConfig(
        process_accel_std_mm_s2=args.process_accel_std_mm_s2,
        measurement_std_mm=args.measurement_std_mm,
        gate_chi2=args.gate_chi2,
        hard_gate_chi2=args.hard_gate_chi2,
        orientation_alpha=args.orientation_alpha,
        orientation_beta=args.orientation_beta,
        orientation_gate_deg=args.orientation_gate_deg,
        orientation_hard_gate_deg=args.orientation_hard_gate_deg,
        reset_gap_s=args.reset_gap_s,
    )


def run_simulation(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, ground_truth, metadata = _simulate_series(args)
    config = _filter_config_from_args(args)
    filtered, diagnostics = filter_series(raw, config)
    summary = {
        "experiment_type": "simulation",
        "metadata": metadata,
        "filter_config": asdict(config),
        "metrics": {
            "framewise_pnp": ground_truth_metrics(raw, ground_truth, args.outlier_position_threshold_mm),
            "robust_temporal_filter": ground_truth_metrics(filtered, ground_truth, args.outlier_position_threshold_mm),
        },
        "filter_diagnostics": {
            "translation_rejection_rate_percent": float(100.0 * np.mean([not step.translation_accepted for step in diagnostics])),
            "rotation_rejection_rate_percent": float(100.0 * np.mean([not step.rotation_accepted for step in diagnostics])),
        },
    }
    raw_steps = [FilterStep(p, rv, True, True, 0.0, 0.0) for p, rv in zip(raw.position_mm, raw.rvec)]
    write_pose_csv(output_dir / "framewise_pnp.csv", raw, raw_steps)
    _save_ground_truth_csv(output_dir / "ground_truth.csv", ground_truth)
    write_pose_csv(output_dir / "robust_temporal_filter.csv", filtered, diagnostics)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_metric_markdown(summary), encoding="utf-8")
    _plot_simulation(output_dir, ground_truth, raw, filtered)
    print(json.dumps(_json_ready(summary["metrics"]), ensure_ascii=False, indent=2))


def run_real_analysis(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_pose_csv(args.input)
    config = _filter_config_from_args(args)
    filtered, diagnostics = filter_series(raw, config)
    sequence_type = args.sequence_type
    if sequence_type == "auto":
        sequence_type = "stationary" if "stop" in args.input.name.lower() else "moving"
    metric_function = stationary_metrics if sequence_type == "stationary" else motion_metrics
    summary = {
        "experiment_type": "real_recording",
        "input": str(args.input),
        "sequence_type": sequence_type,
        "filter_config": asdict(config),
        "metrics": {"raw_pnp": metric_function(raw), "robust_temporal_filter": metric_function(filtered)},
        "filter_diagnostics": {
            "translation_rejection_rate_percent": float(100.0 * np.mean([not step.translation_accepted for step in diagnostics])),
            "rotation_rejection_rate_percent": float(100.0 * np.mean([not step.rotation_accepted for step in diagnostics])),
            "median_translation_innovation_mm": float(np.median([step.innovation_mm for step in diagnostics])),
            "p95_translation_innovation_mm": float(np.percentile([step.innovation_mm for step in diagnostics], 95)),
        },
    }
    raw_steps = [FilterStep(p, rv, True, True, 0.0, 0.0) for p, rv in zip(raw.position_mm, raw.rvec)]
    write_pose_csv(output_dir / "raw_input_normalized.csv", raw, raw_steps)
    write_pose_csv(output_dir / "robust_temporal_filter.csv", filtered, diagnostics)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_metric_markdown(summary), encoding="utf-8")
    _plot_real(output_dir, raw, filtered, f"Real recording: {args.input.name}")
    print(json.dumps(_json_ready(summary["metrics"]), ensure_ascii=False, indent=2))


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--process-accel-std-mm-s2", type=float, default=2500.0)
    parser.add_argument("--measurement-std-mm", type=float, default=45.0)
    parser.add_argument("--gate-chi2", type=float, default=16.27)
    parser.add_argument("--hard-gate-chi2", type=float, default=81.0)
    parser.add_argument("--orientation-alpha", type=float, default=0.35)
    parser.add_argument("--orientation-beta", type=float, default=0.025)
    parser.add_argument("--orientation-gate-deg", type=float, default=20.0)
    parser.add_argument("--orientation-hard-gate-deg", type=float, default=70.0)
    parser.add_argument("--reset-gap-s", type=float, default=0.25)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulation = subparsers.add_parser("simulate", help="known-ground-truth 2D projection + PnP experiment")
    simulation.add_argument("--output-dir", type=Path, required=True)
    simulation.add_argument("--marker-json", type=Path, default=Path("config/marker/multi_led_marker.json"))
    simulation.add_argument("--calibration-json", type=Path, default=Path("config/calibration_1017.json"))
    simulation.add_argument("--trajectory", choices=("circular", "linear", "sinusoidal"), default="sinusoidal")
    simulation.add_argument("--duration-s", type=float, default=10.0)
    simulation.add_argument("--rate-hz", type=float, default=100.0)
    simulation.add_argument("--pixel-noise-px", type=float, default=1.0)
    simulation.add_argument("--timestamp-jitter-us", type=float, default=1000.0)
    simulation.add_argument("--correlated-noise-px", type=float, default=0.35)
    simulation.add_argument("--correlated-noise-rho", type=float, default=0.85)
    simulation.add_argument("--outlier-probability", type=float, default=0.02)
    simulation.add_argument("--outlier-std-px", type=float, default=8.0)
    simulation.add_argument("--ransac-threshold-px", type=float, default=4.0)
    simulation.add_argument("--outlier-position-threshold-mm", type=float, default=50.0)
    simulation.add_argument("--seed", type=int, default=20260929)
    add_filter_arguments(simulation)
    simulation.set_defaults(handler=run_simulation)

    real = subparsers.add_parser("real", help="filter and characterize an existing pose CSV")
    real.add_argument("--input", type=Path, required=True)
    real.add_argument("--output-dir", type=Path, required=True)
    real.add_argument("--sequence-type", choices=("auto", "stationary", "moving"), default="auto")
    add_filter_arguments(real)
    real.set_defaults(handler=run_real_analysis)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "simulate":
        # Defaults are relative to the event_mocap_unified project root.
        args.marker_json = args.marker_json.resolve()
        args.calibration_json = args.calibration_json.resolve()
    args.handler(args)


if __name__ == "__main__":
    main()
