#!/usr/bin/env python3
"""Characterize noise patterns in a stationary event-camera pose CSV.

This analysis deliberately describes *pose-output* noise.  A pose-only CSV
cannot distinguish pixel-centroid noise from calibration error, PnP ambiguity,
or timestamp mismatch; those causes require the marker-level diagnostic log
planned for RAW replay.  Still, the script provides evidence for whether a
causal filter should model white noise, temporal correlation, periodic
components, or rare impulsive pose failures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, welch

import evaluate_temporal_pose_filter as pose_tools


EPS = 1e-12


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def huber_linear_detrend(time_s: np.ndarray, values: np.ndarray, iterations: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Robust line fit to avoid one mirror pose flip defining the trend."""

    design = np.column_stack([np.ones_like(time_s), time_s])
    weights = np.ones_like(time_s)
    coefficients = np.zeros(2)
    for _ in range(iterations):
        weighted_design = design * weights[:, None]
        weighted_values = values * weights
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        residual = values - design @ coefficients
        scale = max(robust_scale(residual), 1e-6)
        standardized = np.abs(residual) / (1.345 * scale)
        updated = np.where(standardized <= 1.0, 1.0, 1.0 / standardized)
        if np.max(np.abs(updated - weights)) < 1e-5:
            break
        weights = updated
    return values - design @ coefficients, coefficients


def resample_uniform(time_s: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Linear resampling for PSD/ACF only; original timestamps remain in JSON."""

    dts = np.diff(time_s)
    positive_dts = dts[dts > 0]
    if len(positive_dts) == 0:
        raise ValueError("timestamps do not increase")
    dt = float(np.median(positive_dts))
    uniform_time = np.arange(time_s[0], time_s[-1] + 0.5 * dt, dt)
    uniform_values = np.column_stack([np.interp(uniform_time, time_s, values[:, axis]) for axis in range(values.shape[1])])
    return uniform_time, uniform_values, dt


def normalized_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator < EPS:
        return np.r_[1.0, np.zeros(max_lag)]
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 : len(centered) + max_lag]
    return correlation / denominator


def allan_deviation(values: np.ndarray, sample_dt_s: float, cluster_sizes: Iterable[int]) -> List[Dict[str, float]]:
    """Non-overlapping Allan deviation of a scalar zero-mean sequence."""

    output: List[Dict[str, float]] = []
    for cluster_size in cluster_sizes:
        groups = len(values) // cluster_size
        if groups < 3:
            continue
        cluster_means = values[: groups * cluster_size].reshape(groups, cluster_size).mean(axis=1)
        adev = math.sqrt(0.5 * float(np.mean(np.diff(cluster_means) ** 2)))
        output.append({"tau_s": float(cluster_size * sample_dt_s), "allan_deviation": float(adev)})
    return output


def top_psd_peaks(values: np.ndarray, sample_rate_hz: float) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    nperseg = min(256, len(values))
    frequencies, powers = welch(values, fs=sample_rate_hz, nperseg=nperseg, detrend="constant")
    if len(frequencies) < 3:
        return frequencies, powers, []
    # Ignore DC / sub-Hz trend; frequency candidates are not claimed to be a
    # physical flicker source until marker-level events confirm them.
    eligible = np.where(frequencies >= 1.0)[0]
    peaks, _ = find_peaks(powers)
    peaks = np.intersect1d(peaks, eligible)
    ordered = peaks[np.argsort(powers[peaks])[::-1]][:3]
    return frequencies, powers, [
        {"frequency_hz": float(frequencies[index]), "power": float(powers[index])} for index in ordered
    ]


def rotation_tangent_residuals(series: pose_tools.PoseSeries) -> np.ndarray:
    reference = pose_tools._rotation_medoids(series.rvec)
    return np.asarray(
        [pose_tools.log_so3(reference.T @ pose_tools.rotvec_to_matrix(rv)) for rv in series.rvec], dtype=np.float64
    ) * 180.0 / math.pi


def axis_characterization(
    time_s: np.ndarray,
    residuals: np.ndarray,
    label: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    uniform_time, uniform_residuals, sample_dt = resample_uniform(time_s, residuals)
    sample_rate = 1.0 / sample_dt
    max_lag = min(int(round(0.25 / sample_dt)), len(uniform_time) - 1)
    acfs = np.column_stack([normalized_acf(uniform_residuals[:, axis], max_lag) for axis in range(3)])
    psd_results = [top_psd_peaks(uniform_residuals[:, axis], sample_rate) for axis in range(3)]
    dominant = {axis: peaks for axis, (_, _, peaks) in zip(("x", "y", "z"), psd_results)}
    norm = np.linalg.norm(uniform_residuals, axis=1)
    clusters = [1, 2, 4, 8, 16, 32, 64, 128]
    result = {
        "label": label,
        "sample_rate_hz_after_resampling": float(sample_rate),
        "robust_sigma": [float(robust_scale(residuals[:, axis])) for axis in range(3)],
        "rms": [float(math.sqrt(np.mean(residuals[:, axis] ** 2))) for axis in range(3)],
        "lag1_acf": [float(acfs[1, axis]) if len(acfs) > 1 else 0.0 for axis in range(3)],
        "lag10_acf": [float(acfs[min(10, len(acfs) - 1), axis]) for axis in range(3)],
        "dominant_psd_peaks": dominant,
        "norm_allan_deviation": allan_deviation(norm, sample_dt, clusters),
    }
    spectra = [(frequency, power) for frequency, power, _ in psd_results]
    return result, uniform_time, uniform_residuals, acfs, spectra


def detect_step_outliers(position: np.ndarray) -> Dict[str, Any]:
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    median = float(np.median(steps))
    sigma = max(robust_scale(steps), 1e-6)
    threshold = median + 6.0 * sigma
    indices = np.flatnonzero(steps > threshold)
    return {
        "median_step_mm": median,
        "robust_step_sigma_mm": sigma,
        "outlier_threshold_mm": threshold,
        "count": int(len(indices)),
        "rate_percent": float(100.0 * len(indices) / max(len(steps), 1)),
        "maximum_step_mm": float(np.max(steps)) if len(steps) else 0.0,
        "sample_indices_after_jump": [int(index + 1) for index in indices[:50]],
    }


def interpret(result: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    outlier = result["step_outliers"]
    position = result["translation"]
    lag1 = max(abs(value) for value in position["lag1_acf"])
    if outlier["count"]:
        notes.append(
            f"Rare impulsive pose failures are present: {outlier['count']} step(s) exceed the robust {outlier['outlier_threshold_mm']:.1f} mm threshold."
        )
    if lag1 >= 0.3:
        notes.append(
            f"Translation residuals are temporally correlated (max lag-1 ACF={lag1:.2f}); an independent-white-noise-only model is insufficient."
        )
    else:
        notes.append(
            f"Translation lag-1 correlation is limited (max |ACF|={lag1:.2f}); a white-noise component is plausible after detrending."
        )
    peak_frequencies = [
        peak["frequency_hz"]
        for peaks in position["dominant_psd_peaks"].values()
        for peak in peaks
    ]
    if peak_frequencies:
        notes.append(
            "PSD peak candidates appear in the pose output; confirm them against per-pixel/event-rate spectra before attributing them to illumination flicker or sensor noise."
        )
    return notes


def plot_report(
    output_path: Path,
    time_s: np.ndarray,
    translation_residuals: np.ndarray,
    translation_uniform_time: np.ndarray,
    translation_acf: np.ndarray,
    translation_spectra: List[Tuple[np.ndarray, np.ndarray]],
    title: str,
) -> None:
    labels = ("x", "y", "z")
    colors = ("#1b9e77", "#d95f02", "#7570b3")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, (label, color) in enumerate(zip(labels, colors)):
        axes[0].plot(time_s, translation_residuals[:, axis], color=color, alpha=0.65, label=label)
    axes[0].set_title("Robust-detrended translation residual")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("residual [mm]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    lag_seconds = np.arange(len(translation_acf)) * (translation_uniform_time[1] - translation_uniform_time[0])
    for axis, (label, color) in enumerate(zip(labels, colors)):
        axes[1].plot(lag_seconds * 1e3, translation_acf[:, axis], color=color, label=label)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Autocorrelation after uniform resampling")
    axes[1].set_xlabel("lag [ms]")
    axes[1].set_ylabel("normalized ACF")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    for (frequency, power), label, color in zip(translation_spectra, labels, colors):
        axes[2].semilogy(frequency, np.maximum(power, 1e-15), color=color, label=label)
    axes[2].set_title("Welch PSD candidate spectrum")
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylabel("PSD [mm²/Hz]")
    axes[2].set_xlim(left=0.0)
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    series = pose_tools.load_pose_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    time_s = (series.timestamp_us - series.timestamp_us[0]) * 1e-6
    translation_residuals = np.column_stack(
        [huber_linear_detrend(time_s, series.position_mm[:, axis])[0] for axis in range(3)]
    )
    rotation = rotation_tangent_residuals(series)
    rotation_residuals = np.column_stack(
        [huber_linear_detrend(time_s, rotation[:, axis])[0] for axis in range(3)]
    )
    translation, uniform_time, _, translation_acf, translation_spectra = axis_characterization(
        time_s, translation_residuals, "translation_mm"
    )
    orientation, _, _, _, _ = axis_characterization(time_s, rotation_residuals, "rotation_deg")
    timestamp_dts = np.diff(series.timestamp_us)
    result: Dict[str, Any] = {
        "input": str(args.input),
        "samples": len(series),
        "duration_s": float(time_s[-1]) if len(time_s) else 0.0,
        "timestamp_interval_us": {
            "median": float(np.median(timestamp_dts)),
            "mean": float(np.mean(timestamp_dts)),
            "std": float(np.std(timestamp_dts)),
            "p05": float(np.percentile(timestamp_dts, 5)),
            "p95": float(np.percentile(timestamp_dts, 95)),
        },
        "step_outliers": detect_step_outliers(series.position_mm),
        "translation": translation,
        "rotation": orientation,
    }
    result["interpretation"] = interpret(result)
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_lines = ["# 정지 pose-output 노이즈 특성화", ""]
    report_lines += [f"입력: `{args.input}`", ""]
    report_lines += ["## 관찰", ""]
    report_lines += [f"- 표본 수: {result['samples']}, 기간: {result['duration_s']:.3f} s"]
    report_lines += [
        f"- timestamp 간격: median {result['timestamp_interval_us']['median']:.1f} µs, "
        f"std {result['timestamp_interval_us']['std']:.1f} µs"
    ]
    report_lines += [
        f"- robust step outlier: {result['step_outliers']['count']}개, "
        f"최대 step {result['step_outliers']['maximum_step_mm']:.1f} mm"
    ]
    report_lines += ["", "## 해석", ""]
    report_lines += [f"- {note}" for note in result["interpretation"]]
    report_lines += ["", "이 결과는 최종 pose CSV의 통계이며, 원인을 센서/LED 수준으로 단정하지 않는다.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    plot_report(
        args.output_dir / "noise_characterization.png",
        time_s,
        translation_residuals,
        uniform_time,
        translation_acf,
        translation_spectra,
        f"Stationary pose-output noise: {args.input.name}",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
