#!/usr/bin/env python3
"""Visualize pose estimation results from CSV output."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize pose CSV")
    parser.add_argument("csv", nargs="?", default="pose_log.csv", help="CSV file path")
    parser.add_argument("--outlier-threshold", type=float, default=5000,
                        help="Remove points with |x|,|y|,|z| > threshold")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Handle both column naming conventions
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_us"

    if args.end > 0:
        df = df.iloc[args.start:args.end]
    elif args.start > 0:
        df = df.iloc[args.start:]

    # Outlier removal
    thr = args.outlier_threshold
    mask = (df["x"].abs() < thr) & (df["y"].abs() < thr) & (df["z"].abs() < thr)
    df = df[mask]
    print(f"Plotting {len(df)} points (removed {(~mask).sum()} outliers)")

    x, y, z = df["x"].values, df["y"].values, df["z"].values
    t = df[ts_col].values
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)

    print(f"Start: ({x[0]:.1f}, {y[0]:.1f}, {z[0]:.1f})")
    print(f"End:   ({x[-1]:.1f}, {y[-1]:.1f}, {z[-1]:.1f})")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=t_norm, cmap="viridis", marker="o", s=5)
    ax.set_box_aspect([1, 1, 1])

    plt.colorbar(sc, ax=ax, label="Normalized Time")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Camera 3D Trajectory")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
