#!/usr/bin/env python3
"""Stable command entry point for production-matched thesis simulations.

The implementation is kept in ``run_thesis_pose_experiments.py`` so its
projection/PnP logic can be inspected independently.  This wrapper supplies
the simulation-specific summary renderer before invoking that implementation.
"""

from __future__ import annotations

import run_thesis_pose_experiments as experiment
import evaluate_temporal_pose_filter as helpers


_render_summary = helpers._metric_markdown


def _render_production_simulation_summary(summary):
    if summary.get("experiment_type") == "production_matched_simulation":
        summary = dict(summary)
        summary["experiment_type"] = "simulation"
    return _render_summary(summary)


helpers._metric_markdown = _render_production_simulation_summary


if __name__ == "__main__":
    experiment.main()
