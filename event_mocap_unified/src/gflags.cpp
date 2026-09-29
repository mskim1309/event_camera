// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Gflags parameter definitions with defaults.
// Priority: default < YAML config < CLI argument.

#include <gflags/gflags.h>

// ============================================================
// Mode & I/O
// ============================================================
DEFINE_string(mode,          "sim",            "Execution mode: sim, file, live");
// NOTE: FLAGS_config is defined in util/yaml.cc
DEFINE_string(marker_json,   "",               "Path to active marker JSON");
DEFINE_string(calib_json,    "",               "Path to camera calibration JSON");
DEFINE_string(camera_config, "",               "Camera bias config JSON (live/file)");
DEFINE_string(input,         "",               "Event file path (.raw) for file mode");
DEFINE_string(output,        "result/pose_log.csv",    "CSV output path");
DEFINE_bool(visualize,       false,            "Enable Rerun visualization");
DEFINE_bool(realtime,        false,            "Realtime playback for file mode");

// ============================================================
// Pose Estimation
// ============================================================
DEFINE_int32(ba_window_size, 5,                "Bundle adjustment sliding window size");
DEFINE_string(pnp_candidate_mode, "auto",
              "PnP candidate mode: auto (IPPE for exactly four coplanar matches; "
              "SQPNP-flagged RANSAC otherwise) or legacy_iterative_ransac for ablation; "
              "acceptance residual gates use the selected consensus");
DEFINE_string(pnp_diagnostics_frame_output, "",
              "Optional CSV path for one row per PnP estimation attempt");
DEFINE_string(pnp_diagnostics_correspondence_output, "",
              "Optional CSV path for one row per observed marker candidate");
DEFINE_bool(temporal_filter, false,
            "Apply causal robust temporal filtering after raw PnP (off preserves raw output)");
DEFINE_string(temporal_filter_diagnostics_output, "",
              "Optional CSV path for raw/filter pose and gate diagnostics");
DEFINE_double(temporal_filter_process_accel_std_mm_s2, 10000.0,
              "Temporal filter white-acceleration std (mm/s^2)");
DEFINE_double(temporal_filter_measurement_std_mm, 15.0,
              "Temporal filter position measurement std (mm)");
DEFINE_double(temporal_filter_gate_chi2, 16.27,
              "Temporal filter soft translation innovation gate (chi-square)");
DEFINE_double(temporal_filter_hard_gate_chi2, 81.0,
              "Temporal filter hard translation innovation gate (chi-square)");
DEFINE_double(temporal_filter_orientation_alpha, 0.35,
              "Temporal filter SO(3) complementary update gain");
DEFINE_double(temporal_filter_orientation_beta, 0.025,
              "Temporal filter SO(3) angular-velocity update gain");
DEFINE_double(temporal_filter_orientation_gate_deg, 180.0,
              "Temporal filter soft rotation gate (default 180 disables shortest-path gating)");
DEFINE_double(temporal_filter_orientation_hard_gate_deg, 360.0,
              "Temporal filter hard rotation gate (default 360 disables shortest-path gating)");
DEFINE_int32(temporal_filter_reset_gap_us, 250000,
             "Temporal filter reset when timestamp gap exceeds this value (us)");
DEFINE_int32(temporal_filter_max_consecutive_rejections, 4,
             "Temporal filter reinitialize after this many translation rejections");

// ============================================================
// 2D Tracking (Metavision SDK)
// ============================================================
DEFINE_int32(det_num_bits,          8,         "Bits encoding LED ID");
DEFINE_int32(det_base_period_us,    400,       "LED blink base period (us)");
DEFINE_double(det_tolerance,        0.3,       "Period measurement tolerance");

DEFINE_double(trk_radius,              30.0,   "Tracker search radius (px)");
DEFINE_double(trk_monitoring_freq_hz,  30.0,   "Tracker monitoring frequency (Hz)");
DEFINE_int32(trk_inactivity_period_us, 1000,   "Mark lost after this silence (us)");
DEFINE_double(trk_alpha_pos,           0.05,   "Position smoothing factor");
DEFINE_double(trk_distance_pct,        0.3,    "Radius update factor");

DEFINE_int32(obs_max_samples,       50,        "Max buffered samples per LED");
DEFINE_int32(obs_max_age_us,        2000,      "Discard observations older than (us)");
DEFINE_int32(obs_delta_n_events,    5000,      "Emit observations every N events");

// ============================================================
// Simulation
// ============================================================
DEFINE_string(trajectory,    "circular",       "Trajectory type: circular, linear, sinusoidal");
DEFINE_double(sim_duration,  10.0,             "Simulation duration (seconds)");
DEFINE_double(sim_rate_hz,   100.0,            "Simulation observation rate (Hz)");
DEFINE_double(sim_pixel_noise, 0.5,            "Pixel noise std (simulation)");

// Circular
DEFINE_double(sim_circle_radius,  200.0,       "Circular trajectory radius (mm)");
DEFINE_double(sim_circle_height,  500.0,       "Circular trajectory height (mm)");
DEFINE_double(sim_circle_freq,    0.5,         "Circular trajectory frequency (Hz)");

// Linear
DEFINE_double(sim_linear_speed,   50.0,        "Linear trajectory speed (mm/s)");
DEFINE_double(sim_linear_z,       500.0,       "Linear trajectory z distance (mm)");

// Sinusoidal
DEFINE_double(sim_sin_amp_x,   100.0,          "Sinusoidal amplitude X (mm)");
DEFINE_double(sim_sin_amp_y,   50.0,           "Sinusoidal amplitude Y (mm)");
DEFINE_double(sim_sin_amp_z,   30.0,           "Sinusoidal amplitude Z (mm)");
DEFINE_double(sim_sin_freq_x,  0.3,            "Sinusoidal frequency X (Hz)");
DEFINE_double(sim_sin_freq_y,  0.5,            "Sinusoidal frequency Y (Hz)");
DEFINE_double(sim_sin_freq_z,  0.2,            "Sinusoidal frequency Z (Hz)");
DEFINE_double(sim_sin_base_z,  500.0,          "Sinusoidal base Z distance (mm)");
