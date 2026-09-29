// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Gflags parameter declarations for event_mocap_unified.
// All flags can be overridden via CLI or YAML config file.

#pragma once

#include <gflags/gflags.h>

// ============================================================
// Mode & I/O
// ============================================================
DECLARE_string(mode);            // sim, file, live
DECLARE_string(config);          // YAML config file path (defined in util/yaml.cc)
DECLARE_string(marker_json);     // active marker JSON
DECLARE_string(calib_json);      // camera calibration JSON
DECLARE_string(camera_config);   // camera bias config (live/file)
DECLARE_string(input);           // event file path (.raw)
DECLARE_string(output);          // CSV output path
DECLARE_bool(visualize);         // enable Rerun visualization
DECLARE_bool(realtime);          // realtime playback (file mode)

// ============================================================
// Pose Estimation
// ============================================================
DECLARE_int32(ba_window_size);   // sliding window BA frame count
// `auto`: IPPE for exactly four coplanar matches, otherwise a SQPNP-flagged
// RANSAC candidate. `legacy_iterative_ransac` preserves the former candidate
// stage for an ablation; neither mode changes the temporal output filter.
// Acceptance gates use the selected consensus; diagnostics retain residuals
// for every matched marker observation.
DECLARE_string(pnp_candidate_mode);

// Optional raw-PnP diagnostics. These do not change the historical pose CSV.
DECLARE_string(pnp_diagnostics_frame_output);
DECLARE_string(pnp_diagnostics_correspondence_output);

// Causal post-PnP output filter. Disabled by default so the current repaired
// robust-PnP measurement baseline is preserved; filtered output is never a
// PnP prior.
DECLARE_bool(temporal_filter);
DECLARE_string(temporal_filter_diagnostics_output);
DECLARE_double(temporal_filter_process_accel_std_mm_s2);
DECLARE_double(temporal_filter_measurement_std_mm);
DECLARE_double(temporal_filter_gate_chi2);
DECLARE_double(temporal_filter_hard_gate_chi2);
DECLARE_double(temporal_filter_orientation_alpha);
DECLARE_double(temporal_filter_orientation_beta);
DECLARE_double(temporal_filter_orientation_gate_deg);
DECLARE_double(temporal_filter_orientation_hard_gate_deg);
DECLARE_int32(temporal_filter_reset_gap_us);
DECLARE_int32(temporal_filter_max_consecutive_rejections);

// ============================================================
// 2D Tracking (Metavision SDK) — defined regardless, used only with HAS_METAVISION
// ============================================================
DECLARE_int32(det_num_bits);
DECLARE_int32(det_base_period_us);
DECLARE_double(det_tolerance);

DECLARE_double(trk_radius);
DECLARE_double(trk_monitoring_freq_hz);
DECLARE_int32(trk_inactivity_period_us);
DECLARE_double(trk_alpha_pos);
DECLARE_double(trk_distance_pct);

DECLARE_int32(obs_max_samples);
DECLARE_int32(obs_max_age_us);
DECLARE_int32(obs_delta_n_events);

// ============================================================
// Simulation
// ============================================================
DECLARE_string(trajectory);      // circular, linear, sinusoidal
DECLARE_double(sim_duration);
DECLARE_double(sim_rate_hz);
DECLARE_double(sim_pixel_noise);

// Circular
DECLARE_double(sim_circle_radius);
DECLARE_double(sim_circle_height);
DECLARE_double(sim_circle_freq);

// Linear
DECLARE_double(sim_linear_speed);
DECLARE_double(sim_linear_z);

// Sinusoidal
DECLARE_double(sim_sin_amp_x);
DECLARE_double(sim_sin_amp_y);
DECLARE_double(sim_sin_amp_z);
DECLARE_double(sim_sin_freq_x);
DECLARE_double(sim_sin_freq_y);
DECLARE_double(sim_sin_freq_z);
DECLARE_double(sim_sin_base_z);
