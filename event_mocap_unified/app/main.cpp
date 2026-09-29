// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <optional>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "util/yaml.h"
#include "util/google_log.h"

#include "event_mocap/gflags.h"
#include "event_mocap/types.hpp"
#include "event_mocap/config_loader.hpp"
#include "event_mocap/pose_estimator.hpp"
#include "event_mocap/temporal_pose_filter.hpp"
#include "event_mocap/data_source.hpp"
#include "event_mocap/visualizer.hpp"

int main(int argc, char* argv[]) {
    // --- Init gflags + glog ---
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // --- Load YAML config (if provided) — overrides defaults, CLI overrides YAML ---
    if (!FLAGS_config.empty()) {
        LOG(INFO) << "Loading config: " << FLAGS_config;
        rvl::LoadYAMLConfigAndSyncGoogleFlags(FLAGS_config);
    }

    // --- Validate required flags ---
    if (FLAGS_marker_json.empty()) {
        LOG(ERROR) << "--marker_json is required";
        return 1;
    }
    if (FLAGS_calib_json.empty()) {
        LOG(ERROR) << "--calib_json is required";
        return 1;
    }

    // --- Load configs ---
    auto marker = emocap::loadActiveMarker(FLAGS_marker_json);
    auto intrinsics = emocap::loadCalibration(FLAGS_calib_json);

    LOG(INFO) << "Loaded " << marker.size() << " markers, camera "
              << intrinsics.width << "x" << intrinsics.height
              << " fx=" << intrinsics.fx;

    // --- Parse mode ---
    emocap::Mode mode;
    if (FLAGS_mode == "sim")       mode = emocap::Mode::SIM;
    else if (FLAGS_mode == "file") mode = emocap::Mode::FILE;
    else if (FLAGS_mode == "live") mode = emocap::Mode::LIVE;
    else {
        LOG(ERROR) << "Unknown mode: " << FLAGS_mode;
        return 1;
    }

#ifndef HAS_METAVISION
    if (mode != emocap::Mode::SIM) {
        LOG(ERROR) << "MetavisionSDK not available — only sim mode is supported";
        return 1;
    }
#endif

    // --- Pose estimator ---
    const auto candidate_mode = emocap::parsePnPCandidateMode(FLAGS_pnp_candidate_mode);
    if (!candidate_mode) {
        LOG(ERROR) << "Unknown --pnp_candidate_mode: " << FLAGS_pnp_candidate_mode
                   << " (expected auto or legacy_iterative_ransac)";
        return 1;
    }
    emocap::PoseEstimator estimator(marker, intrinsics, FLAGS_ba_window_size,
                                    *candidate_mode);

    // Keep this as a separate output stage: the repaired robust PnP estimator
    // remains the measurement source, and filter output is never fed back as
    // a PnP prior. Disabling it preserves the current robust baseline.
    emocap::TemporalPoseFilterConfig temporal_filter_config;
    temporal_filter_config.process_accel_std_mm_s2 =
        FLAGS_temporal_filter_process_accel_std_mm_s2;
    temporal_filter_config.measurement_std_mm =
        FLAGS_temporal_filter_measurement_std_mm;
    temporal_filter_config.gate_chi2 = FLAGS_temporal_filter_gate_chi2;
    temporal_filter_config.hard_gate_chi2 = FLAGS_temporal_filter_hard_gate_chi2;
    temporal_filter_config.orientation_alpha = FLAGS_temporal_filter_orientation_alpha;
    temporal_filter_config.orientation_beta = FLAGS_temporal_filter_orientation_beta;
    temporal_filter_config.orientation_gate_deg = FLAGS_temporal_filter_orientation_gate_deg;
    temporal_filter_config.orientation_hard_gate_deg =
        FLAGS_temporal_filter_orientation_hard_gate_deg;
    temporal_filter_config.reset_gap_us =
        static_cast<emocap::Timestamp>(FLAGS_temporal_filter_reset_gap_us);
    temporal_filter_config.max_consecutive_translation_rejections =
        FLAGS_temporal_filter_max_consecutive_rejections;
    emocap::TemporalPoseFilter temporal_filter(temporal_filter_config);

    // --- Visualizer ---
    std::unique_ptr<emocap::Visualizer> vis;
    if (FLAGS_visualize) {
        vis = std::make_unique<emocap::Visualizer>("EventMoCap", intrinsics, marker);
        vis->logMarkers();
    }

    // --- CSV output ---
    std::ofstream csv(FLAGS_output);
    csv << "timestamp,x,y,z,rvec_x,rvec_y,rvec_z\n";

    // PnP diagnostics are deliberately separate from the historical pose CSV.
    // They capture estimator-stage evidence before optional temporal filtering.
    std::ofstream pnp_frame_diagnostics_csv;
    std::ofstream pnp_correspondence_diagnostics_csv;
    const bool pnp_diagnostics_enabled =
        !FLAGS_pnp_diagnostics_frame_output.empty() ||
        !FLAGS_pnp_diagnostics_correspondence_output.empty();
    if (!FLAGS_pnp_diagnostics_frame_output.empty()) {
        pnp_frame_diagnostics_csv.open(FLAGS_pnp_diagnostics_frame_output);
        if (!pnp_frame_diagnostics_csv) {
            LOG(ERROR) << "Cannot open PnP frame diagnostics output: "
                       << FLAGS_pnp_diagnostics_frame_output;
            return 1;
        }
        pnp_frame_diagnostics_csv
            << "frame_seq,timestamp_us,candidate_observation_count,"
            << "matched_correspondence_count,candidate_mode,candidate_solver,"
            << "geometry_coplanar,candidate_attempted,candidate_success,"
            << "candidate_reported_inlier_count,candidate_consensus_valid,"
            << "candidate_consensus_reprojection_valid,ransac_attempted,ransac_success,"
            << "ransac_reported_inlier_count,ransac_consensus_valid,"
            << "ransac_post_reprojection_valid,refinement_attempted,"
            << "refinement_success,final_pose_available,final_consensus_reprojection_valid\n";
    }
    if (!FLAGS_pnp_diagnostics_correspondence_output.empty()) {
        pnp_correspondence_diagnostics_csv.open(
            FLAGS_pnp_diagnostics_correspondence_output);
        if (!pnp_correspondence_diagnostics_csv) {
            LOG(ERROR) << "Cannot open PnP correspondence diagnostics output: "
                       << FLAGS_pnp_diagnostics_correspondence_output;
            return 1;
        }
        pnp_correspondence_diagnostics_csv
            << "frame_seq,timestamp_us,marker_id,observed_u_px,observed_v_px,"
            << "matched_to_marker,in_candidate_consensus,in_ransac,"
            << "candidate_reprojection_available,candidate_reprojection_error_px,"
            << "final_pose_available,"
            << "final_reprojection_available,final_reprojection_error_px\n";
    }
    std::uint64_t pnp_diagnostics_frame_sequence = 0;

    // The main CSV keeps its historical schema.  A second, optional file
    // records raw measurements and every filter decision for thesis analysis.
    std::ofstream temporal_diagnostics_csv;
    if (!FLAGS_temporal_filter_diagnostics_output.empty()) {
        temporal_diagnostics_csv.open(FLAGS_temporal_filter_diagnostics_output);
        if (!temporal_diagnostics_csv) {
            LOG(ERROR) << "Cannot open temporal-filter diagnostics output: "
                       << FLAGS_temporal_filter_diagnostics_output;
            return 1;
        }
        temporal_diagnostics_csv
            << "timestamp,temporal_filter_enabled,output_source,"
            << "raw_measurement_available,output_pose_available,"
            << "raw_x,raw_y,raw_z,raw_rvec_x,raw_rvec_y,raw_rvec_z,"
            << "output_x,output_y,output_z,output_rvec_x,output_rvec_y,output_rvec_z,"
            << "measurement_valid,initialized,reset,reinitialized_from_measurement,"
            << "translation_accepted,rotation_accepted,"
            << "dt_s,translation_innovation_mm,translation_mahalanobis2,"
            << "rotation_innovation_deg,consecutive_translation_rejections\n";
    }

    const auto write_optional_pose_csv = [&](const std::optional<emocap::Pose>& pose) {
        if (!pose) {
            temporal_diagnostics_csv << ",,,,,,";
            return;
        }
        const auto position = pose->position_world();
        temporal_diagnostics_csv
            << "," << position.x() << "," << position.y() << "," << position.z()
            << "," << pose->rvec.at<double>(0)
            << "," << pose->rvec.at<double>(1)
            << "," << pose->rvec.at<double>(2);
    };
    int pose_count = 0;

    // --- Observation callback (shared by all modes) ---
    auto on_observations = [&](emocap::Timestamp ts,
                               const std::vector<emocap::MarkerObservation>& obs) {
        // Log 2D observations
        if (vis) vis->logObservations(ts, obs);

        const std::uint64_t pnp_frame_seq = pnp_diagnostics_enabled
            ? pnp_diagnostics_frame_sequence++
            : 0;
        emocap::PoseEstimateDiagnostics pnp_diagnostics;
        const auto raw_result = estimator.estimate(
            ts, obs, pnp_diagnostics_enabled ? &pnp_diagnostics : nullptr);

        // Write estimator diagnostics before the optional temporal stage so a
        // failed measurement (nullopt) remains a visible frame-level record.
        if (pnp_frame_diagnostics_csv) {
            pnp_frame_diagnostics_csv
                << pnp_frame_seq << "," << pnp_diagnostics.timestamp << ","
                << pnp_diagnostics.candidate_observation_count << ","
                << pnp_diagnostics.matched_correspondence_count << ","
                << pnp_diagnostics.candidate_mode << ","
                << pnp_diagnostics.candidate_solver << ","
                << static_cast<int>(pnp_diagnostics.geometry_coplanar) << ","
                << static_cast<int>(pnp_diagnostics.candidate_attempted) << ","
                << static_cast<int>(pnp_diagnostics.candidate_success) << ","
                << pnp_diagnostics.candidate_reported_inlier_count << ","
                << static_cast<int>(pnp_diagnostics.candidate_consensus_valid) << ","
                << static_cast<int>(
                       pnp_diagnostics.candidate_consensus_reprojection_valid) << ","
                << static_cast<int>(pnp_diagnostics.ransac_attempted) << ","
                << static_cast<int>(pnp_diagnostics.ransac_success) << ","
                << pnp_diagnostics.ransac_reported_inlier_count << ","
                << static_cast<int>(pnp_diagnostics.ransac_consensus_valid) << ","
                << static_cast<int>(pnp_diagnostics.ransac_post_reprojection_valid) << ","
                << static_cast<int>(pnp_diagnostics.refinement_attempted) << ","
                << static_cast<int>(pnp_diagnostics.refinement_success) << ","
                << static_cast<int>(pnp_diagnostics.final_pose_available) << ","
                << static_cast<int>(
                       pnp_diagnostics.final_consensus_reprojection_valid) << "\n";
        }
        if (pnp_correspondence_diagnostics_csv) {
            for (const auto& correspondence : pnp_diagnostics.correspondences) {
                pnp_correspondence_diagnostics_csv
                    << pnp_frame_seq << "," << pnp_diagnostics.timestamp << ","
                    << correspondence.id << "," << correspondence.observed_uv.x << ","
                    << correspondence.observed_uv.y << ","
                    << static_cast<int>(correspondence.matched_to_marker) << ","
                    << static_cast<int>(correspondence.in_candidate_consensus) << ","
                    << static_cast<int>(correspondence.in_ransac) << ","
                    << static_cast<int>(correspondence.candidate_reprojection_available) << ",";
                if (correspondence.candidate_reprojection_available &&
                    std::isfinite(correspondence.candidate_reprojection_error_px)) {
                    pnp_correspondence_diagnostics_csv
                        << correspondence.candidate_reprojection_error_px;
                }
                pnp_correspondence_diagnostics_csv
                    << "," << static_cast<int>(pnp_diagnostics.final_pose_available) << ","
                    << static_cast<int>(correspondence.final_reprojection_available) << ",";
                if (correspondence.final_reprojection_available &&
                    std::isfinite(correspondence.final_reprojection_error_px)) {
                    pnp_correspondence_diagnostics_csv
                        << correspondence.final_reprojection_error_px;
                }
                pnp_correspondence_diagnostics_csv << "\n";
            }
        }

        std::optional<emocap::Pose> output_pose;
        emocap::TemporalPoseFilterDiagnostics temporal_diagnostics;
        std::string output_source = "none";

        if (raw_result) {
            if (FLAGS_temporal_filter) {
                const auto filtered = temporal_filter.update(*raw_result);
                output_pose = filtered.pose;
                temporal_diagnostics = filtered.diagnostics;
                output_source = "filtered_measurement";
            } else {
                output_pose = *raw_result;
                output_source = "raw_measurement";
                // Make an explicitly requested diagnostic trace readable
                // when this run is the unfiltered control condition.
                temporal_diagnostics.measurement_valid = true;
                temporal_diagnostics.translation_accepted = true;
                temporal_diagnostics.rotation_accepted = true;
            }
        } else if (FLAGS_temporal_filter) {
            // A failed four-LED RANSAC consensus is not a raw pose. The
            // filter may expose only a short, causal prediction from its
            // last accepted state; it is never fed back to PnP as a prior.
            output_pose = temporal_filter.predict(ts);
            temporal_diagnostics.initialized = temporal_filter.initialized();
            if (output_pose) output_source = "prediction";
        }

        if (temporal_diagnostics_csv) {
            temporal_diagnostics_csv
                << ts << "," << static_cast<int>(FLAGS_temporal_filter) << ","
                << output_source << ","
                << static_cast<int>(raw_result.has_value()) << ","
                << static_cast<int>(output_pose.has_value());
            write_optional_pose_csv(raw_result);
            write_optional_pose_csv(output_pose);
            temporal_diagnostics_csv
                << ","
                << static_cast<int>(temporal_diagnostics.measurement_valid) << ","
                << static_cast<int>(temporal_diagnostics.initialized) << ","
                << static_cast<int>(temporal_diagnostics.reset) << ","
                << static_cast<int>(temporal_diagnostics.reinitialized_from_measurement) << ","
                << static_cast<int>(temporal_diagnostics.translation_accepted) << ","
                << static_cast<int>(temporal_diagnostics.rotation_accepted) << ","
                << temporal_diagnostics.dt_s << ","
                << temporal_diagnostics.translation_innovation_mm << ","
                << temporal_diagnostics.translation_mahalanobis2 << ","
                << temporal_diagnostics.rotation_innovation_deg << ","
                << temporal_diagnostics.consecutive_translation_rejections << "\n";
        }

        // With filtering disabled, a failed robust-RANSAC measurement emits
        // no pose row. With filtering enabled, only a bounded prediction
        // from the independent output stage reaches this point.
        if (!output_pose) return;

        const auto pos = output_pose->position_world();
        csv << ts << "," << pos.x() << "," << pos.y() << "," << pos.z() << ","
            << output_pose->rvec.at<double>(0) << ","
            << output_pose->rvec.at<double>(1) << ","
            << output_pose->rvec.at<double>(2) << "\n";

        if (vis) vis->logEstimatedPose(*output_pose);
        ++pose_count;
    };

    // --- Run ---
    LOG(INFO) << "Mode: " << FLAGS_mode;

    if (mode == emocap::Mode::SIM) {
        emocap::SimConfig sim;
        sim.duration_s     = FLAGS_sim_duration;
        sim.rate_hz        = FLAGS_sim_rate_hz;
        sim.pixel_noise    = FLAGS_sim_pixel_noise;
        sim.circle_radius  = FLAGS_sim_circle_radius;
        sim.circle_height  = FLAGS_sim_circle_height;
        sim.circle_freq    = FLAGS_sim_circle_freq;
        sim.linear_speed   = FLAGS_sim_linear_speed;
        sim.linear_z       = FLAGS_sim_linear_z;
        sim.sin_amp_x      = FLAGS_sim_sin_amp_x;
        sim.sin_amp_y      = FLAGS_sim_sin_amp_y;
        sim.sin_amp_z      = FLAGS_sim_sin_amp_z;
        sim.sin_freq_x     = FLAGS_sim_sin_freq_x;
        sim.sin_freq_y     = FLAGS_sim_sin_freq_y;
        sim.sin_freq_z     = FLAGS_sim_sin_freq_z;
        sim.sin_base_z     = FLAGS_sim_sin_base_z;

        if (FLAGS_trajectory == "linear")          sim.traj_type = emocap::TrajectoryType::LINEAR;
        else if (FLAGS_trajectory == "sinusoidal") sim.traj_type = emocap::TrajectoryType::SINUSOIDAL;
        else                                       sim.traj_type = emocap::TrajectoryType::CIRCULAR;

        emocap::SimulationSource source(marker, intrinsics, sim);

        source.setGroundTruthCallback([&](emocap::Timestamp ts, const emocap::Pose& gt) {
            if (vis) vis->logGroundTruth(gt);
        });

        source.start(on_observations);

    }
#ifdef HAS_METAVISION
    else if (mode == emocap::Mode::FILE) {
        if (FLAGS_input.empty()) {
            LOG(ERROR) << "-input is required for file mode";
            return 1;
        }

        // Build TrackerParams from gflags
        emocap::TrackerParams tp;
        tp.num_bits             = FLAGS_det_num_bits;
        tp.base_period_us       = static_cast<std::uint32_t>(FLAGS_det_base_period_us);
        tp.det_tolerance        = static_cast<float>(FLAGS_det_tolerance);
        tp.tracker_radius       = static_cast<float>(FLAGS_trk_radius);
        tp.monitoring_freq_hz   = static_cast<float>(FLAGS_trk_monitoring_freq_hz);
        tp.inactivity_period_us = FLAGS_trk_inactivity_period_us;
        tp.alpha_pos            = static_cast<float>(FLAGS_trk_alpha_pos);
        tp.distance_pct         = static_cast<float>(FLAGS_trk_distance_pct);
        tp.max_samples_per_id   = static_cast<size_t>(FLAGS_obs_max_samples);
        tp.max_obs_age_us       = FLAGS_obs_max_age_us;
        tp.delta_n_events       = static_cast<size_t>(FLAGS_obs_delta_n_events);

        emocap::EventFileSource::Params params;
        params.event_file_path    = FLAGS_input;
        params.camera_config_path = FLAGS_camera_config;
        params.marker             = marker;
        params.realtime_playback  = FLAGS_realtime;
        params.tracker_params     = tp;

        emocap::EventFileSource source(params);
        source.start(on_observations);

    } else {  // LIVE
        emocap::TrackerParams tp;
        tp.num_bits             = FLAGS_det_num_bits;
        tp.base_period_us       = static_cast<std::uint32_t>(FLAGS_det_base_period_us);
        tp.det_tolerance        = static_cast<float>(FLAGS_det_tolerance);
        tp.tracker_radius       = static_cast<float>(FLAGS_trk_radius);
        tp.monitoring_freq_hz   = static_cast<float>(FLAGS_trk_monitoring_freq_hz);
        tp.inactivity_period_us = FLAGS_trk_inactivity_period_us;
        tp.alpha_pos            = static_cast<float>(FLAGS_trk_alpha_pos);
        tp.distance_pct         = static_cast<float>(FLAGS_trk_distance_pct);
        tp.max_samples_per_id   = static_cast<size_t>(FLAGS_obs_max_samples);
        tp.max_obs_age_us       = FLAGS_obs_max_age_us;
        tp.delta_n_events       = static_cast<size_t>(FLAGS_obs_delta_n_events);

        emocap::LiveCameraSource::Params params;
        params.camera_config_path = FLAGS_camera_config;
        params.marker             = marker;
        params.tracker_params     = tp;

        emocap::LiveCameraSource source(params);
        source.start(on_observations);
    }
#endif  // HAS_METAVISION

    csv.close();
    temporal_diagnostics_csv.close();
    pnp_frame_diagnostics_csv.close();
    pnp_correspondence_diagnostics_csv.close();
    LOG(INFO) << pose_count << " poses estimated -> " << FLAGS_output;

    if (vis) vis->block(300.0);
    return 0;
}
