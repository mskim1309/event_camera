// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#pragma once

#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include "event_mocap/types.hpp"

namespace emocap {

// `auto` chooses a candidate solver from the matched marker geometry.
// `legacy_iterative_ransac` is retained as an ablation of the former
// candidate-stage solver; it still uses the current inlier and residual gates.
enum class PnPCandidateMode {
    AUTO,
    LEGACY_ITERATIVE_RANSAC,
};

std::optional<PnPCandidateMode> parsePnPCandidateMode(const std::string& value);
const char* pnpCandidateModeName(PnPCandidateMode mode);

// Per-input-observation details from one PnP attempt. `matched_to_marker`
// distinguishes an observed decoded ID from one that has a known 3D marker
// coordinate. The candidate residual may expose a rejected candidate;
// the final residual is populated only if a final pose is available.
struct PoseCorrespondenceDiagnostics {
    std::uint32_t id = 0;
    cv::Point2f observed_uv{};
    bool matched_to_marker = false;
    bool in_candidate_consensus = false;
    bool in_ransac = false;
    bool candidate_reprojection_available = false;
    double candidate_reprojection_error_px = std::numeric_limits<double>::quiet_NaN();
    bool final_reprojection_available = false;
    double final_reprojection_error_px = std::numeric_limits<double>::quiet_NaN();
};

// Optional, measurement-stage diagnostics. These fields intentionally refer
// only to the PnP estimator, before any temporal output filter is applied.
// The `*_consensus_reprojection_valid` flags describe the selected robust
// consensus/inlier set; per-correspondence residuals remain available for all
// matched marker observations whenever their corresponding pose exists.
struct PoseEstimateDiagnostics {
    Timestamp timestamp = 0;
    std::size_t candidate_observation_count = 0;
    std::size_t matched_correspondence_count = 0;
    std::string candidate_mode = "unknown";
    std::string candidate_solver = "none";
    bool geometry_coplanar = false;
    bool candidate_attempted = false;
    bool candidate_success = false;
    std::size_t candidate_reported_inlier_count = 0;
    bool candidate_consensus_valid = false;
    bool candidate_consensus_reprojection_valid = false;
    bool ransac_attempted = false;
    bool ransac_success = false;
    std::size_t ransac_reported_inlier_count = 0;
    bool ransac_consensus_valid = false;
    bool ransac_post_reprojection_valid = false;
    bool refinement_attempted = false;
    bool refinement_success = false;
    bool final_pose_available = false;
    bool final_consensus_reprojection_valid = false;
    std::vector<PoseCorrespondenceDiagnostics> correspondences;
};

class PoseEstimator {
public:
    PoseEstimator(const ActiveMarker& marker,
                  const CameraIntrinsics& intrinsics,
                  int ba_window_size = 5,
                  PnPCandidateMode candidate_mode = PnPCandidateMode::AUTO);

    // Estimate pose from current 2D observations. Returns nullopt on failure.
    std::optional<Pose> estimate(Timestamp ts,
                                 const std::vector<MarkerObservation>& obs,
                                 PoseEstimateDiagnostics* diagnostics = nullptr);

    void reset();

private:
    bool runSlidingWindowBA(cv::Mat& out_rvec, cv::Mat& out_tvec);
    bool solvePnPAndRefine(const std::vector<cv::Point3f>& pts3d,
                           const std::vector<cv::Point2f>& pts2d,
                           cv::Mat& rvec, cv::Mat& tvec);

    ActiveMarker marker_;
    cv::Mat K_, D_;
    double fx_, fy_, cx_, cy_;
    std::vector<double> dist_coeffs_;

    int ba_window_size_;
    PnPCandidateMode candidate_mode_;
    std::deque<FrameObservation> ba_window_;
};

}  // namespace emocap
