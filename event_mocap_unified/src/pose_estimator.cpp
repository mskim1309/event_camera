// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#include "event_mocap/pose_estimator.hpp"

#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>


namespace {

// All geometric candidate and final residual checks use the same strict gate.
constexpr std::size_t kMinPnPCorrespondences = 4;
constexpr double kRansacReprojectionErrorPixels = 4.0;

bool makePoseColumn64(const cv::Mat& input, cv::Mat& output) {
    if (input.empty() || input.total() != 3) return false;
    try {
        input.reshape(1, 3).convertTo(output, CV_64F);
    } catch (const cv::Exception&) {
        return false;
    }
    return output.rows == 3 && output.cols == 1 && cv::checkRange(output);
}

bool poseIsFinite(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat normalized_rvec, normalized_tvec;
    return makePoseColumn64(rvec, normalized_rvec) &&
           makePoseColumn64(tvec, normalized_tvec);
}

bool computeReprojectionErrors(const std::vector<cv::Point3f>& pts3d,
                               const std::vector<cv::Point2f>& pts2d,
                               const cv::Mat& camera_matrix,
                               const cv::Mat& distortion,
                               const cv::Mat& rvec,
                               const cv::Mat& tvec,
                               std::vector<double>& errors) {
    errors.clear();
    if (pts3d.size() != pts2d.size() || pts3d.empty() ||
        !poseIsFinite(rvec, tvec)) {
        return false;
    }

    std::vector<cv::Point2f> reprojected;
    try {
        cv::projectPoints(pts3d, rvec, tvec, camera_matrix, distortion, reprojected);
    } catch (const cv::Exception&) {
        return false;
    }
    if (reprojected.size() != pts2d.size()) return false;

    errors.reserve(pts2d.size());
    for (std::size_t i = 0; i < pts2d.size(); ++i) {
        const double error_pixels = cv::norm(reprojected[i] - pts2d[i]);
        if (!std::isfinite(error_pixels)) {
            errors.clear();
            return false;
        }
        errors.push_back(error_pixels);
    }
    return true;
}

bool reprojectsWithinThreshold(const std::vector<cv::Point3f>& pts3d,
                               const std::vector<cv::Point2f>& pts2d,
                               const cv::Mat& camera_matrix,
                               const cv::Mat& distortion,
                               const cv::Mat& rvec,
                               const cv::Mat& tvec) {
    if (pts3d.size() < kMinPnPCorrespondences) return false;
    std::vector<double> errors;
    if (!computeReprojectionErrors(pts3d, pts2d, camera_matrix, distortion,
                                   rvec, tvec, errors)) {
        return false;
    }
    return std::all_of(errors.begin(), errors.end(), [](double error_pixels) {
        return error_pixels <= kRansacReprojectionErrorPixels;
    });
}

bool allPointsHavePositiveDepth(const std::vector<cv::Point3f>& pts3d,
                                const cv::Mat& rvec,
                                const cv::Mat& tvec) {
    if (pts3d.empty()) return false;
    cv::Mat normalized_rvec, normalized_tvec;
    if (!makePoseColumn64(rvec, normalized_rvec) ||
        !makePoseColumn64(tvec, normalized_tvec)) {
        return false;
    }

    cv::Mat R;
    try {
        cv::Rodrigues(normalized_rvec, R);
    } catch (const cv::Exception&) {
        return false;
    }
    if (R.rows != 3 || R.cols != 3 || !cv::checkRange(R)) return false;

    for (const auto& point : pts3d) {
        const double depth = R.at<double>(2, 0) * point.x +
                             R.at<double>(2, 1) * point.y +
                             R.at<double>(2, 2) * point.z +
                             normalized_tvec.at<double>(2, 0);
        if (!std::isfinite(depth) || depth <= 1e-6) return false;
    }
    return true;
}

bool pointsAreCoplanar(const std::vector<cv::Point3f>& pts3d) {
    if (pts3d.size() < kMinPnPCorrespondences) return false;

    cv::Point3d centroid(0.0, 0.0, 0.0);
    for (const auto& point : pts3d) {
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
    }
    const double n = static_cast<double>(pts3d.size());
    centroid.x /= n;
    centroid.y /= n;
    centroid.z /= n;

    cv::Mat centered(static_cast<int>(pts3d.size()), 3, CV_64F);
    for (std::size_t i = 0; i < pts3d.size(); ++i) {
        centered.at<double>(static_cast<int>(i), 0) = pts3d[i].x - centroid.x;
        centered.at<double>(static_cast<int>(i), 1) = pts3d[i].y - centroid.y;
        centered.at<double>(static_cast<int>(i), 2) = pts3d[i].z - centroid.z;
    }

    cv::Mat singular_values;
    try {
        cv::SVD::compute(centered, singular_values, cv::noArray(), cv::noArray(),
                         cv::SVD::NO_UV);
    } catch (const cv::Exception&) {
        return false;
    }
    if (singular_values.total() < 3) return false;
    const double largest = singular_values.at<double>(0);
    const double smallest = singular_values.at<double>(2);
    return std::isfinite(largest) && std::isfinite(smallest) && largest > 1e-9 &&
           smallest <= std::max(1e-9, largest * 1e-6);
}

// OpenCV reports RANSAC inlier indices with respect to the correspondence
// vectors passed to solvePnPRansac. Keep those three vectors synchronized so
// that every downstream optimization sees the same robust consensus set.
bool selectConsensusIndices(const std::vector<int>& inliers,
                            const std::vector<cv::Point3f>& pts3d,
                            const std::vector<cv::Point2f>& pts2d,
                            const std::vector<std::uint32_t>& ids,
                            std::vector<cv::Point3f>& selected_pts3d,
                            std::vector<cv::Point2f>& selected_pts2d,
                            std::vector<std::uint32_t>& selected_ids) {
    if (pts3d.size() != pts2d.size() || pts3d.size() != ids.size() ||
        inliers.size() < kMinPnPCorrespondences) {
        return false;
    }

    // Do not silently accept malformed RANSAC output: duplicated or
    // out-of-range indices would otherwise give Ceres an inconsistent set.
    std::vector<bool> seen(pts3d.size(), false);
    selected_pts3d.clear();
    selected_pts2d.clear();
    selected_ids.clear();
    selected_pts3d.reserve(inliers.size());
    selected_pts2d.reserve(inliers.size());
    selected_ids.reserve(inliers.size());

    for (int inlier_index : inliers) {
        if (inlier_index < 0 ||
            static_cast<std::size_t>(inlier_index) >= pts3d.size()) {
            return false;
        }
        const std::size_t index = static_cast<std::size_t>(inlier_index);
        if (seen[index]) return false;
        seen[index] = true;
        selected_pts3d.push_back(pts3d[index]);
        selected_pts2d.push_back(pts2d[index]);
        selected_ids.push_back(ids[index]);
    }

    return selected_pts3d.size() >= kMinPnPCorrespondences;
}

struct PnPCandidate {
    std::string solver = "none";
    bool attempted = false;
    bool success = false;
    bool uses_ransac = false;
    cv::Mat rvec;
    cv::Mat tvec;
    std::vector<int> inliers;
};

PnPCandidate solveRansacCandidate(const std::vector<cv::Point3f>& pts3d,
                                  const std::vector<cv::Point2f>& pts2d,
                                  const cv::Mat& camera_matrix,
                                  const cv::Mat& distortion,
                                  int flag,
                                  const char* solver_name) {
    PnPCandidate candidate;
    candidate.solver = solver_name;
    candidate.attempted = true;
    candidate.uses_ransac = true;

    cv::Mat rvec, tvec;
    bool solved = false;
    try {
        solved = cv::solvePnPRansac(pts3d, pts2d, camera_matrix, distortion,
                                    rvec, tvec, false, 100,
                                    kRansacReprojectionErrorPixels, 0.99,
                                    candidate.inliers, flag);
    } catch (const cv::Exception&) {
        return candidate;
    }

    if (solved && makePoseColumn64(rvec, candidate.rvec) &&
        makePoseColumn64(tvec, candidate.tvec)) {
        candidate.success = true;
    }
    return candidate;
}

// IPPE can return up to two planar-pose candidates. Select only a cheiral
// candidate, then use its all-point residual (rather than solution order) to
// choose the seed. IPPE_SQUARE is deliberately never used here.
PnPCandidate solveIppeCandidate(const std::vector<cv::Point3f>& pts3d,
                                const std::vector<cv::Point2f>& pts2d,
                                const cv::Mat& camera_matrix,
                                const cv::Mat& distortion) {
    PnPCandidate candidate;
    candidate.solver = "IPPE";
    candidate.attempted = true;

    std::vector<cv::Mat> rvecs, tvecs;
    int solution_count = 0;
    try {
        solution_count = cv::solvePnPGeneric(pts3d, pts2d, camera_matrix, distortion,
                                             rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception&) {
        return candidate;
    }

    double best_squared_error = std::numeric_limits<double>::infinity();
    const int available = std::min({solution_count, static_cast<int>(rvecs.size()),
                                    static_cast<int>(tvecs.size())});
    for (int i = 0; i < available; ++i) {
        cv::Mat rvec, tvec;
        if (!makePoseColumn64(rvecs[static_cast<std::size_t>(i)], rvec) ||
            !makePoseColumn64(tvecs[static_cast<std::size_t>(i)], tvec) ||
            !allPointsHavePositiveDepth(pts3d, rvec, tvec)) {
            continue;
        }

        std::vector<double> errors;
        if (!computeReprojectionErrors(pts3d, pts2d, camera_matrix, distortion,
                                       rvec, tvec, errors)) {
            continue;
        }
        double squared_error = 0.0;
        for (double error : errors) squared_error += error * error;
        if (squared_error < best_squared_error) {
            best_squared_error = squared_error;
            candidate.rvec = rvec;
            candidate.tvec = tvec;
            candidate.success = true;
        }
    }
    if (candidate.success) {
        candidate.inliers.resize(pts3d.size());
        for (std::size_t i = 0; i < candidate.inliers.size(); ++i)
            candidate.inliers[i] = static_cast<int>(i);
    }
    return candidate;
}

}  // namespace
namespace emocap {

// --- Ceres reprojection error with radial+tangential distortion ---
struct ReprojectionError {
    ReprojectionError(const cv::Point3f& p3d, const cv::Point2f& obs,
                      double fx, double fy, double cx, double cy,
                      const std::vector<double>& dist)
        : X_(p3d.x), Y_(p3d.y), Z_(p3d.z),
          u_(obs.x), v_(obs.y),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy), dist_(dist) {}

    template <typename T>
    bool operator()(const T* const angle_axis, const T* const t, T* residuals) const {
        T point[3] = {T(X_), T(Y_), T(Z_)};
        T p[3];
        ceres::AngleAxisRotatePoint(angle_axis, point, p);
        p[0] += t[0]; p[1] += t[1]; p[2] += t[2];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T r2 = xp * xp + yp * yp;

        // Radial distortion
        T radial = T(1.0);
        int n = static_cast<int>(dist_.size());
        if (n >= 1) radial += T(dist_[0]) * r2;
        if (n >= 2) radial += T(dist_[1]) * r2 * r2;
        if (n >= 5) radial += T(dist_[4]) * r2 * r2 * r2;

        T x_dist = xp * radial;
        T y_dist = yp * radial;

        // Tangential distortion
        if (n >= 3) {
            T p1 = T(dist_[2]);
            T p2 = (n >= 4) ? T(dist_[3]) : T(0.0);
            x_dist += T(2.0) * p1 * xp * yp + p2 * (r2 + T(2.0) * xp * xp);
            y_dist += p1 * (r2 + T(2.0) * yp * yp) + T(2.0) * p2 * xp * yp;
        }

        residuals[0] = T(fx_) * x_dist + T(cx_) - T(u_);
        residuals[1] = T(fy_) * y_dist + T(cy_) - T(v_);
        return true;
    }

    static ceres::CostFunction* Create(const cv::Point3f& p3d, const cv::Point2f& obs,
                                        double fx, double fy, double cx, double cy,
                                        const std::vector<double>& dist) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
            new ReprojectionError(p3d, obs, fx, fy, cx, cy, dist));
    }

    double X_, Y_, Z_, u_, v_, fx_, fy_, cx_, cy_;
    std::vector<double> dist_;
};

// --- PoseEstimator ---

std::optional<PnPCandidateMode> parsePnPCandidateMode(const std::string& value) {
    if (value == "auto") return PnPCandidateMode::AUTO;
    if (value == "legacy_iterative_ransac") {
        return PnPCandidateMode::LEGACY_ITERATIVE_RANSAC;
    }
    return std::nullopt;
}

const char* pnpCandidateModeName(PnPCandidateMode mode) {
    switch (mode) {
    case PnPCandidateMode::AUTO:
        return "auto";
    case PnPCandidateMode::LEGACY_ITERATIVE_RANSAC:
        return "legacy_iterative_ransac";
    }
    return "unknown";
}

PoseEstimator::PoseEstimator(const ActiveMarker& marker,
                             const CameraIntrinsics& intrinsics,
                             int ba_window_size,
                             PnPCandidateMode candidate_mode)
    : marker_(marker),
      ba_window_size_(std::max(1, ba_window_size)),
      candidate_mode_(candidate_mode)
{
    K_ = intrinsics.K();
    D_ = intrinsics.D();
    fx_ = intrinsics.fx; fy_ = intrinsics.fy;
    cx_ = intrinsics.cx; cy_ = intrinsics.cy;
    dist_coeffs_ = intrinsics.dist_coeffs;
}

void PoseEstimator::reset() {
    ba_window_.clear();
}

std::optional<Pose> PoseEstimator::estimate(Timestamp ts,
                                             const std::vector<MarkerObservation>& obs,
                                             PoseEstimateDiagnostics* diagnostics)
{
    if (diagnostics) {
        *diagnostics = PoseEstimateDiagnostics{};
        diagnostics->timestamp = ts;
        diagnostics->candidate_observation_count = obs.size();
        diagnostics->candidate_mode = pnpCandidateModeName(candidate_mode_);
        diagnostics->correspondences.reserve(obs.size());
    }

    // Build 3D-2D correspondences
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<std::uint32_t> ids;
    std::vector<std::size_t> diagnostic_indices;

    for (const auto& o : obs) {
        std::size_t diagnostic_index = 0;
        if (diagnostics) {
            PoseCorrespondenceDiagnostics correspondence;
            correspondence.id = o.id;
            correspondence.observed_uv = o.pt2d;
            diagnostics->correspondences.push_back(correspondence);
            diagnostic_index = diagnostics->correspondences.size() - 1;
        }

        auto it = marker_.find(o.id);
        if (it == marker_.end()) continue;

        pts3d.push_back(it->second);
        pts2d.push_back(o.pt2d);
        ids.push_back(o.id);
        if (diagnostics) {
            diagnostics->correspondences[diagnostic_index].matched_to_marker = true;
            diagnostic_indices.push_back(diagnostic_index);
        }
    }

    if (diagnostics)
        diagnostics->matched_correspondence_count = pts3d.size();

    if (pts3d.size() < 4) return std::nullopt;

    // `auto` uses IPPE only for the exactly-four, coplanar case. It avoids
    // IPPE_SQUARE because this marker is not constrained to its square-point
    // ordering. All other geometries use OpenCV's SQPNP RANSAC flag. OpenCV
    // may internally choose a P3P/EPnP minimal sampler, so diagnostics name
    // this accurately as a SQPNP-flagged RANSAC call rather than a minimal
    // SQPNP solver.
    const bool geometry_coplanar = pointsAreCoplanar(pts3d);
    const bool use_ippe = candidate_mode_ == PnPCandidateMode::AUTO &&
                          pts3d.size() == kMinPnPCorrespondences && geometry_coplanar;
    PnPCandidate candidate;
    if (candidate_mode_ == PnPCandidateMode::LEGACY_ITERATIVE_RANSAC) {
        candidate = solveRansacCandidate(pts3d, pts2d, K_, D_, cv::SOLVEPNP_ITERATIVE,
                                         "RANSAC(flag=ITERATIVE; legacy)");
    } else if (use_ippe) {
        candidate = solveIppeCandidate(pts3d, pts2d, K_, D_);
    } else {
        candidate = solveRansacCandidate(pts3d, pts2d, K_, D_, cv::SOLVEPNP_SQPNP,
                                         "RANSAC(flag=SQPNP)");
    }

    if (diagnostics) {
        diagnostics->candidate_solver = candidate.solver;
        diagnostics->geometry_coplanar = geometry_coplanar;
        diagnostics->candidate_attempted = candidate.attempted;
        diagnostics->candidate_success = candidate.success;
        diagnostics->candidate_reported_inlier_count = candidate.inliers.size();
        diagnostics->ransac_attempted = candidate.uses_ransac && candidate.attempted;
        diagnostics->ransac_success = candidate.uses_ransac && candidate.success;
        diagnostics->ransac_reported_inlier_count =
            candidate.uses_ransac ? candidate.inliers.size() : 0;
        for (int inlier_index : candidate.inliers) {
            if (inlier_index < 0) continue;
            const auto correspondence_index = static_cast<std::size_t>(inlier_index);
            if (correspondence_index >= diagnostic_indices.size()) continue;
            auto& correspondence =
                diagnostics->correspondences[diagnostic_indices[correspondence_index]];
            correspondence.in_candidate_consensus = true;
            correspondence.in_ransac = candidate.uses_ransac;
        }
    }

    // Keep the candidate residual even when a later consensus/final gate
    // rejects it. This exposes the reported-RANSAC-success versus strict
    // consensus-reprojection failure that occurs for near-frontal
    // configurations. The per-correspondence candidate residual below still
    // covers every matched observation, not only the selected consensus.
    if (diagnostics && candidate.success) {
        std::vector<double> candidate_errors;
        if (computeReprojectionErrors(pts3d, pts2d, K_, D_, candidate.rvec,
                                      candidate.tvec, candidate_errors)) {
            for (std::size_t i = 0; i < candidate_errors.size() &&
                                    i < diagnostic_indices.size(); ++i) {
                auto& correspondence =
                    diagnostics->correspondences[diagnostic_indices[i]];
                correspondence.candidate_reprojection_available = true;
                correspondence.candidate_reprojection_error_px = candidate_errors[i];
            }
        }
    }

    std::vector<cv::Point3f> inlier_pts3d;
    std::vector<cv::Point2f> inlier_pts2d;
    std::vector<std::uint32_t> inlier_ids;
    const bool candidate_consensus_valid =
        candidate.success && selectConsensusIndices(candidate.inliers, pts3d, pts2d, ids,
                                                     inlier_pts3d, inlier_pts2d, inlier_ids);
    const bool candidate_consensus_reprojection_valid =
        candidate_consensus_valid &&
        allPointsHavePositiveDepth(inlier_pts3d, candidate.rvec, candidate.tvec) &&
        reprojectsWithinThreshold(inlier_pts3d, inlier_pts2d, K_, D_,
                                  candidate.rvec, candidate.tvec);

    if (diagnostics) {
        diagnostics->candidate_consensus_valid = candidate_consensus_valid;
        diagnostics->candidate_consensus_reprojection_valid =
            candidate_consensus_reprojection_valid;
        if (candidate.uses_ransac) {
            diagnostics->ransac_consensus_valid = candidate_consensus_valid;
            diagnostics->ransac_post_reprojection_valid = candidate_consensus_reprojection_valid;
        }
    }

    if (!candidate_consensus_valid || !candidate_consensus_reprojection_valid) {
        return std::nullopt;
    }

    const std::vector<cv::Point3f>& refine_pts3d = inlier_pts3d;
    const std::vector<cv::Point2f>& refine_pts2d = inlier_pts2d;
    const std::vector<std::uint32_t>& refine_ids = inlier_ids;

    // Transactionally add only the candidate consensus. It is rolled back if
    // iterative/Ceres refinement fails a final cheirality/residual check, so
    // a rejected frame cannot poison the BA seed window.
    FrameObservation fr;
    fr.ts = ts;
    fr.ids = refine_ids;
    fr.pts2d = refine_pts2d;
    fr.rvec_seed = candidate.rvec.clone();
    fr.tvec_seed = candidate.tvec.clone();
    std::optional<FrameObservation> evicted;
    if (static_cast<int>(ba_window_.size()) >= ba_window_size_) {
        evicted = std::move(ba_window_.front());
        ba_window_.pop_front();
    }
    ba_window_.push_back(std::move(fr));
    const auto rollback_current_frame = [&]() {
        ba_window_.pop_back();
        if (evicted) ba_window_.push_front(std::move(*evicted));
    };

    // The geometry-aware candidate is deliberately the iterative seed. No
    // previous-pose or unseeded ITERATIVE solve is permitted in this path.
    cv::Mat rvec_ref = candidate.rvec.clone(), tvec_ref = candidate.tvec.clone();
    const bool refinement_success =
        solvePnPAndRefine(refine_pts3d, refine_pts2d, rvec_ref, tvec_ref);
    if (diagnostics) {
        diagnostics->refinement_attempted = true;
        diagnostics->refinement_success = refinement_success;
    }
    if (!refinement_success) {
        // The candidate itself has already passed the strict consensus gate.
        rvec_ref = candidate.rvec.clone();
        tvec_ref = candidate.tvec.clone();
    }

    const bool final_consensus_reprojection_valid =
        poseIsFinite(rvec_ref, tvec_ref) &&
        allPointsHavePositiveDepth(refine_pts3d, rvec_ref, tvec_ref) &&
        reprojectsWithinThreshold(refine_pts3d, refine_pts2d, K_, D_, rvec_ref, tvec_ref);
    if (diagnostics) {
        diagnostics->final_consensus_reprojection_valid = final_consensus_reprojection_valid;
    }
    if (!final_consensus_reprojection_valid) {
        rollback_current_frame();
        return std::nullopt;
    }

    // Commit only an accepted final pose as the seed carried into a later BA
    // window. The rejection branch above has already restored the prior
    // window, so this never stores a failed refinement.
    ba_window_.back().rvec_seed = rvec_ref.clone();
    ba_window_.back().tvec_seed = tvec_ref.clone();

    if (diagnostics) {
        diagnostics->final_pose_available = true;
        std::vector<double> final_errors;
        if (computeReprojectionErrors(pts3d, pts2d, K_, D_, rvec_ref, tvec_ref,
                                      final_errors)) {
            for (std::size_t i = 0; i < final_errors.size() &&
                                    i < diagnostic_indices.size(); ++i) {
                auto& correspondence =
                    diagnostics->correspondences[diagnostic_indices[i]];
                correspondence.final_reprojection_available = true;
                correspondence.final_reprojection_error_px = final_errors[i];
            }
        }
    }

    Pose pose;
    pose.ts = ts;
    pose.rvec = rvec_ref.clone();
    pose.tvec = tvec_ref.clone();
    return pose;
}

bool PoseEstimator::solvePnPAndRefine(const std::vector<cv::Point3f>& pts3d,
                                       const std::vector<cv::Point2f>& pts2d,
                                       cv::Mat& rvec, cv::Mat& tvec)
{
    if (pts3d.size() < kMinPnPCorrespondences || pts3d.size() != pts2d.size()) {
        return false;
    }

    // Keep the geometry-aware candidate as the ITERATIVE seed. In particular,
    // do not substitute a previous pose or silently invoke an unseeded solve:
    // that was the instability observed for near-frontal non-planar markers.
    cv::Mat normalized_rvec, normalized_tvec;
    if (!makePoseColumn64(rvec, normalized_rvec) ||
        !makePoseColumn64(tvec, normalized_tvec)) {
        return false;
    }
    rvec = normalized_rvec;
    tvec = normalized_tvec;
    bool iterative_ok = false;
    try {
        iterative_ok = cv::solvePnP(pts3d, pts2d, K_, D_, rvec, tvec,
                                    true, cv::SOLVEPNP_ITERATIVE);
    } catch (const cv::Exception&) {
        return false;
    }
    if (!iterative_ok || !makePoseColumn64(rvec, normalized_rvec) ||
        !makePoseColumn64(tvec, normalized_tvec)) {
        return false;
    }
    rvec = normalized_rvec;
    tvec = normalized_tvec;

    // BA must inherit the candidate-seeded estimate for every stored frame;
    // runSlidingWindowBA never calls unseeded ITERATIVE initialization.
    if (!ba_window_.empty()) {
        ba_window_.back().rvec_seed = rvec.clone();
        ba_window_.back().tvec_seed = tvec.clone();
    }

    // Sliding-window BA if enough frames
    if (ba_window_.size() > 1) {
        cv::Mat rvec_ba, tvec_ba;
        if (runSlidingWindowBA(rvec_ba, tvec_ba) && poseIsFinite(rvec_ba, tvec_ba)) {
            rvec = rvec_ba; tvec = tvec_ba;
            return true;
        }
    }

    // Single-pose Ceres refinement
    double aa[3] = {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
    double tr[3] = {tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};

    ceres::Problem problem;
    problem.AddParameterBlock(aa, 3);
    problem.AddParameterBlock(tr, 3);
    for (size_t i = 0; i < pts3d.size(); ++i) {
        problem.AddResidualBlock(
            ReprojectionError::Create(pts3d[i], pts2d[i], fx_, fy_, cx_, cy_, dist_coeffs_),
            new ceres::HuberLoss(1.0), aa, tr);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.num_threads = 4;
    opts.max_num_iterations = 50;
    opts.function_tolerance = 1e-8;
    opts.parameter_tolerance = 1e-8;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    if (!summary.IsSolutionUsable()) return false;

    cv::Mat refined_rvec(3, 1, CV_64F), refined_tvec(3, 1, CV_64F);
    for (int k = 0; k < 3; ++k) {
        refined_rvec.at<double>(k, 0) = aa[k];
        refined_tvec.at<double>(k, 0) = tr[k];
    }
    if (!poseIsFinite(refined_rvec, refined_tvec)) return false;
    rvec = refined_rvec;
    tvec = refined_tvec;
    return true;
}

bool PoseEstimator::runSlidingWindowBA(cv::Mat& out_rvec, cv::Mat& out_tvec)
{
    if (ba_window_.empty()) return false;

    const int W = static_cast<int>(ba_window_.size());
    std::vector<double> aa_vec(W * 3), tr_vec(W * 3);

    // Each stored frame entered this window only after its geometry-aware
    // candidate passed the strict gate. Use that candidate-seeded iterative
    // estimate directly; do not reintroduce unseeded ITERATIVE PnP here.
    for (int i = 0; i < W; ++i) {
        const auto& fr = ba_window_[i];
        cv::Mat rv, tv;
        if (fr.ids.size() < kMinPnPCorrespondences ||
            fr.ids.size() != fr.pts2d.size() ||
            !makePoseColumn64(fr.rvec_seed, rv) ||
            !makePoseColumn64(fr.tvec_seed, tv)) {
            return false;
        }

        for (int k = 0; k < 3; ++k) {
            aa_vec[3 * i + k] = rv.at<double>(k);
            tr_vec[3 * i + k] = tv.at<double>(k);
        }
    }

    // Build Ceres problem
    ceres::Problem problem;
    for (int i = 0; i < W; ++i) {
        problem.AddParameterBlock(&aa_vec[3 * i], 3);
        problem.AddParameterBlock(&tr_vec[3 * i], 3);
    }

    for (int i = 0; i < W; ++i) {
        const auto& fr = ba_window_[i];
        for (size_t j = 0; j < fr.ids.size(); ++j) {
            auto it = marker_.find(fr.ids[j]);
            if (it == marker_.end()) continue;
            problem.AddResidualBlock(
                ReprojectionError::Create(it->second, fr.pts2d[j], fx_, fy_, cx_, cy_, dist_coeffs_),
                new ceres::HuberLoss(1.0), &aa_vec[3 * i], &tr_vec[3 * i]);
        }
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.num_threads = 4;
    opts.max_num_iterations = 200;
    opts.function_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    if (!summary.IsSolutionUsable()) return false;

    const int last = W - 1;
    cv::Mat refined_rvec(3, 1, CV_64F), refined_tvec(3, 1, CV_64F);
    for (int k = 0; k < 3; ++k) {
        refined_rvec.at<double>(k, 0) = aa_vec[3 * last + k];
        refined_tvec.at<double>(k, 0) = tr_vec[3 * last + k];
    }
    if (!poseIsFinite(refined_rvec, refined_tvec)) return false;
    out_rvec = refined_rvec;
    out_tvec = refined_tvec;
    return true;
}

}  // namespace emocap
