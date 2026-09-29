// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Causal robust temporal filtering for the pose measurements produced by
// PoseEstimator.  This is deliberately header-only so the core simulator can
// use it without changing the project's CMake source list.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>

#include "event_mocap/types.hpp"

namespace emocap {

// Units follow the rest of this project: positions are millimetres, time is
// microseconds at the API boundary, and angular quantities are degrees where
// explicitly named *_deg.
struct TemporalPoseFilterConfig {
    // Translation state is [camera centre in world, velocity in world].
    double process_accel_std_mm_s2 = 10000.0;
    double measurement_std_mm = 15.0;
    double gate_chi2 = 16.27;       // chi-square(3), approximately 99.9 %
    double hard_gate_chi2 = 81.0;   // reject a clearly implausible update

    // Rotation is kept on SO(3); Rodrigues vectors are never averaged.
    // These defaults cover the full shortest-path SO(3) innovation range
    // (0--180 degrees), so rotation gating is intentionally inactive until
    // a dataset-specific threshold is validated.
    double orientation_alpha = 0.35;
    double orientation_beta = 0.025;
    double orientation_gate_deg = 180.0;
    double orientation_hard_gate_deg = 360.0;

    // Reinitialise rather than bridge discontinuous file/live timestamps.
    Timestamp reset_gap_us = 250000;
    int max_consecutive_translation_rejections = 4;
};

struct TemporalPoseFilterDiagnostics {
    bool measurement_valid = false;
    bool initialized = false;
    bool reset = false;
    // True only when the filter deliberately reset itself from a measurement
    // that had just failed the translation gate after repeated rejections.
    // This keeps rejection statistics honest: the pose was used to recover.
    bool reinitialized_from_measurement = false;
    bool translation_accepted = false;
    bool rotation_accepted = false;
    double dt_s = 0.0;
    double translation_innovation_mm = 0.0;
    double translation_mahalanobis2 = 0.0;
    double rotation_innovation_deg = 0.0;
    int consecutive_translation_rejections = 0;
};

struct TemporalPoseFilterResult {
    Pose pose;
    TemporalPoseFilterDiagnostics diagnostics;
};

// A causal SE(3) output filter.  It takes a complete raw PnP measurement and
// returns a complete filtered Pose in the same convention: R_wc/tvec are
// world-to-camera, while the translation state itself is the camera centre in
// world C_w.  `update` never changes the PnP estimator's own state.
class TemporalPoseFilter {
public:
    explicit TemporalPoseFilter(TemporalPoseFilterConfig config = {})
        : config_(sanitizeConfig(config)) {}

    void reset() {
        initialized_ = false;
        last_ts_ = 0;
        x_.setZero();
        P_.setIdentity();
        R_wc_.setIdentity();
        angular_velocity_.setZero();
        consecutive_translation_rejections_ = 0;
    }

    bool initialized() const { return initialized_; }
    const TemporalPoseFilterConfig& config() const { return config_; }

    // A non-mutating constant-velocity prediction for a short missing-PnP
    // interval. It never becomes a PnP prior and returns nullopt for equal,
    // out-of-order, or overlong timestamp gaps.
    std::optional<Pose> predict(Timestamp ts) const {
        if (!initialized_) return std::nullopt;
        if (ts <= last_ts_) return std::nullopt;
        const long double gap_us = static_cast<long double>(ts) -
                                   static_cast<long double>(last_ts_);
        if (gap_us > static_cast<long double>(config_.reset_gap_us)) {
            return std::nullopt;
        }
        const double dt_s = static_cast<double>(gap_us * 1e-6L);

        Eigen::Matrix<double, 6, 1> x_pred = x_;
        x_pred.head<3>() += dt_s * x_.tail<3>();
        const Eigen::Matrix3d R_pred = R_wc_ * expSO3(angular_velocity_ * dt_s);
        return makePose(ts, x_pred.head<3>(), R_pred);
    }

    TemporalPoseFilterResult update(const Pose& measurement) {
        Eigen::Vector3d measured_position;
        Eigen::Matrix3d measured_R_wc;
        if (!poseToState(measurement, &measured_position, &measured_R_wc)) {
            TemporalPoseFilterResult result;
            result.pose = measurement;
            return result;
        }

        if (!initialized_) {
            return initialize(measurement.ts, measured_position, measured_R_wc, false);
        }

        const long double gap_us = static_cast<long double>(measurement.ts) -
                                   static_cast<long double>(last_ts_);
        const double dt_s = static_cast<double>(gap_us * 1e-6L);
        if (!(dt_s > 0.0) || gap_us > static_cast<long double>(config_.reset_gap_us)) {
            return initialize(measurement.ts, measured_position, measured_R_wc, true);
        }

        // Predict a constant-velocity translation state driven by white
        // acceleration noise, then apply a robust measurement update.
        const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
        H.block<3, 3>(0, 0) = I3;
        Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
        F.block<3, 3>(0, 3) = dt_s * I3;

        const double accel_variance =
            config_.process_accel_std_mm_s2 * config_.process_accel_std_mm_s2;
        Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
        Q.block<3, 3>(0, 0) = (dt_s * dt_s * dt_s * dt_s / 4.0) * I3;
        Q.block<3, 3>(0, 3) = (dt_s * dt_s * dt_s / 2.0) * I3;
        Q.block<3, 3>(3, 0) = (dt_s * dt_s * dt_s / 2.0) * I3;
        Q.block<3, 3>(3, 3) = (dt_s * dt_s) * I3;
        Q *= accel_variance;

        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q;
        P_ = 0.5 * (P_ + P_.transpose());
        last_ts_ = measurement.ts;

        TemporalPoseFilterDiagnostics diagnostics;
        diagnostics.measurement_valid = true;
        diagnostics.dt_s = dt_s;

        const Eigen::Vector3d translation_innovation = measured_position - H * x_;
        diagnostics.translation_innovation_mm = translation_innovation.norm();

        const double measurement_variance = config_.measurement_std_mm * config_.measurement_std_mm;
        const Eigen::Matrix3d R_measurement = measurement_variance * I3;
        const Eigen::Matrix3d S = H * P_ * H.transpose() + R_measurement;
        Eigen::LDLT<Eigen::Matrix3d> ldlt_S(S);
        bool solvable = ldlt_S.info() == Eigen::Success && ldlt_S.isPositive();
        if (solvable) {
            const Eigen::Vector3d solved = ldlt_S.solve(translation_innovation);
            diagnostics.translation_mahalanobis2 = translation_innovation.dot(solved);
            solvable = std::isfinite(diagnostics.translation_mahalanobis2);
        }
        if (!solvable) diagnostics.translation_mahalanobis2 = std::numeric_limits<double>::infinity();

        diagnostics.translation_accepted =
            diagnostics.translation_mahalanobis2 <= config_.hard_gate_chi2;
        if (diagnostics.translation_accepted) {
            // A plausible but large innovation remains usable, but has less
            // influence than a nominal one (Huber-like covariance inflation).
            const double inflation = std::max(
                1.0, diagnostics.translation_mahalanobis2 / config_.gate_chi2);
            const Eigen::Matrix3d S_effective =
                H * P_ * H.transpose() + inflation * R_measurement;
            Eigen::LDLT<Eigen::Matrix3d> ldlt_effective(S_effective);
            if (ldlt_effective.info() == Eigen::Success && ldlt_effective.isPositive()) {
                const Eigen::Matrix<double, 6, 3> PHt = P_ * H.transpose();
                const Eigen::Matrix<double, 3, 6> solved =
                    ldlt_effective.solve(PHt.transpose());
                const Eigen::Matrix<double, 6, 3> K = solved.transpose();
                x_ += K * translation_innovation;
                const Eigen::Matrix<double, 6, 6> I6 =
                    Eigen::Matrix<double, 6, 6>::Identity();
                const Eigen::Matrix<double, 6, 6> A = I6 - K * H;
                P_ = A * P_ * A.transpose() + K * (inflation * R_measurement) * K.transpose();
                P_ = 0.5 * (P_ + P_.transpose());
                consecutive_translation_rejections_ = 0;
            } else {
                diagnostics.translation_accepted = false;
                ++consecutive_translation_rejections_;
            }
        } else {
            ++consecutive_translation_rejections_;
        }

        // Predict and update orientation directly on SO(3).  Right-multiplied
        // relative rotations match log(R_pred^T R_meas), avoiding the branch
        // cut and geometric error of averaging three Rodrigues components.
        const Eigen::Matrix3d predicted_R_wc =
            R_wc_ * expSO3(angular_velocity_ * dt_s);
        const Eigen::Vector3d orientation_innovation =
            logSO3(predicted_R_wc.transpose() * measured_R_wc);
        diagnostics.rotation_innovation_deg =
            orientation_innovation.norm() * kDegreesPerRadian;
        diagnostics.rotation_accepted =
            diagnostics.rotation_innovation_deg <= config_.orientation_hard_gate_deg;
        if (diagnostics.rotation_accepted) {
            const double attenuation = std::min(
                1.0,
                config_.orientation_gate_deg /
                    std::max(diagnostics.rotation_innovation_deg, kTiny));
            const double alpha = config_.orientation_alpha * attenuation;
            const double beta = config_.orientation_beta * attenuation;
            R_wc_ = predicted_R_wc * expSO3(alpha * orientation_innovation);
            angular_velocity_ += (beta / std::max(dt_s, 1e-4)) * orientation_innovation;
        } else {
            R_wc_ = predicted_R_wc;
        }

        diagnostics.consecutive_translation_rejections = consecutive_translation_rejections_;
        if (consecutive_translation_rejections_ >=
            config_.max_consecutive_translation_rejections) {
            // Do not leave the causal predictor detached after a true, large
            // camera motion or a prolonged tracking interruption.
            TemporalPoseFilterResult reinitialized =
                initialize(measurement.ts, measured_position, measured_R_wc, true);
            reinitialized.diagnostics.translation_innovation_mm = diagnostics.translation_innovation_mm;
            reinitialized.diagnostics.translation_mahalanobis2 = diagnostics.translation_mahalanobis2;
            reinitialized.diagnostics.rotation_innovation_deg = diagnostics.rotation_innovation_deg;
            reinitialized.diagnostics.reinitialized_from_measurement = true;
            reinitialized.diagnostics.translation_accepted = false;
            reinitialized.diagnostics.rotation_accepted = diagnostics.rotation_accepted;
            return reinitialized;
        }

        TemporalPoseFilterResult result;
        result.pose = makePose(measurement.ts, x_.head<3>(), R_wc_);
        result.diagnostics = diagnostics;
        return result;
    }

private:
    static constexpr double kTiny = 1e-12;
    static constexpr double kDegreesPerRadian = 57.2957795130823208768;

    static TemporalPoseFilterConfig sanitizeConfig(TemporalPoseFilterConfig config) {
        const auto positive_or = [](double value, double fallback) {
            return std::isfinite(value) && value > 0.0 ? value : fallback;
        };
        config.process_accel_std_mm_s2 = positive_or(config.process_accel_std_mm_s2, 10000.0);
        config.measurement_std_mm = positive_or(config.measurement_std_mm, 15.0);
        config.gate_chi2 = positive_or(config.gate_chi2, 16.27);
        config.hard_gate_chi2 = std::max(
            config.gate_chi2, positive_or(config.hard_gate_chi2, 81.0));
        config.orientation_alpha = std::clamp(
            std::isfinite(config.orientation_alpha) ? config.orientation_alpha : 0.35, 0.0, 1.0);
        config.orientation_beta = std::max(
            0.0, std::isfinite(config.orientation_beta) ? config.orientation_beta : 0.025);
        config.orientation_gate_deg = positive_or(config.orientation_gate_deg, 180.0);
        config.orientation_hard_gate_deg = std::max(
            config.orientation_gate_deg,
            positive_or(config.orientation_hard_gate_deg, 360.0));
        config.reset_gap_us = std::max<Timestamp>(1, config.reset_gap_us);
        config.max_consecutive_translation_rejections =
            std::max(1, config.max_consecutive_translation_rejections);
        return config;
    }

    static bool cvVectorToEigen(const cv::Mat& source, Eigen::Vector3d* destination) {
        if (source.empty() || source.channels() != 1 || source.total() != 3) return false;
        cv::Mat flat;
        try {
            source.reshape(1, 1).convertTo(flat, CV_64F);
        } catch (const cv::Exception&) {
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            (*destination)[i] = flat.at<double>(0, i);
        }
        return destination->allFinite();
    }

    static bool poseToState(const Pose& pose, Eigen::Vector3d* position_world,
                            Eigen::Matrix3d* R_wc) {
        Eigen::Vector3d rvec;
        Eigen::Vector3d tvec;
        if (!cvVectorToEigen(pose.rvec, &rvec) || !cvVectorToEigen(pose.tvec, &tvec)) {
            return false;
        }

        cv::Mat cv_rvec(3, 1, CV_64F);
        for (int i = 0; i < 3; ++i) cv_rvec.at<double>(i) = rvec[i];
        cv::Mat cv_R;
        try {
            cv::Rodrigues(cv_rvec, cv_R);
        } catch (const cv::Exception&) {
            return false;
        }
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                (*R_wc)(row, col) = cv_R.at<double>(row, col);
            }
        }
        if (!R_wc->allFinite()) return false;
        *position_world = -R_wc->transpose() * tvec;
        return position_world->allFinite();
    }

    static Pose makePose(Timestamp ts, const Eigen::Vector3d& position_world,
                         const Eigen::Matrix3d& R_wc) {
        Pose pose;
        pose.ts = ts;
        cv::Mat cv_R(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                cv_R.at<double>(row, col) = R_wc(row, col);
            }
        }
        cv::Rodrigues(cv_R, pose.rvec);
        pose.tvec = cv::Mat(3, 1, CV_64F);
        const Eigen::Vector3d tvec = -R_wc * position_world;
        for (int i = 0; i < 3; ++i) pose.tvec.at<double>(i) = tvec[i];
        return pose;
    }

    static Eigen::Matrix3d expSO3(const Eigen::Vector3d& tangent) {
        const double angle = tangent.norm();
        if (angle < kTiny) {
            Eigen::Matrix3d skew;
            skew << 0.0, -tangent.z(), tangent.y(),
                    tangent.z(), 0.0, -tangent.x(),
                    -tangent.y(), tangent.x(), 0.0;
            return Eigen::Matrix3d::Identity() + skew;
        }
        return Eigen::AngleAxisd(angle, tangent / angle).toRotationMatrix();
    }

    static Eigen::Vector3d logSO3(const Eigen::Matrix3d& rotation) {
        Eigen::Quaterniond q(rotation);
        if (!q.coeffs().allFinite() || q.norm() < kTiny) return Eigen::Vector3d::Zero();
        q.normalize();
        // q and -q encode the same rotation.  Keep w non-negative so log
        // takes the shortest branch, including stable behavior near pi.
        if (q.w() < 0.0) q.coeffs() *= -1.0;
        const double sine_half_angle = q.vec().norm();
        if (sine_half_angle < kTiny) return 2.0 * q.vec();
        const double angle = 2.0 * std::atan2(sine_half_angle, q.w());
        return (angle / sine_half_angle) * q.vec();
    }

    TemporalPoseFilterResult initialize(Timestamp ts, const Eigen::Vector3d& position_world,
                                        const Eigen::Matrix3d& R_wc, bool reset_event) {
        initialized_ = true;
        last_ts_ = ts;
        x_.setZero();
        x_.head<3>() = position_world;
        P_.setZero();
        const double position_variance = config_.measurement_std_mm * config_.measurement_std_mm;
        const double velocity_std = std::max(10.0 * config_.measurement_std_mm, 100.0);
        P_.block<3, 3>(0, 0) = position_variance * Eigen::Matrix3d::Identity();
        P_.block<3, 3>(3, 3) = velocity_std * velocity_std * Eigen::Matrix3d::Identity();
        R_wc_ = R_wc;
        angular_velocity_.setZero();
        consecutive_translation_rejections_ = 0;

        TemporalPoseFilterResult result;
        result.pose = makePose(ts, position_world, R_wc);
        result.diagnostics.measurement_valid = true;
        result.diagnostics.initialized = true;
        result.diagnostics.reset = reset_event;
        result.diagnostics.translation_accepted = true;
        result.diagnostics.rotation_accepted = true;
        return result;
    }

    TemporalPoseFilterConfig config_;
    bool initialized_ = false;
    Timestamp last_ts_ = 0;
    Eigen::Matrix<double, 6, 1> x_ = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> P_ = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix3d R_wc_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d angular_velocity_ = Eigen::Vector3d::Zero();
    int consecutive_translation_rejections_ = 0;
};

}  // namespace emocap
