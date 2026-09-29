// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#include "event_mocap/marker_simulator.hpp"

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <iostream>

namespace emocap {

MarkerSimulator::MarkerSimulator(const ActiveMarker& marker,
                                 const CameraIntrinsics& intrinsics,
                                 const SimConfig& config)
    : marker_(marker), intrinsics_(intrinsics), config_(config) {}

Pose MarkerSimulator::cameraPoseAt(double t) const {
    double cx_w = 0, cy_w = 0, cz_w = 0;  // camera center in world

    switch (config_.traj_type) {
    case TrajectoryType::CIRCULAR: {
        double omega = 2.0 * M_PI * config_.circle_freq;
        cx_w = config_.circle_radius * std::cos(omega * t);
        cy_w = config_.circle_radius * std::sin(omega * t);
        cz_w = config_.circle_height;
        break;
    }
    case TrajectoryType::LINEAR: {
        cx_w = config_.linear_speed * t - config_.linear_speed * config_.duration_s / 2.0;
        cy_w = 0;
        cz_w = config_.linear_z;
        break;
    }
    case TrajectoryType::SINUSOIDAL: {
        cx_w = config_.sin_amp_x * std::sin(2.0 * M_PI * config_.sin_freq_x * t);
        cy_w = config_.sin_amp_y * std::sin(2.0 * M_PI * config_.sin_freq_y * t);
        cz_w = config_.sin_base_z + config_.sin_amp_z * std::sin(2.0 * M_PI * config_.sin_freq_z * t);
        break;
    }
    }

    // Camera looks toward the marker origin (0,0,0) from position (cx,cy,cz)
    // Camera convention: z-axis points forward (toward target), y-axis down
    Eigen::Vector3d cam_pos(cx_w, cy_w, cz_w);
    Eigen::Vector3d target(0, 0, 0);
    Eigen::Vector3d forward = (target - cam_pos).normalized();
    Eigen::Vector3d up_hint(0, 0, -1);  // "up" in world ~ -z (markers on z=0 plane)

    // If forward is nearly parallel to up_hint, pick different hint
    if (std::abs(forward.dot(up_hint)) > 0.99)
        up_hint = Eigen::Vector3d(0, 1, 0);

    Eigen::Vector3d right = forward.cross(up_hint).normalized();
    Eigen::Vector3d down = forward.cross(right).normalized();

    // R_wc: columns = [right, down, forward] (camera axes in world)
    Eigen::Matrix3d R_cw;  // camera-to-world
    R_cw.col(0) = right;
    R_cw.col(1) = down;
    R_cw.col(2) = forward;

    // R_wc = R_cw^T (world-to-camera rotation)
    Eigen::Matrix3d R_wc = R_cw.transpose();

    // tvec = -R_wc * cam_pos
    Eigen::Vector3d tvec = -R_wc * cam_pos;

    // Convert to angle-axis
    cv::Mat R_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R_cv.at<double>(r, c) = R_wc(r, c);

    Pose pose;
    pose.ts = static_cast<Timestamp>(t * 1e6);  // seconds → microseconds
    cv::Rodrigues(R_cv, pose.rvec);
    pose.tvec = cv::Mat(3, 1, CV_64F);
    pose.tvec.at<double>(0) = tvec.x();
    pose.tvec.at<double>(1) = tvec.y();
    pose.tvec.at<double>(2) = tvec.z();

    return pose;
}

std::vector<MarkerObservation> MarkerSimulator::projectMarkers(const Pose& pose) const {
    std::vector<MarkerObservation> obs;
    std::normal_distribution<double> noise(0.0, config_.pixel_noise);

    cv::Mat K = intrinsics_.K();
    cv::Mat D = intrinsics_.D();

    for (const auto& [id, pt3d] : marker_) {
        // Project: p_cam = R * p_world + t
        std::vector<cv::Point3f> obj_pts = {pt3d};
        std::vector<cv::Point2f> img_pts;
        cv::projectPoints(obj_pts, pose.rvec, pose.tvec, K, D, img_pts);

        double u = img_pts[0].x + noise(rng_);
        double v = img_pts[0].y + noise(rng_);

        // Check if within image bounds
        if (u >= 0 && u < intrinsics_.width && v >= 0 && v < intrinsics_.height) {
            obs.push_back({id, cv::Point2f(static_cast<float>(u), static_cast<float>(v))});
        }
    }
    return obs;
}

void MarkerSimulator::run(ObservationCallback obs_cb, GroundTruthCallback gt_cb) {
    double dt = 1.0 / config_.rate_hz;
    int n_steps = static_cast<int>(config_.duration_s * config_.rate_hz);

    std::cout << "[Sim] Running " << n_steps << " steps, dt=" << dt * 1e3
              << "ms, trajectory=" << static_cast<int>(config_.traj_type) << std::endl;

    for (int i = 0; i < n_steps; ++i) {
        double t = i * dt;
        Pose gt = cameraPoseAt(t);
        auto observations = projectMarkers(gt);

        if (gt_cb) gt_cb(gt.ts, gt);
        if (obs_cb && !observations.empty()) obs_cb(gt.ts, observations);
    }

    std::cout << "[Sim] Done." << std::endl;
}

}  // namespace emocap
