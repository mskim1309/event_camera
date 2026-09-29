// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#include "event_mocap/visualizer.hpp"

#include <iostream>
#include <thread>
#include <chrono>

#ifdef RERUN_FOUND

#include <rerun.hpp>
#include "util/viewer/rerun_viewer_util.h"

namespace emocap {

struct Visualizer::Impl {
    rerun::RecordingStream rec;
    CameraIntrinsics intrinsics;
    ActiveMarker marker;

    // Accumulated trajectory positions for line strip rendering
    std::vector<rvl::Vec3> est_positions;
    std::vector<rvl::Vec3> gt_positions;

    Impl(const std::string& name, const CameraIntrinsics& intr, const ActiveMarker& m)
        : rec(name), intrinsics(intr), marker(m)
    {
        rec.spawn().exit_on_failure();
        rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);

        // World origin axes at identity
        rvl::Mat34 origin = rvl::IdentityMat34();
        rvl::rrLogAxes(rec, "world/origin", origin, 50.0f);
    }
};

Visualizer::Visualizer(const std::string& app_name,
                       const CameraIntrinsics& intrinsics,
                       const ActiveMarker& marker)
    : impl_(std::make_unique<Impl>(app_name, intrinsics, marker)) {}

Visualizer::~Visualizer() = default;

bool Visualizer::enabled() const { return true; }

void Visualizer::logMarkers() {
    std::vector<rerun::Position3D> pts;
    std::vector<std::string> labels;
    for (const auto& [id, p] : impl_->marker) {
        pts.push_back(rerun::Position3D(p.x, p.y, p.z));
        labels.push_back("LED " + std::to_string(id));
    }
    impl_->rec.log_static("world/markers",
        rerun::Points3D(pts)
            .with_radii(rerun::Radius::ui_points(8.0f))
            .with_colors(rvl::rr_purple)
            .with_labels(labels));
}

void Visualizer::logEstimatedPose(const Pose& pose) {
    double t_sec = pose.ts / 1e6;
    impl_->rec.set_time_seconds("time", t_sec);

    rvl::Vec3 pos = pose.position_world();
    rvl::Mat34 T_cw = pose.T_cw();

    // Camera coordinate axes
    rvl::rrLogAxes(impl_->rec, "world/estimated/axes", T_cw, 30.0f);

    // Camera frustum
    rvl::IntrinsicPrm intr(impl_->intrinsics.fx, impl_->intrinsics.fy,
                            impl_->intrinsics.cx, impl_->intrinsics.cy,
                            impl_->intrinsics.width, impl_->intrinsics.height);
    rvl::Mat34 identity = rvl::IdentityMat34();
    rvl::rrLogCamera(impl_->rec, "world/estimated/camera", T_cw, identity, intr);

    // Accumulated trajectory
    impl_->est_positions.push_back(pos);
    if (impl_->est_positions.size() >= 2) {
        rvl::rrLogTrajectory(impl_->rec, "world/estimated/path",
                             rvl::rr_green, 2.0, impl_->est_positions);
    }
}

void Visualizer::logGroundTruth(const Pose& pose) {
    double t_sec = pose.ts / 1e6;
    impl_->rec.set_time_seconds("time", t_sec);

    rvl::Vec3 pos = pose.position_world();
    rvl::Mat34 T_cw = pose.T_cw();

    // GT axes (smaller)
    rvl::rrLogAxes(impl_->rec, "world/ground_truth/axes", T_cw, 20.0f);

    // GT trajectory
    impl_->gt_positions.push_back(pos);
    if (impl_->gt_positions.size() >= 2) {
        rvl::rrLogTrajectory(impl_->rec, "world/ground_truth/path",
                             rvl::rr_orange, 2.0, impl_->gt_positions);
    }
}

void Visualizer::logError(Timestamp ts, double pos_err_mm, double rot_err_deg) {
    impl_->rec.set_time_seconds("time", ts / 1e6);
    impl_->rec.log("error/position_mm", rerun::Scalar(pos_err_mm));
    impl_->rec.log("error/rotation_deg", rerun::Scalar(rot_err_deg));
}

void Visualizer::logObservations(Timestamp ts, const std::vector<MarkerObservation>& obs) {
    impl_->rec.set_time_seconds("time", ts / 1e6);

    int w = impl_->intrinsics.width;
    int h = impl_->intrinsics.height;

    // Background frame (dark gray, camera resolution)
    std::vector<uint8_t> bg(w * h * 3, 30);
    impl_->rec.log("tracking",
        rerun::Image::from_rgb24(bg, {static_cast<uint32_t>(w), static_cast<uint32_t>(h)}));

    // 2D marker observations overlaid on image
    std::vector<rerun::Position2D> pts;
    std::vector<std::string> labels;
    for (const auto& o : obs) {
        pts.push_back(rerun::Position2D(o.pt2d.x, o.pt2d.y));
        labels.push_back("LED " + std::to_string(o.id));
    }
    impl_->rec.log("tracking",
        rerun::Points2D(pts)
            .with_radii(rerun::Radius::ui_points(6.0f))
            .with_colors(rvl::rr_yellow)
            .with_labels(labels));
}

void Visualizer::block(double seconds) {
    std::cout << "[Vis] Rerun viewer open. Blocking for " << seconds << "s..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(seconds * 1000)));
}

}  // namespace emocap

#else  // !RERUN_FOUND — no-op stubs

namespace emocap {

struct Visualizer::Impl {};

Visualizer::Visualizer(const std::string&, const CameraIntrinsics&, const ActiveMarker&)
    : impl_(std::make_unique<Impl>()) {}
Visualizer::~Visualizer() = default;
bool Visualizer::enabled() const { return false; }
void Visualizer::logMarkers() {}
void Visualizer::logEstimatedPose(const Pose&) {}
void Visualizer::logGroundTruth(const Pose&) {}
void Visualizer::logError(Timestamp, double, double) {}
void Visualizer::logObservations(Timestamp, const std::vector<MarkerObservation>&) {}
void Visualizer::block(double) {
    std::cout << "[Vis] Rerun SDK not found — visualization disabled." << std::endl;
}

}  // namespace emocap

#endif  // RERUN_FOUND
