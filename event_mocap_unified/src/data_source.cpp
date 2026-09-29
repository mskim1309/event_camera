// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// EventFileSource / LiveCameraSource use Prophesee Metavision SDK.
// Reference: https://docs.prophesee.ai/

#include "event_mocap/data_source.hpp"

#include <thread>
#include <chrono>
#include <iostream>

namespace emocap {

// ============================================================
// SimulationSource (always available)
// ============================================================

SimulationSource::SimulationSource(const ActiveMarker& marker,
                                   const CameraIntrinsics& intrinsics,
                                   const SimConfig& config)
    : sim_(std::make_unique<MarkerSimulator>(marker, intrinsics, config)) {}

void SimulationSource::setGroundTruthCallback(GroundTruthCallback cb) { gt_cb_ = cb; }

void SimulationSource::start(DataObservationCallback cb) {
    running_ = true;
    sim_->run(cb, gt_cb_);
    running_ = false;
}

void SimulationSource::stop() { running_ = false; }
bool SimulationSource::is_running() const { return running_; }

}  // namespace emocap

// ============================================================
// EventFileSource / LiveCameraSource (requires MetavisionSDK)
// ============================================================
#ifdef HAS_METAVISION

#include "event_mocap/active_marker_tracker.hpp"
#include <metavision/sdk/driver/camera.h>

namespace emocap {

// --- EventFileSource ---

struct EventFileSource::Impl {
    Metavision::Camera camera;
    std::unique_ptr<ActiveMarkerTracker> tracker;
    std::atomic<bool> running{false};
};

EventFileSource::EventFileSource(const Params& params)
    : impl_(std::make_unique<Impl>()), params_(params) {}
EventFileSource::~EventFileSource() = default;

void EventFileSource::start(DataObservationCallback cb) {
    auto cam_config = Metavision::FileConfigHints().real_time_playback(params_.realtime_playback);
    impl_->camera = Metavision::Camera::from_file(params_.event_file_path, cam_config);

    impl_->tracker = std::make_unique<ActiveMarkerTracker>(params_.marker, params_.tracker_params);
    impl_->tracker->setCallback(cb);
    impl_->tracker->attach(impl_->camera);

    impl_->running = true;
    impl_->camera.start();
    while (impl_->camera.is_running())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    impl_->camera.stop();
    impl_->running = false;
}

void EventFileSource::stop() {
    if (impl_->running) impl_->camera.stop();
}

bool EventFileSource::is_running() const { return impl_->running; }

// --- LiveCameraSource ---

struct LiveCameraSource::Impl {
    Metavision::Camera camera;
    std::unique_ptr<ActiveMarkerTracker> tracker;
    std::atomic<bool> running{false};
};

LiveCameraSource::LiveCameraSource(const Params& params)
    : impl_(std::make_unique<Impl>()), params_(params) {}
LiveCameraSource::~LiveCameraSource() = default;

void LiveCameraSource::start(DataObservationCallback cb) {
    Metavision::DeviceConfig device_config;
    device_config.enable_biases_range_check_bypass(true);
    impl_->camera = Metavision::Camera::from_first_available(device_config);
    if (!params_.camera_config_path.empty())
        impl_->camera.load(params_.camera_config_path);

    impl_->tracker = std::make_unique<ActiveMarkerTracker>(params_.marker, params_.tracker_params);
    impl_->tracker->setCallback(cb);
    impl_->tracker->attach(impl_->camera);

    impl_->running = true;
    impl_->camera.start();
    while (impl_->camera.is_running())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    impl_->camera.stop();
    impl_->running = false;
}

void LiveCameraSource::stop() {
    if (impl_->running) impl_->camera.stop();
}

bool LiveCameraSource::is_running() const { return impl_->running; }

}  // namespace emocap

#endif  // HAS_METAVISION
