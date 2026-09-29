// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// EventFileSource / LiveCameraSource use Prophesee Metavision SDK.
// Reference: https://docs.prophesee.ai/

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <atomic>

#include "event_mocap/types.hpp"
#include "event_mocap/marker_simulator.hpp"

#ifdef HAS_METAVISION
#include "event_mocap/active_marker_tracker.hpp"
#endif

namespace emocap {

// Callback: receives observations per frame
using DataObservationCallback = std::function<void(
    Timestamp ts, const std::vector<MarkerObservation>& obs)>;

// --- Abstract DataSource ---
class DataSource {
public:
    virtual ~DataSource() = default;
    virtual void start(DataObservationCallback cb) = 0;
    virtual void stop() = 0;
    virtual bool is_running() const = 0;
};

// --- Simulation (always available) ---
class SimulationSource : public DataSource {
public:
    SimulationSource(const ActiveMarker& marker,
                     const CameraIntrinsics& intrinsics,
                     const SimConfig& config);

    void start(DataObservationCallback cb) override;
    void stop() override;
    bool is_running() const override;

    void setGroundTruthCallback(GroundTruthCallback cb);

private:
    std::unique_ptr<MarkerSimulator> sim_;
    GroundTruthCallback gt_cb_;
    std::atomic<bool> running_{false};
};

// --- Event file / Live camera (requires MetavisionSDK) ---
#ifdef HAS_METAVISION

class EventFileSource : public DataSource {
public:
    struct Params {
        std::string event_file_path;
        std::string camera_config_path;
        ActiveMarker marker;
        bool realtime_playback = false;
        TrackerParams tracker_params;
    };

    explicit EventFileSource(const Params& params);
    ~EventFileSource();
    void start(DataObservationCallback cb) override;
    void stop() override;
    bool is_running() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Params params_;
};

class LiveCameraSource : public DataSource {
public:
    using Params = EventFileSource::Params;

    explicit LiveCameraSource(const Params& params);
    ~LiveCameraSource();
    void start(DataObservationCallback cb) override;
    void stop() override;
    bool is_running() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Params params_;
};

#endif  // HAS_METAVISION

}  // namespace emocap
