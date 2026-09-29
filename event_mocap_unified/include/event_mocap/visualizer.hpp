// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "event_mocap/types.hpp"

namespace emocap {

class Visualizer {
public:
    Visualizer(const std::string& app_name,
               const CameraIntrinsics& intrinsics,
               const ActiveMarker& marker);
    ~Visualizer();

    // Log static marker 3D positions (call once)
    void logMarkers();

    // Log estimated camera pose at time t
    void logEstimatedPose(const Pose& pose);

    // Log ground truth pose (simulation mode)
    void logGroundTruth(const Pose& pose);

    // Log scalar error (simulation mode)
    void logError(Timestamp ts, double pos_err_mm, double rot_err_deg);

    // Log 2D observations on image plane
    void logObservations(Timestamp ts, const std::vector<MarkerObservation>& obs);

    // Block to keep viewer open
    void block(double seconds = 300.0);

    bool enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace emocap
