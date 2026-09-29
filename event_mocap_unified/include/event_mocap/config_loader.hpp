// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#pragma once

#include <string>
#include "event_mocap/types.hpp"

namespace emocap {

ActiveMarker loadActiveMarker(const std::string& json_path);
CameraIntrinsics loadCalibration(const std::string& json_path);

}  // namespace emocap
