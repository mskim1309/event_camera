// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "util/rvl_common_types.h"
#include "util/geometry.h"

namespace emocap {

// Import rvl types — Vec3, Mat3, Mat34, Quat, etc. available directly
using namespace rvl;

// Microsecond timestamp (matches Metavision::timestamp = long long)
using Timestamp = std::int64_t;

// 3D marker map: LED ID -> 3D position (mm)
using ActiveMarker = std::unordered_map<std::uint32_t, cv::Point3f>;

// Single 2D observation of a marker
struct MarkerObservation {
    std::uint32_t id;
    cv::Point2f pt2d;
};

// Camera intrinsics + distortion
struct CameraIntrinsics {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;
    std::vector<double> dist_coeffs;  // k1, k2, p1, p2, k3

    cv::Mat K() const {
        cv::Mat m = cv::Mat::eye(3, 3, CV_64F);
        m.at<double>(0, 0) = fx;
        m.at<double>(1, 1) = fy;
        m.at<double>(0, 2) = cx;
        m.at<double>(1, 2) = cy;
        return m;
    }

    cv::Mat D() const {
        if (dist_coeffs.empty()) return {};
        cv::Mat m(1, static_cast<int>(dist_coeffs.size()), CV_64F);
        for (int i = 0; i < static_cast<int>(dist_coeffs.size()); ++i)
            m.at<double>(0, i) = dist_coeffs[i];
        return m;
    }

    rvl::Mat3 K_eigen() const {
        rvl::Mat3 m;
        m << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return m;
    }
};

// 6-DOF pose result
struct Pose {
    Timestamp ts = 0;
    cv::Mat rvec;  // angle-axis (world->camera)
    cv::Mat tvec;  // translation (world->camera)

    // Camera center in world frame: C_w = -R_wc * tvec
    rvl::Vec3 position_world() const {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Mat C = -R.t() * tvec;
        return rvl::Vec3(C.at<double>(0), C.at<double>(1), C.at<double>(2));
    }

    // Rotation matrix: world->camera
    rvl::Mat3 R_wc() const {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        rvl::Mat3 m;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                m(r, c) = R.at<double>(r, c);
        return m;
    }

    // R camera->world
    rvl::Mat3 R_cw() const { return R_wc().transpose(); }

    // As Mat34 (camera-to-world transform)
    rvl::Mat34 T_cw() const {
        return rvl::ToMat34(R_cw(), position_world());
    }

    // Rotation as rvl angle-axis vector
    rvl::Vec3 rvec_eigen() const {
        return rvl::Vec3(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    }

    rvl::Vec3 tvec_eigen() const {
        return rvl::Vec3(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    }
};

// Frame observation for BA window
struct FrameObservation {
    Timestamp ts;
    std::vector<std::uint32_t> ids;
    std::vector<cv::Point2f> pts2d;
    cv::Mat rvec_seed;  // geometry-aware candidate seed, world -> camera
    cv::Mat tvec_seed;
};

// Execution mode
enum class Mode { SIM, FILE, LIVE };

}  // namespace emocap
