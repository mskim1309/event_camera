// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#include "event_mocap/config_loader.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <stdexcept>
#include <iostream>

namespace emocap {

ActiveMarker loadActiveMarker(const std::string& json_path) {
    namespace pt = boost::property_tree;
    ActiveMarker marker;
    pt::ptree root;
    pt::read_json(json_path, root);

    for (const auto& led_node : root.get_child("active marker")) {
        auto id = led_node.second.get<std::uint32_t>("led.id");
        std::vector<float> xyz;
        for (const auto& v : led_node.second.get_child("led.xyz"))
            xyz.push_back(v.second.get_value<float>());
        marker[id] = cv::Point3f{xyz[0], xyz[1], xyz[2]};
    }
    return marker;
}

CameraIntrinsics loadCalibration(const std::string& json_path) {
    namespace pt = boost::property_tree;
    pt::ptree root;
    pt::read_json(json_path, root);

    // Support multiple JSON layouts
    pt::ptree intrinsics;
    if (auto node = root.get_child_optional("intrinsics"))
        intrinsics = *node;
    else if (auto node = root.get_child_optional("proj_master"))
        intrinsics = *node;
    else
        intrinsics = root;

    CameraIntrinsics cam;
    cam.width  = intrinsics.get<int>("width", 1280);
    cam.height = intrinsics.get<int>("height", 720);

    std::vector<double> K;
    if (auto knode = intrinsics.get_child_optional("K")) {
        for (auto& v : *knode) K.push_back(v.second.get_value<double>());
    }
    if (K.size() == 9) {
        cam.fx = K[0]; cam.cx = K[2];
        cam.fy = K[4]; cam.cy = K[5];
    }

    if (auto dnode = intrinsics.get_child_optional("D")) {
        for (auto& v : *dnode)
            cam.dist_coeffs.push_back(v.second.get_value<double>());
    }

    return cam;
}

}  // namespace emocap
