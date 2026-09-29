// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#ifndef RERUN_VIEWER_UTIL_H
#define RERUN_VIEWER_UTIL_H

#include <string>
#include <map>
#include <vector>

#ifdef RERUN_FOUND

#include <rerun.hpp>
#include "util/viewer/rerun_collection_adapters.h"
#include "util/geometry.h"
#include "util/rvl_common_types.h"

namespace rvl {

struct IntrinsicPrm{
	IntrinsicPrm(){}
	IntrinsicPrm(
			double f_, double cx_, double cy_, 
			int w_, int h_, std::vector<double> dc = {}){
		fx = f_; fy = f_; cx = cx_; cy = cy_; 
		w = w_; h = h_; ks = dc;
	}
	IntrinsicPrm(
			double fx_, double fy_, double cx_, double cy_, 
			int w_, int h_, std::vector<double> dc = {}){
		fx = fx_; fy = fy_; cx = cx_; cy = cy_; 
		w = w_; h = h_; ks = dc;
	}
	int w; int h; 
	double fx; double fy; double cx; double cy;
	std::vector<double> ks = {};
};

// Color constants
const rerun::Color rr_red = rerun::Color(255, 0, 0);
const rerun::Color rr_blue = rerun::Color(0, 0, 255);
const rerun::Color rr_green = rerun::Color(0, 255, 0);
const rerun::Color rr_darkgreen = rerun::Color(0, 170, 0);
const rerun::Color rr_purple = rerun::Color(255, 0, 255);
const rerun::Color rr_yellow = rerun::Color(255, 255, 0);
const rerun::Color rr_orange = rerun::Color(255, 165, 0);
const rerun::Color rr_gray = rerun::Color(100, 100, 100);
const rerun::Color rr_white = rerun::Color(255, 255, 255);

//-----------------------------------------------------------------------------
// Log coordinate axes at a pose
inline void rrLogAxes(
    rerun::RecordingStream &rec, const std::string &entity,
    const Mat34 &pose, float axis_length = 0.15f) {
  Eigen::Vector3f t_f = GetTr(pose).cast<float>();
  Eigen::Matrix3f R_f = GetRot(pose).cast<float>() * axis_length;

  std::vector<rerun::Position3D> origins;
  std::vector<rerun::Vector3D> vectors;
  for (int i = 0; i < 3; i++) {
    origins.push_back(rerun::Position3D(rerun::Vec3D(t_f.data())));
    vectors.push_back(rerun::Vector3D(rerun::Vec3D(R_f.col(i).data())));
  }
  std::vector<rerun::Color> colors = {rr_red, rr_green, rr_blue};
  rec.log(entity,
          rerun::Arrows3D::from_vectors(vectors)
              .with_origins(origins)
              .with_colors(colors));
}
//-----------------------------------------------------------------------------
// Log a camera transform and intrinsics.
inline void rrLogCamera(
    rerun::RecordingStream &rec, const std::string &entity,
        const Mat34 &curpose, const Mat34 &cam2rig, const IntrinsicPrm &intr){
  Mat34 cam2world = MergedTransform(curpose, cam2rig);
  Eigen::Vector3f t_f = GetTr(cam2world).cast<float>();
  Eigen::Matrix3f R_f = GetRot(cam2world).cast<float>();
  rec.log(entity,
          rerun::Transform3D(rerun::Vec3D(t_f.data()), rerun::Mat3x3(R_f.data()), false));
  rec.log(entity,
          rerun::Pinhole::from_focal_length_and_resolution(
              { static_cast<float>(intr.fx), static_cast<float>(intr.fy) },
              { static_cast<float>(intr.w), static_cast<float>(intr.h) }
          ).with_image_plane_distance(0.3f));
}


// Log a camera transform and intrinsics and IMG.
// inline void rrLogCameraWithImg(
//     rerun::RecordingStream &rec, const std::string &entity, 
//     const cv::Mat img, // suppose that img is 3 channel byte img
// 		const Mat34 &curpose, const Mat34 &cam2rig, const IntrinsicPrm &intr){
//   Mat34 cam2world = MergedTransform(curpose, cam2rig);
//   Eigen::Vector3f t_f = GetTr(cam2world).cast<float>();
//   Eigen::Matrix3f R_f = GetRot(cam2world).cast<float>();
//   rec.log(entity,
//           rerun::Transform3D(rerun::Vec3D(t_f.data()), rerun::Mat3x3(R_f.data()), false));
//   rec.log(entity,
//           rerun::Pinhole::from_focal_length_and_resolution(
//               { static_cast<float>(intr.fx), static_cast<float>(intr.fy) },
//               { static_cast<float>(intr.w), static_cast<float>(intr.h) }
//           ));
//   rec.log(entity, rerun::Image::from_rgb24(rerun::Collection<uint8_t>(img), width_height(img)));
// }
//-----------------------------------------------------------------------------
// Log trajectory from map<int, Mat34>
inline void rrLogTrajectory(
    rerun::RecordingStream &rec, const std::string &entity,
    const rerun::Color color, const double radii,
    const std::map<int, Mat34> &poses) {
  if (poses.empty()) return;
  std::vector<rerun::datatypes::Vec3D> path;
  for (const auto &entry : poses) {
    Eigen::Vector3f pos_f = GetTr(entry.second).cast<float>();
    path.push_back(rerun::datatypes::Vec3D(pos_f.data()));
  }
  rec.log(entity, rerun::LineStrips3D(rerun::LineStrip3D(path))
                      .with_radii(rerun::Radius::ui_points(radii))
                      .with_colors(color));
}

//-----------------------------------------------------------------------------
// Log trajectory from vector of Vec3
inline void rrLogTrajectory(
    rerun::RecordingStream &rec, const std::string &entity,
    const rerun::Color color, const double radii,
    const std::vector<Vec3> &positions) {
  if (positions.empty()) return;
  std::vector<rerun::datatypes::Vec3D> path;
  for (const auto &pos : positions) {
    Eigen::Vector3f pos_f = pos.cast<float>();
    path.push_back(rerun::datatypes::Vec3D(pos_f.data()));
  }
  rec.log(entity, rerun::LineStrips3D(rerun::LineStrip3D(path))
                      .with_radii(rerun::Radius::ui_points(radii))
                      .with_colors(color));
}

//-----------------------------------------------------------------------------
// Log trajectory with poses (axes at intervals)
inline void rrLogTrajectoryWithPoses(
    rerun::RecordingStream &rec, const std::string &entity,
    const rerun::Color color, const double radii,
    const std::vector<Mat34> &poses, int pose_skip = 10,
    float axis_length = 0.15f) {
  if (poses.empty()) return;

  // Log trajectory line
  std::vector<rerun::datatypes::Vec3D> path;
  for (const auto &pose : poses) {
    Eigen::Vector3f pos_f = GetTr(pose).cast<float>();
    path.push_back(rerun::datatypes::Vec3D(pos_f.data()));
  }
  rec.log(entity + "/path", rerun::LineStrips3D(rerun::LineStrip3D(path))
                                .with_radii(rerun::Radius::ui_points(radii))
                                .with_colors(color));

  // Log pose axes at intervals
  std::vector<rerun::Position3D> origins;
  std::vector<rerun::Vector3D> vectors;
  std::vector<rerun::Color> colors;

  for (size_t i = 0; i < poses.size(); i += pose_skip) {
    Eigen::Vector3f t_f = GetTr(poses[i]).cast<float>();
    Eigen::Matrix3f R_f = GetRot(poses[i]).cast<float>() * axis_length;

    // X axis (red)
    origins.push_back(rerun::Position3D(t_f.x(), t_f.y(), t_f.z()));
    vectors.push_back(rerun::Vector3D(R_f(0, 0), R_f(1, 0), R_f(2, 0)));
    colors.push_back(rr_red);

    // Y axis (green)
    origins.push_back(rerun::Position3D(t_f.x(), t_f.y(), t_f.z()));
    vectors.push_back(rerun::Vector3D(R_f(0, 1), R_f(1, 1), R_f(2, 1)));
    colors.push_back(rr_green);

    // Z axis (blue)
    origins.push_back(rerun::Position3D(t_f.x(), t_f.y(), t_f.z()));
    vectors.push_back(rerun::Vector3D(R_f(0, 2), R_f(1, 2), R_f(2, 2)));
    colors.push_back(rr_blue);
  }

  rec.log(entity + "/axes",
          rerun::Arrows3D::from_vectors(vectors)
              .with_origins(origins)
              .with_colors(colors));
}

//-----------------------------------------------------------------------------
// Log points
inline void rrLogPoints(
    rerun::RecordingStream &rec, const std::string &entity,
    const rerun::Color color, const double radii,
    const std::vector<Vec3> &points) {
  if (points.empty()) return;
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(points.size());
  for (const auto &p : points) {
    pts.push_back(p.cast<float>());
  }
  rec.log(entity, rerun::Points3D(pts)
                      .with_radii(rerun::Radius::ui_points(radii))
                      .with_colors(color));
}

//-----------------------------------------------------------------------------
// Log text message
inline void rrLogText(
    rerun::RecordingStream &rec, const std::string &entity,
    const std::string &msg) {
  rec.log(entity, rerun::TextLog(msg));
}

}  // namespace rvl

#endif  // RERUN_FOUND

#endif  // RERUN_VIEWER_UTIL_H