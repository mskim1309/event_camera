// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.

#ifndef RERUN_COLLECTION_ADAPTERS_H
#define RERUN_COLLECTION_ADAPTERS_H

#ifdef RERUN_FOUND

#include <rerun.hpp>
#include <Eigen/Core>
#include <vector>
#include <cstring>

// Adapter for logging std::vector<Eigen::Vector3f> as rerun positions.
template <>
struct rerun::CollectionAdapter<rerun::Position3D, std::vector<Eigen::Vector3f>> {
  Collection<rerun::Position3D> operator()(const std::vector<Eigen::Vector3f>& container){
    return Collection<rerun::Position3D>::borrow(container.data(), container.size());
  }

  Collection<rerun::Position3D> operator()(std::vector<Eigen::Vector3f>&& container){
    std::vector<rerun::Position3D> positions(container.size());
    std::memcpy(positions.data(), container.data(), container.size() * sizeof(Eigen::Vector3f));
    return Collection<rerun::Position3D>::take_ownership(std::move(positions));
  }
};

// Adapter for logging an Eigen::Matrix3Xf as rerun positions.
template <>
struct rerun::CollectionAdapter<rerun::Position3D, Eigen::Matrix3Xf>{
  static_assert(
    sizeof(rerun::Position3D) == sizeof(Eigen::Matrix3Xf::Scalar) * Eigen::Matrix3Xf::RowsAtCompileTime,
    "Incompatible types"
  );

  Collection<rerun::Position3D> operator()(const Eigen::Matrix3Xf& matrix){
    static_assert(alignof(rerun::Position3D) <= alignof(Eigen::Matrix3Xf::Scalar));
    return Collection<rerun::Position3D>::borrow(
        reinterpret_cast<const void*>(matrix.data()),
        matrix.cols());
  }

  Collection<rerun::Position3D> operator()(Eigen::Matrix3Xf&& matrix){
    std::vector<rerun::Position3D> positions(matrix.cols());
    std::memcpy(positions.data(), matrix.data(), matrix.size() * sizeof(rerun::Position3D));
    return Collection<rerun::Position3D>::take_ownership(std::move(positions));
  }
};

#endif  // RERUN_FOUND

#endif  // RERUN_COLLECTION_ADAPTERS_H