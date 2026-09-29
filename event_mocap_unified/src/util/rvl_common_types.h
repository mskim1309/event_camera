// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim   
//   

#ifndef _RVL_COMMON_TYPES_H_
#define _RVL_COMMON_TYPES_H_

#include <cstdint>
#include <optional>

#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace rvl {

#define EPS_DOUBLE __DBL_EPSILON__
#define EPS_FLOAT __FLT_EPSILON__

typedef unsigned char byte;
typedef Eigen::Quaterniond Quat;
typedef Eigen::VectorXd VecX;
typedef Eigen::MatrixXd MatXX;

typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector4d Vec4;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 9, 1> Vec9;

typedef Eigen::Matrix<double, 2, 2> Mat2;
typedef Eigen::Matrix<double, 3, 3> Mat3;
typedef Eigen::Matrix<double, 4, 4> Mat4;
typedef Eigen::Matrix<double, 6, 6> Mat6;
typedef Eigen::Matrix<double, 7, 7> Mat7;
typedef Eigen::Matrix<double, 2, 4> Mat24;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> Mat1X;
typedef Eigen::Matrix<double, 2, Eigen::Dynamic> Mat2X;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Mat3X;
typedef Eigen::Matrix<double, 4, Eigen::Dynamic> Mat4X;
typedef Eigen::Matrix<double, 6, Eigen::Dynamic> Mat6X;
typedef Eigen::Matrix<double, 7, Eigen::Dynamic> Mat7X;

typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> Mask;

typedef Eigen::VectorXf VecXf;
typedef Eigen::MatrixXf MatXXf;
typedef Eigen::Matrix<float, 3, 3> Mat3f;
typedef Eigen::Matrix<float, 2, 4> Mat24f;
typedef Eigen::Matrix<float, 3, 4> Mat34f;
typedef Eigen::Matrix<float, 4, 4> Mat4f;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Mati;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> Matu8;
typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector4f Vec4f;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::Matrix<int64_t, 3, 1> Vec3l;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> Mat1Xf;
typedef Eigen::Matrix<float, 2, Eigen::Dynamic> Mat2Xf;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Mat3Xf;
typedef Eigen::Matrix<float, 4, Eigen::Dynamic> Mat4Xf;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic> Mat1Xi;
typedef Eigen::Matrix<int, 2, Eigen::Dynamic> Mat2Xi;
typedef Eigen::Matrix<int, 3, Eigen::Dynamic> Mat3Xi;
typedef Eigen::Matrix<int, 4, Eigen::Dynamic> Mat4Xi;
typedef Eigen::Matrix<uint8_t, 3, 1> Vec3u8;
typedef Eigen::Matrix<uint8_t, 4, 1> Vec4u8;
typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> Mat1Xu8;
typedef Eigen::Matrix<uint8_t, 2, Eigen::Dynamic> Mat2Xu8;
typedef Eigen::Matrix<uint8_t, 3, Eigen::Dynamic> Mat3Xu8;
typedef Eigen::Matrix<uint8_t, 4, Eigen::Dynamic> Mat4Xu8;

typedef Eigen::Array<uint8_t, 3, 1> RGB8;
typedef Eigen::Array<uint8_t, 4, 1> RGBA8;
typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
typedef Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu16;

// Aligned Eigen containers
template <typename Type>
using AlignedVector = std::vector<Type, Eigen::aligned_allocator<Type>>;

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class.
#define DISABLE_COPY_CONSTRUCTOR_AND_ASSIGNMENT(TypeName) \
 private:                                                 \
  TypeName(const TypeName&);                              \
  void operator=(const TypeName&)

//-----------------------------------------------------------------------------

template <int d, typename T>
inline Eigen::Matrix<T, d, 1> ToVec(std::initializer_list<T> arg) {
  const int dim = (d < 0 ? arg.size() : std::min(d, int(arg.size())));
  Eigen::Matrix<T, d, 1> ret = Eigen::Matrix<T, d, 1>::Zero();
  auto it = arg.begin();
  for (int i = 0; i < dim; ++i) ret[i] = *it++;
  return ret;
}

template <typename T>
inline rvl::Vec2 ToVec2(const std::pair<T, T>& p) {
  return rvl::Vec2(p.first, p.second);
}

template <typename T>
inline double Deg2Rad(T deg) {
  return static_cast<double>(deg) / 180.0 * M_PI;
}

template <typename T>
inline double Rad2Deg(T rad) {
  return static_cast<double>(rad) / M_PI * 180.0;
}

template <typename T, int r, int c>
inline Eigen::Matrix<double, r, c> Deg2Rad(const Eigen::Matrix<T, r, c>& deg) {
  return deg.template cast<double>() / 180.0 * M_PI;
}

template <typename T, int r, int c>
inline Eigen::Matrix<double, r, c> Rad2Deg(const Eigen::Matrix<T, r, c>& deg) {
  return deg.template cast<double>() * 180.0 / M_PI;
}

template <typename T, int r, int c>
inline Eigen::Matrix<T, r, c> FilterMatrix(
    const Eigen::Matrix<T, r, c>& mat, const rvl::Mask& cond,
    std::optional<int> num_valid = std::nullopt,
    std::vector<int>* out_indices = nullptr) {
  const int nvalid = (num_valid ? *num_valid : cond.count());
  if (out_indices) {
    out_indices->clear();
    out_indices->reserve(nvalid);
  }
  Eigen::Matrix<T, r, Eigen::Dynamic> out(mat.rows(), nvalid);
  for (int n = 0, idx = 0; n < mat.cols(); n++) {
    if (!cond(n)) continue;
    out.col(idx++) = mat.col(n);
    if (out_indices) out_indices->push_back(n);
  }
  return out;
}


struct ErrorExceptionThrower {
  std::stringstream ss;
  ~ErrorExceptionThrower() noexcept(false) {
    // In general, it is bad to throw an exception in a destructor...
    throw std::runtime_error(ss.str());
  }
  std::ostream& stream() { return ss; }  // hack to use binary << operators
};

#define THROW_ERR() rvl::ErrorExceptionThrower().stream()
#define THROW_ERR_IF(cond) \
  if (cond) rvl::ErrorExceptionThrower().stream()

#define THROW_ERRSTR(str) throw std::runtime_error(str)

}  // namespace rvl

#endif  // _RVL_COMMON_TYPES_H_

