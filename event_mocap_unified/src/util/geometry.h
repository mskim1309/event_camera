// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim   
//   

#ifndef _GEOM_GEOMETRY_H_
#define _GEOM_GEOMETRY_H_

#include <glog/logging.h>
#include <math.h>
#include <iostream>
#include <map>
#include <vector>

#include "util/rvl_common_types.h"

namespace rvl {

// Fit a value into the period by adding or subtracting periods.
// e.g., FitPeriod(-pi, 2 * pi) = pi = FitPeriod(5 * pi, 2 * pi)
inline double FitPeriodic(double v, double period) {
  if (v < 0) v += period * ceil(-v / period);
  if (v >= period) v -= period * floor(v / period);
  return v;
}

inline Mat34 IdentityMat34() {
  Mat34 P = Mat4::Identity().block(0, 0, 3, 4);
  return P;
}

// UnitVector(vec) : use vec.normalized() instead.

template <typename T, int d, int n>
inline Eigen::Matrix<T, d + 1, n> Hom(const Eigen::Matrix<T, d, n>& x) {
  Eigen::Matrix<T, d + 1, n> h(x.rows() + 1, x.cols());
  h.block(0, 0, x.rows(), x.cols()) = x;
  h.row(x.rows()).fill(1.0);
  return h;
}

template <typename T, int d>
inline Eigen::Matrix<T, d + 1, 1> Hom(const Eigen::Matrix<T, d, 1>& x) {
  Eigen::Matrix<T, d + 1, 1> h;
  h << x, 1.0;
  return h;
}

template <typename T, int d, int n>
inline Eigen::Matrix<T, d - 1, n> Euc(const Eigen::Matrix<T, d, n>& h) {
  Eigen::Array<T, d - 1, n> x = h.block(0, 0, h.rows() - 1, h.cols()).array();
  Eigen::Array<T, 1, n> y = h.row(h.rows() - 1);
  x.rowwise() /= y;
  return x.matrix();
}

template <typename T, int d>
inline Eigen::Matrix<T, d - 1, 1> Euc(const Eigen::Matrix<T, d, 1>& h) {
  Eigen::Matrix<T, d - 1, 1> x = h.block(0, 0, d - 1, 1) / h(d - 1, 0);
  return x;
}

inline Mat3X Project(const Mat34& P, const Mat3X& X) {
  Mat3X x = P * Hom(X);
  return (x.array().rowwise() / x.row(2).array()).matrix();
}

inline Mat3 GetRot(const Mat34& P) {
  return P.block(0, 0, 3, 3);
}
inline Vec3 GetTr(const Mat34& P) {
  return P.block(0, 3, 3, 1);
}
inline void SetRot(const Mat3& R, Mat34* P) {
  P->block(0, 0, 3, 3) = R;
}
inline void SetTr(const Vec3& tr, Mat34* P) {
  P->block(0, 3, 3, 1) = tr;
}

inline Vec3 GetRotVec(const Vec6& pose) {
  return pose.block(0, 0, 3, 1);
}
inline Vec3 GetTr(const Vec6& pose) {
  return pose.block(3, 0, 3, 1);
}
inline void SetRotVec(const Vec3& rot, Vec6* p) {
  p->block(0, 0, 3, 1) = rot;
}
inline void SetTr(const Vec3& tr, Vec6* p) {
  p->block(3, 0, 3, 1) = tr;
}

inline Vec3 ApplyTransformVector(const Mat34& pose, const Vec3& pt) {
  // TODO(jwlim): which version is better?
  // return pose * Hom(pt);
  Vec3 out;
  const double x = pt(0), y = pt(1), z = pt(2);
  out(0) = pose(0, 0) * x + pose(0, 1) * y + pose(0, 2) * z + pose(0, 3);
  out(1) = pose(1, 0) * x + pose(1, 1) * y + pose(1, 2) * z + pose(1, 3);
  out(2) = pose(2, 0) * x + pose(2, 1) * y + pose(2, 2) * z + pose(2, 3);
  return out;
}

inline Mat3X ApplyTransform(const Mat34& pose, const Mat3X& pts) {
  return (GetRot(pose) * pts).colwise() + GetTr(pose);
}

inline void ApplyTransform(const Mat34& pose, const Mat3X& pts, Mat3X* out) {
  const int num_pts = pts.cols();
  if (out->rows() != 3 || out->cols() != num_pts) out->resize(3, num_pts);
  *out = (GetRot(pose) * pts).colwise() + GetTr(pose);
}

template <typename T>
inline Eigen::Matrix<T, 3, -1> ApplyTransform(
    const Mat34& pose, const Eigen::Matrix<T, 3, -1>& pts) {
  return (GetRot(pose).cast<T>() * pts).colwise() + GetTr(pose).cast<T>();
}

template <typename T>
inline void ApplyTransform(const Mat34& pose,
                           const Eigen::Matrix<T, 3, -1>& pts,
                           Eigen::Matrix<T, 3, -1>* out) {
  const int num_pts = pts.cols();
  if (out->rows() != 3 || out->cols() != num_pts) out->resize(3, num_pts);
  *out = (GetRot(pose).cast<T>() * pts).colwise() + GetTr(pose).cast<T>();
}

template <typename T>
inline Eigen::Matrix<T, 3, 1> ApplyTransformVector(
    const Mat34& pose, const Eigen::Matrix<T, 3, 1>& pt) {
  Eigen::Matrix<T, 3, 1> out;
  const T x = pt(0), y = pt(1), z = pt(2);
  out(0) = pose(0, 0) * x + pose(0, 1) * y + pose(0, 2) * z + pose(0, 3);
  out(1) = pose(1, 0) * x + pose(1, 1) * y + pose(1, 2) * z + pose(1, 3);
  out(2) = pose(2, 0) * x + pose(2, 1) * y + pose(2, 2) * z + pose(2, 3);
  return out;
}

////////////////////////////////////////////////////////////////////////////////////


inline Vec3 toRVec(const Mat3& R){
  Eigen::AngleAxisd Raa(R); return Raa.axis() * Raa.angle();
}
inline Vec3 toRVec(const Quat& q){
  Eigen::AngleAxisd Raa(q.normalized()); return Raa.axis() * Raa.angle();
}
inline Quat toQuat(const Mat3& R){
  Quat ret(R); return ret.normalized();
}
inline Mat3 toRMat(const Vec3& r){
  if(r.norm() > 1e-7){
    Eigen::AngleAxisd Raa(r.norm(), r / r.norm());
    return Raa.toRotationMatrix();
  } else return Mat3::Identity(); 
}
inline Quat toQuat(const Vec3& r){
  Quat ret(toRMat(r)); return ret.normalized();
}
inline Mat3 toRMat(const Quat& q){
  return q.normalized().toRotationMatrix(); 
}

inline Mat3 get_R(const Mat34& T){ return T.block<3,3>(0,0); }

inline Mat3 get_R(const Vec6& T){ return toRMat(T.block<3,1>(0,0)); }

inline Vec3 get_t(const Mat34& T){ return T.block<3,1>(0,3); }

inline Vec3 get_t(const Vec6& T){ return T.block<3,1>(3,0); }

inline Vec6 toVec6(const Mat34& T){
  Vec6 ret = Vec6::Zero();
  ret.block<3,1>(0,0) = toRVec(get_R(T));
  ret.block<3,1>(3,0) = get_t(T);
  return ret;
}

inline Mat34 toMat34(const Vec6& T){
  Mat34 ret = Mat34::Identity();
  ret.block<3,3>(0,0) = get_R(T);
  ret.block<3,1>(0,3) = get_t(T);
  return ret;
}

inline Mat34 InvT(const Mat34& T){
  Mat34 ret = Mat34::Identity();
  ret.block<3,3>(0,0) = get_R(T).transpose();
  ret.block<3,1>(0,3) = -get_R(T).transpose()*get_t(T);
  return ret;
}

inline Vec6 InvT(const Vec6& T){
  return toVec6(InvT(toMat34(T)));
}

inline Mat34 MergeT(const Mat34& T1, const Mat34& T2){
  Mat34 ret = Mat34::Identity();
  ret.block<3,3>(0,0) = get_R(T1)*get_R(T2);
  ret.block<3,1>(0,3) = get_R(T1)*get_t(T2) + get_t(T1);
  return ret;
}

inline Vec6 MergeT(const Vec6& T1, const Vec6& T2){
  return toVec6(MergeT(toMat34(T1), toMat34(T2)));
}

inline Vec3 ApplyTVec(const Mat34& T, const Vec3& V){ 
  return(get_R(T)*V + get_t(T));
}

inline Mat3X ApplyT(const Mat34& T, const Mat3X& Vs){ 
  return((get_R(T)*Vs).colwise() + get_t(T)); 
}

inline Vec3 ApplyTVec(const Vec6& T, const Vec3& V){ 
  return(get_R(T)*V + get_t(T));
}

inline Mat3X ApplyT(const Vec6& T, const Mat3X& Vs){ 
  return((get_R(T)*Vs).colwise() + get_t(T)); 
}

///////////////////////////////////////////////////////////////////////////////////

inline Mat3 CrossProductMatrix(const Vec3& x) {
  Mat3 X;
  X << 0, -x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
  return X;
}

inline Mat3 AxisAngleRotationMatrix(const Vec3& axis) {
  double theta = axis.norm();
  if (theta < 1e-14) {
    return Mat3::Identity();
  }
  Vec3 w = axis / theta;
  Mat3 W = CrossProductMatrix(w);
  return Mat3::Identity() + sin(theta) * W + (1 - cos(theta)) * W * W;
}

inline Vec3 AxisAngleRotationVector(const Mat3& mat) {
  Eigen::AngleAxisd angle_axis;
  angle_axis.fromRotationMatrix(mat);
  return angle_axis.axis() * angle_axis.angle();
}

inline Vec3 QuatToAxisAngleRotationVector(const Vec4& quat) {
  Vec3 q_ = quat.head(3);
  double qnorm = q_.norm();
  if (qnorm <= 1e-14) {
    return Vec3::Zero();
  }
  return q_ / qnorm * 2 * acos(quat(3));
}

inline Vec4 QuaternionRotationVector(const Vec3& axis_angle) {
  double theta = axis_angle.norm();
  if (theta < 1e-14) {
    return Vec4(0, 0, 0, 1);
  }
  Vec3 q = axis_angle / theta * sin(theta / 2.0);
  return Vec4(q.x(), q.y(), q.z(), cos(theta / 2.0));
}

inline Mat3 QuaternionRotationMatrix(const Vec4& quat) {
  Vec3 q_ = quat.head(3);
  double w = quat(3);
  return (w * w - q_.dot(q_)) * Mat3::Identity() + 2 * q_ * q_.transpose() +
         2 * w * CrossProductMatrix(q_);
}

inline Mat3 EulerRotationMatrix(const Vec3& euler_rad) {
  return AxisAngleRotationMatrix(Vec3(euler_rad.x(), 0, 0)) *
         AxisAngleRotationMatrix(Vec3(0, euler_rad.y(), 0)) *
         AxisAngleRotationMatrix(Vec3(0, 0, euler_rad.z()));
}

inline Vec3 EulerRotationVector(const Mat3& mat) {
  const double s = sqrt(mat(0, 0) * mat(0, 0) + mat(1, 0) * mat(1, 0));
  const double y = atan2(-mat(2, 0), s);
  if (s > 1e-9) {
    return Vec3(atan2(mat(2, 1), mat(2, 2)), y, atan2(mat(1, 0), mat(0, 0)));
  } else {
    return Vec3(atan2(mat(1, 2), mat(1, 1)), y, 0);
  }
}

inline Mat34 ToPoseMatrix(const Vec6& param) {
  Mat34 P;
  P << AxisAngleRotationMatrix(param.segment(0, 3)), param.segment(3, 3);
  return P;
}

inline Vec6 ToPoseVector(const Mat34& mat) {
  const Vec3 rot = AxisAngleRotationVector(mat.block(0, 0, 3, 3));
  const Vec3 tr = mat.block(0, 3, 3, 1);
  Vec6 rt;
  rt << rot, tr;
  return rt;
}

inline Mat4 ToMat4(const Mat34& P) {
  Mat4 H;
  H << P, 0.0, 0.0, 0.0, 1.0;
  return H;
}

inline Mat34 ToMat34(const Mat4& H) {
  Mat34 P = H.block(0, 0, 3, 4);
  return P;
}

inline Mat34 ToMat34(const Mat3& R, const Vec3& t) {
  Mat34 P;
  P.block(0, 0, 3, 3) = R;
  P.block(0, 3, 3, 1) = t;
  return P;
}

inline Mat34 ToMat34(const Vec3& r, const Vec3& t) {
  Mat3 R = AxisAngleRotationMatrix(r);
  return ToMat34(R, t);
}

inline Mat4 ToMat4(const Mat3& R, const Vec3& t) {
  Mat34 P;
  P.block(0, 0, 3, 3) = R;
  P.block(0, 3, 3, 1) = t;
  return ToMat4(P);
}

inline Mat4 ToMat4(const Vec3& r, const Vec3& t) {
  Mat3 R = AxisAngleRotationMatrix(r);
  return ToMat4(R, t);
}

inline Mat34 ScaledTransform(double scale, Mat34 P) {
  P.col(3) *= scale;
  return P;
}

inline Mat34 InverseTransform(const Mat34& P, bool rigid = false) {
  return ToMat34(ToMat4(P).inverse());
}

inline Mat34 MergedTransform(const Mat34& P1, const Mat34& P2) {
  return P1 * ToMat4(P2);
}

inline Mat34 RelativeTransform(const Mat34& P_ref, const Mat34& P) {
  return P * ToMat4(InverseTransform(P_ref));
  //  return InverseTransform(P_ref) * ToMat4(P);
}

inline Vec6 ScaledTransform(double scale, Vec6 pose) {
  pose.segment<3>(3) *= scale;
  return pose;
}

inline Vec6 InverseTransform(const Vec6& pose) {
  return ToPoseVector(ToMat34(ToMat4(ToPoseMatrix(pose)).inverse()));
}

inline Vec6 MergedTransform(const Vec6& pose1, const Vec6& pose2) {
  return ToPoseVector(ToPoseMatrix(pose1) * ToMat4(ToPoseMatrix(pose2)));
}

inline Vec6 RelativeTransform(const Vec6& pose_ref, const Vec6& pose) {
  return ToPoseVector(
      RelativeTransform(ToPoseMatrix(pose_ref), ToPoseMatrix(pose)));
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> SkewSymmetric(
    const Eigen::MatrixBase<Derived>& v) {
  Eigen::Matrix<typename Derived::Scalar, 3, 3> mat;
  mat << typename Derived::Scalar(0), -v(2), v(1), v(2),
      typename Derived::Scalar(0), -v(0), -v(1), v(0),
      typename Derived::Scalar(0);
  return mat;
}

template <typename Derived>
inline Eigen::Quaternion<typename Derived::Scalar> DeltaQ(
    const Eigen::MatrixBase<Derived>& theta) {
  typedef typename Derived::Scalar T;
  Eigen::Matrix<T, 3, 1> half_theta = theta / static_cast<T>(2.0);
  return Eigen::Quaternion<T>(static_cast<T>(1.0), half_theta.x(),
                              half_theta.y(), half_theta.z());
}

template <typename T>
inline void InverseRot(const T* rot, T* rot_inv) {
  rot_inv[0] = -rot[0];
  rot_inv[1] = -rot[1];
  rot_inv[2] = -rot[2];
}

inline rvl::Mat3 GetRotationMatFromTwoVector(const rvl::Vec3& a,
                                             const rvl::Vec3& b) {
  return Eigen::Quaterniond::FromTwoVectors(a.normalized(), b.normalized())
      .normalized()
      .toRotationMatrix();
}

inline Mat3 ExpMap(const Vec3& w) {
  const double eps = 1e-5;
  double a = w.norm();
  Mat3 W = CrossProductMatrix(w);
  Mat3 I = Mat3::Identity();
  if (a < eps) {
    return I + W + 0.2 * W * W;
  } else {
    return I + W * (sin(a) / a) + W * W * ((1 - cos(a)) / a / a);
  }
}

inline Vec3 LogMap(const Mat3& R) {
  const double eps = 1e-5;
  double a = acos((R.trace() - 1) / 2);
  Vec3 w(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  if (a < eps) {
    w *= 0.5;
  } else if (a < M_PI - eps) {
    w *= a / 2 / sin(a);
  } else {
    Mat3 S = 0.5 * (R - Mat3::Identity());
    double b = sqrt(S(0, 0) + 1);
    double c = sqrt(S(1, 1) + 1);
    double d = sqrt(S(2, 2) + 1);
    if (b > eps) {
      c = S(1, 0) / b;
      d = S(2, 0) / b;
    } else if (c > eps) {
      b = S(0, 1) / c;
      d = S(2, 1) / c;
    } else {
      b = S(0, 2) / d;
      c = S(1, 2) / d;
    }
    w(0) = b;
    w(1) = c;
    w(2) = d;
  }
  return w;
}

// Matrix row-major (= column-major in images) grid points
inline Mat2X GridPoints(const int w_cols, const int h_rows,
                        const int c_convention = 0,
                        const bool row_major = true) {
  const int num_pts = h_rows * w_cols;
  Mat2X p(2, num_pts);
  if (row_major) {
    for (int i = 0, k = 0; i < w_cols; ++i) {
      for (int j = 0; j < h_rows; ++j, ++k) {
        p.col(k) << i + c_convention, j + c_convention;
      }
    }
  } else {
    for (int j = 0, k = 0; j < h_rows; ++j) {
      for (int i = 0; i < w_cols; ++i, ++k) {
        p.col(k) << i + c_convention, j + c_convention;
      }
    }
  }
  return p;
}

inline rvl::Mat3 InterpolateRotation(const rvl::Mat3& P1, const rvl::Mat3& P2,
                                     double ratio) {
  const Eigen::Quaterniond q1(P1.block<3, 3>(0, 0));
  const Eigen::Quaterniond q2(P2.block<3, 3>(0, 0));
  Eigen::Quaterniond q_interp = q1.slerp(ratio, q2);
  return q_interp.normalized().toRotationMatrix();
}

}  // namespace rvl
#endif  // _GEOM_GEOMETRY_H_
