// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim   
//   

#ifndef _UTIL_STL_UTIL_H_
#define _UTIL_STL_UTIL_H_

#include <math.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <glog/stl_logging.h>

#include "util/rvl_common_types.h"

namespace rvl {
//-----------------------------------------------------------------------------

template <typename K, typename V>
inline const V& FindInMap(const std::map<K, V>& the_map, const K& key,
                          const V& def) {
  typename std::map<K, V>::const_iterator it = the_map.find(key);
  return it == the_map.end() ? def : it->second;
}

template <typename K, typename V>
inline V& FindInMap(std::map<K, V>& the_map, const K& key, V& def) {
  typename std::map<K, V>::iterator it = the_map.find(key);
  return it == the_map.end() ? def : it->second;
}

template <typename K, typename V>
inline bool FindInMap(const std::map<K, V>& the_map, const K& key, V* value) {
  typename std::map<K, V>::const_iterator it = the_map.find(key);
  if (it == the_map.end()) return false;
  if (value != NULL) *value = it->second;
  return true;
}

template <typename K, typename V>
inline bool FindInMap(std::map<K, V>& the_map, const K& key, V* value) {
  typename std::map<K, V>::iterator it = the_map.find(key);
  if (it == the_map.end()) return false;
  if (value != NULL) *value = it->second;
  return true;
}

//-----------------------------------------------------------------------------

template <typename K, typename V>
inline const V& FindInMap(const std::unordered_map<K, V>& the_map, const K& key,
                          const V& def) {
  auto it = the_map.find(key);
  return it == the_map.end() ? def : it->second;
}

template <typename K, typename V>
inline V& FindInMap(std::unordered_map<K, V>& the_map, const K& key, V& def) {
  auto it = the_map.find(key);
  return it == the_map.end() ? def : it->second;
}

template <typename K, typename V>
inline bool FindInMap(const std::unordered_map<K, V>& the_map, const K& key,
                      V* value) {
  auto it = the_map.find(key);
  if (it == the_map.end()) return false;
  if (value != NULL) *value = it->second;
  return true;
}

template <typename K, typename V>
inline bool FindInMap(std::unordered_map<K, V>& the_map, const K& key,
                      V* value) {
  auto it = the_map.find(key);
  if (it == the_map.end()) return false;
  if (value != NULL) *value = it->second;
  return true;
}

//-----------------------------------------------------------------------------

template <int BUF_SIZE = 4096>
inline std::string FormattedString(const char* format, ...) {
  char buf[BUF_SIZE];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, BUF_SIZE, format, args);
  va_end(args);
  return std::string(buf);
}

inline bool StartsWith(const std::string& str, const std::string& pattern) {
  return (str.rfind(pattern, 0) == 0);
}

inline std::string GetDirFromPath(const std::string& file_path) {
  size_t dir_pos = file_path.find_last_of("/\\");
  if (dir_pos == std::string::npos) dir_pos = 0;
  return file_path.substr(0, dir_pos);
}

inline std::string GetFileNameFromPath(const std::string& file_path) {
  size_t dir_pos = file_path.find_last_of("/\\");
  if (dir_pos == std::string::npos) return file_path;
  return file_path.substr(dir_pos + 1);
}

// Get file extension excluding dot '.', e.g., "abc.png" -> "png"
inline std::string GetFileExtension(const std::string& file_path) {
  size_t dot_pos = file_path.rfind(".");
  if (dot_pos == std::string::npos) return std::string();
  size_t dir_pos = file_path.find_last_of("/\\");
  if (dir_pos != std::string::npos && dot_pos < dir_pos) return std::string();
  std::string ext = file_path.substr(dot_pos);
  if (!ext.empty()) ext = ext.substr(1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

inline std::string JoinPath(const std::string& p) {
  return p;
}

template <typename... Args>
inline std::string JoinPath(const std::string& p, Args... args) {
  std::string q = JoinPath(args...);
  if (p.empty()) return q;
  if (q.empty()) return p;
  constexpr char sep = '/';
  std::string path = p;
  if (path[path.length() - 1] != sep) path += sep;
  path += (q[0] == sep ? q.substr(1, q.length() - 1) : q);
  return path;
}

inline size_t SplitStr(const std::string& str, const std::string& delim,
                       std::vector<std::string>* tokens) {
  tokens->clear();
  size_t i, j, npos = std::string::npos;
  for (i = 0; (i = str.find_first_not_of(delim, i)) != npos; i = j) {
    j = str.find_first_of(delim, i);
    if (j == npos) tokens->push_back(str.substr(i));
    else tokens->push_back(str.substr(i, j - i));
  }
  return tokens->size();
}

inline std::vector<std::string> SplitStr(const std::string& str,
                                         const std::string& delim) {
  std::vector<std::string> tokens;
  SplitStr(str, delim, &tokens);
  return tokens;
}

//-----------------------------------------------------------------------------

inline void RandomSample(int n, int k, std::vector<int>* samples) {
  if (n < k) n = k;
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  samples->resize(k);
  for (int i = 0; i < k; ++i) {
    int j = i + (rand() % (n - i));
    std::swap(idx[i], idx[j]);
    samples->at(i) = idx[i];
  }
}

inline int RandomSampleCDF(const std::vector<double>& dist) {
  // dist(cumulative distribution) should be nonnegative and non-decreasing
  // and should end with 1.0
  const double p = double(rand()) / RAND_MAX;
  return std::lower_bound(dist.begin(), dist.end(), p) - dist.begin();
}

}  // namespace rvl
#endif  // _UTIL_STL_UTIL_H_
