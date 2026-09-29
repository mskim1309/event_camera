// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim  
//   

#include "util/file_io_util.h"

#include <dirent.h>     // opendir
#include <sys/stat.h>   // mkdir
#include <sys/types.h>  // mkdir

#include <glog/logging.h>

#include "util/rvl_common_types.h"
#include "util/stl_util.h"

using namespace std;

namespace rvl {

bool MakeDirRecursively(const string& path, bool* already_exist) {
  mode_t mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
  size_t idx = path.find_first_of("/\\", 0);
  if (idx == 0) idx = path.find_first_of("/\\", idx + 1);
  while (idx < path.length()) {
    string sub_path = path.substr(0, idx);
    if (::mkdir(sub_path.c_str(), mode) != 0 && errno != EEXIST) return false;
    idx = path.find_first_of("/\\", idx + 1);
  }
  if (path.back() != '/' && path.back() != '\\') {
    if (::mkdir(path.c_str(), mode) != 0 && errno != EEXIST) return false;
  }
  if (already_exist) *already_exist = (errno == EEXIST);
  return true;
}

}  // namespace rvl
