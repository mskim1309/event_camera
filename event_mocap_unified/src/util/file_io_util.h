// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim   
//   

#ifndef _UTIL_FILE_IO_UTIL_H_
#define _UTIL_FILE_IO_UTIL_H_

#include <dirent.h>     // opendir
#include <sys/stat.h>   // mkdir
#include <sys/types.h>  // mkdir

#include <algorithm>
#include <fstream>
#include <map>
#include <vector>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "util/rvl_common_types.h"
#include "util/stl_util.h"


using namespace std;

namespace rvl {


bool MakeDirRecursively(const std::string& path, bool* already_exist = nullptr);


}  // namespace rvl
#endif // _UTIL_FILE_IO_UTIL_H_
