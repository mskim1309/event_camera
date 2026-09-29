// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim   
//   

#ifndef _UTIL_YAML_H_
#define _UTIL_YAML_H_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "util/rvl_common_types.h"
#include "sigslot/signal.hpp"

DECLARE_string(config);

namespace rvl {

class Flag {
 public:
  inline static sigslot::signal<> after_init_listener{};
  static constexpr std::vector<std::string>& GetNotExplicitlySet() {
    return not_explicitly_set_;
  }

  static void InvokeAfterInit(bool check_explicitly = false);

 private:
  inline static std::vector<std::string> not_explicitly_set_{};
};

bool IsGoogleFlagDefaultValue(const char* flag_name);

bool UpdateGoogleFlag(const std::string& key, const std::string& value,
                      const std::string& log_prefix = "flag");

bool UpdateGoogleFlag(const std::string& name, const YAML::Node& node);

// Load the yaml config file and stores the path into the "_yaml_path" node.
// If the path is directory, tries to load yaml_path/config.yaml.
// Note that these functions may throw runtime_exceptions.
YAML::Node LoadYAMLConfigFile(const std::string& yaml_path);

// https://www.notion.so/YAML-config-5333f7c6aa9c4078b87385b29af2d349
// Load a YAML configuration file and its linked files recursively, and
// synchronize with the corresponding google flags. The priority for is
//   gflags default < YAML config file < gflags command-line argument.
// To upate a YAML node that does not have corresponding gflag, use
// --config_yaml "<yaml_str>". Use ';' for newline in <yaml_str>.
// Note that these functions may throw runtime_exceptions.
YAML::Node LoadYAMLConfigAndSyncGoogleFlags(const std::string& yaml_path);

// Load the application YAML configuration file.
YAML::Node LoadAppYAMLConfigAndSyncGoogleFlags(
    const std::string& app_path, std::string* yaml_path_out = nullptr);

// No exception& versions
bool LoadYAMLConfigFile(const std::string& yaml_path, YAML::Node* config);
bool LoadAppYAMLConfigAndSyncGoogleFlags(const std::string& app_path,
                                         YAML::Node* config,
                                         std::string* yaml_path_out = nullptr);

// Call the following function before google::ParseCommandLineFlags().
void UpdateDefaultLogToStdErrFlags(bool logtostderr = true,
                                   bool colorlogtostderr = true);

//-----------------------------------------------------------------------------

template <typename T>
inline std::vector<T> ParseYAMLAsVector(const YAML::Node& node) {
  std::vector<T> ret;
  THROW_ERR_IF(!node.IsDefined() || node.IsNull()) << "undefined or null node";
  if (node.IsScalar()) {
    std::stringstream ss(node.Scalar());
    while (!ss.eof()) {
      T val;
      ss >> val;
      ret.emplace_back(val);
    }
  } else {
    ret = node.as<std::vector<T>>();
  }
  return ret;
}

template <typename T>
inline std::vector<T> ParseYAMLAsVector(const YAML::Node& node,
                                        const std::vector<T>& def) {
  try {
    return ParseYAMLAsVector<T>(node);
  } catch (std::exception& e) {
    LOG(WARNING) << e.what();
    return def;
  }
}


template <typename T>
inline T GetValueIfDefined(const YAML::Node& config, const std::string& name,
                           const T& def_value) {
  YAML::Node node = config[name];
  return (node.IsDefined() ? node.as<T>() : def_value);
}

template <typename T>
inline void GetValueIfDefined(const YAML::Node& config, const std::string& name,
                              T* value) {
  YAML::Node node = config[name];
  if (value && node.IsDefined()) *value = node.as<T>();
}

std::string ExpandVariables(const std::string& str_, const YAML::Node& node);

// Export yaml node to string/file - use YAML::Load/YAML::LoadFile for import
std::string ExportToString(const YAML::Node& node, bool flow_style = true);
bool ExportToFile(const YAML::Node& node, const std::string& file_path);

//-----------------------------------------------------------------------------

}  // namespace rvl
#endif // _UTIL_YAML_H_
