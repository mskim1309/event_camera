// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim   
//   

#include "util/yaml.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <regex>

#include <gflags/gflags.h>

#include "util/file_io_util.h"
#include "util/rvl_common_types.h"
#include "util/stl_util.h"
#include "util/google_log.h"

DEFINE_string(config, "", "Application YAML config file path");
DEFINE_string(config_yaml_str, "", "YAML string to update the config");
DEFINE_bool(show_updated_flags, true, "Show updated gflags and YAML config");
DEFINE_bool(show_app_config, false, "Show the final app config");

DECLARE_bool(logtostderr);
DECLARE_bool(colorlogtostderr);

using namespace std;

namespace rvl {

bool IsGoogleFlagDefaultValue(const char* flag_name) {
  google::CommandLineFlagInfo flag_info;
  return google::GetCommandLineFlagInfo(flag_name, &flag_info) &&
         flag_info.is_default;
}

bool UpdateGoogleFlag(const string& key, const string& value,
                      const string& log_prefix) {
  google::CommandLineFlagInfo flag_info;
  if (!google::GetCommandLineFlagInfo(key.c_str(), &flag_info)) {
    LOG(ERROR) << log_prefix << " '" << key << "' is not found";
    return false;
  } else if (!flag_info.is_default) {
    LOG(INFO) << log_prefix << " '" << key << "' is already set ("
              << flag_info.current_value << ") - ignored";
  } else if (google::SetCommandLineOptionWithMode(key.c_str(), value.c_str(),
                                                  google::SET_FLAGS_DEFAULT)
                 .empty()) {
    LOG(ERROR) << log_prefix << " failed to set '" << key << "' with '" << value
               << "'";
    return false;
  } else {
    LOG(INFO) << log_prefix << " '" << key << "' is set to '" << value << "'";
  }
  return true;
}

bool UpdateGoogleFlag(const string& name, const YAML::Node& node) {
  if (node.IsDefined()) return UpdateGoogleFlag(name, node.as<string>());
  return false;
}

//-----------------------------------------------------------------------------

string ExpandVariables(const string& str_, const YAML::Node& node) {
  string str = str_;
  const size_t str_npos = string::npos;
  size_t pos = 0, end = str_npos;
  for (; (pos = str.find('$', pos)) != str_npos; pos = 0) {
    bool bracket = false;
    if (pos + 1 >= str.size() || ((bracket = (str[pos + 1] == '{')) &&
                                  (end = str.find('}', pos)) == str_npos)) {
      THROW_ERR() << "syntax error '" << str << "' near " << pos << ":" << end
                  << "/" << bracket;
    }
    if (!bracket && (end = str.find(' ', pos)) == str_npos) end = str.size();
    const string var_name = (bracket ? str.substr(pos + 2, end - pos - 2)
                                     : str.substr(pos + 1, end - pos - 1));
    YAML::Node var_node = node[var_name];
    THROW_ERR_IF(!var_node.IsDefined())
        << "variable '" << var_name << "' not defined";
    try {
      str.replace(pos, bracket ? end - pos + 1 : end - pos,
                  var_node.as<string>());
    } catch (exception& e) {
      THROW_ERR() << "failed to convert variable '" << var_name
                  << "' to string: " << e.what();
    }
  }
  return str;
}

YAML::Node LoadYAMLConfigFile(const string& yaml_path) {
  // switch (IsDir(yaml_path)) {
  //   case -1: THROW_ERR() << "yaml_path '" << yaml_path << "' does not exist";
  //   case 1: return LoadYAMLConfigFile(JoinPath(yaml_path, "config.yaml"));
  //   default:  // regular file
  //     auto node = YAML::LoadFile(yaml_path.c_str());
  //     node["_yaml_path"] = yaml_path;
  //     return node;
  // }
  struct stat path_stat;
  THROW_ERR_IF(stat(yaml_path.c_str(), &path_stat) != 0)  // not exist
      << "yaml_path '" << yaml_path << "' does not exist";
  if (S_ISREG(path_stat.st_mode)) {  // regular file
    auto node = YAML::LoadFile(yaml_path.c_str());
    node["_yaml_path"] = yaml_path;
    return node;
  }
  return LoadYAMLConfigFile(JoinPath(yaml_path, "config.yaml"));
}

static void UpdateGoogleFlagRecursively(const string& name, YAML::Node node,
                                        const YAML::Node& root) {
  if (node.IsScalar()) {
    google::CommandLineFlagInfo flag_info;
    if (google::GetCommandLineFlagInfo(name.c_str(), &flag_info)) {
      if (flag_info.is_default) {
        const string value = ExpandVariables(node.as<string>(), root);
        auto ret = google::SetCommandLineOptionWithMode(
            name.c_str(), value.c_str(), google::SET_FLAGS_VALUE);
        THROW_ERR_IF(ret.empty())
            << "failed to update '" << name << "' to " << value;
        if (FLAGS_show_updated_flags) {
          LOG(INFO) << "- FLAGS_" << name << " <- " << value;
        }
      } else {
        node = flag_info.current_value;
        if (FLAGS_show_updated_flags) {
          LOG(INFO) << "- yaml " << name << " <- " << node;
        }
      }
    }
  } else if (node.IsMap()) {
    for (const auto& n : node) {
      const string key = n.first.as<string>();
      UpdateGoogleFlagRecursively(name + "_" + key, n.second, root);
    }
  }
}

static YAML::Node LoadYAMLConfigAndSyncGoogleFlagsRecursively(
    YAML::Node node, const string& parent_name_, const char* yaml_dir = ".") {
  vector<string> keys_to_load;
  for (const auto& n : node) {
    const string& key = n.first.as<string>();
    if (StartsWith(key, "load_")) {
      THROW_ERR_IF(key.size() <= 5) << "empty node_name for load " << key;
      keys_to_load.emplace_back(key);
    } else {
      UpdateGoogleFlagRecursively(parent_name_ + key, n.second, node);
    }
  }
  for (const auto& key : keys_to_load) {
    const string node_name = key.substr(5);
    THROW_ERR_IF(node[node_name].IsDefined())
        << "sub-node " << key << " already exist";
    string yaml_path = ExpandVariables(node[key].as<string>(), node);
    THROW_ERR_IF(yaml_path.empty()) << "empty yaml_path for " << key;
    YAML::Node sub_node;
    yaml_path = JoinPath(yaml_dir, yaml_path);
    try {
      sub_node = LoadYAMLConfigFile(yaml_path);
    } catch (exception& e) {
      THROW_ERR() << "failed to load '" << yaml_path << "' for " << key << ": "
                  << e.what();
    }
    const string new_yaml_dir = GetDirFromPath(yaml_path);
    node[node_name] = LoadYAMLConfigAndSyncGoogleFlagsRecursively(
        sub_node, parent_name_ + node_name + "_", new_yaml_dir.c_str());
    node.remove(key);  // erase load_ node
  }
  if (VLOG_IS_ON(1)) {
    LOG(WARNING) << "LoadYAMLConfigAndSyncGoogleFlagsRecursively ------- "
                 << parent_name_ << endl
                 << node;
  }
  return node;
}

YAML::Node LoadYAMLConfigAndSyncGoogleFlags(const string& yaml_path) {
  const string yaml_dir = GetDirFromPath(yaml_path);
  VLOG(1) << "LoadYAMLConfigAndSyncGoogleFlags: " << yaml_path << ", dir='"
          << yaml_dir << "'";
  return LoadYAMLConfigAndSyncGoogleFlagsRecursively(
      LoadYAMLConfigFile(yaml_path), "", yaml_dir.c_str());
}

static YAML::Node UpdateYAMLConfigAndGoogleFlagsRecursively(
    YAML::Node config, YAML::Node node, const string& parent_name_) {
  for (const auto& n : node) {
    const string& key = n.first.as<string>();
    THROW_ERR_IF(StartsWith(key, "load_"))
        << "loading file in config_yaml_str is not supported: " << key;
    if (n.second.IsMap()) {
      config[key] = UpdateYAMLConfigAndGoogleFlagsRecursively(
          config[key], n.second, parent_name_ + key + "_");
    } else {
      config[key] = n.second;
      UpdateGoogleFlagRecursively(parent_name_ + key, n.second, node);
    }
  }
  return config;
}

YAML::Node LoadAppYAMLConfigAndSyncGoogleFlags(const string& app_path,
                                               string* yaml_path_out) {
  // google::ParseCommandLineFlags(&argc, &argv, true);
  // google::InitGoogleLogging(argv[0]);
  string yaml_path = FLAGS_config;
  if (yaml_path.empty()) {
    size_t name_pos = app_path.find_last_of("/\\");
    size_t ext_pos = app_path.find_last_of('.');
    name_pos = (name_pos == string::npos ? 0 : name_pos + 1);
    if (ext_pos < name_pos) ext_pos = string::npos;
    const string app_name = app_path.substr(
        name_pos, ext_pos == string::npos ? ext_pos : ext_pos - name_pos);
    yaml_path = app_name + ".yaml";
    struct stat path_stat;
    if (stat(yaml_path.c_str(), &path_stat) != 0) {
      const string app_dir = app_path.substr(0, name_pos);
      const string yaml_path_in_app_dir = app_dir + app_name + ".yaml";
      if (stat(yaml_path_in_app_dir.c_str(), &path_stat) == 0) {
        yaml_path = yaml_path_in_app_dir;
      }
    }
  }
  if (yaml_path_out) *yaml_path_out = yaml_path;
  YAML::Node config = LoadYAMLConfigAndSyncGoogleFlags(yaml_path);
  if (!FLAGS_config_yaml_str.empty()) {
    string yaml_str = FLAGS_config_yaml_str;
    yaml_str = regex_replace(yaml_str, regex("; *"), "\n");
    YAML::Node node = YAML::Load(yaml_str);
    config = UpdateYAMLConfigAndGoogleFlagsRecursively(config, node, "");
  }
  if (FLAGS_show_app_config) LOG(INFO) << "app_config:" << endl << config;
  return config;
}

//-----------------------------------------------------------------------------

bool LoadYAMLConfigFile(const string& yaml_path, YAML::Node* config) {
  CHECK_NOTNULL(config);
  try {
    *config = LoadYAMLConfigFile(yaml_path);
    return true;
  } catch (exception& e) {
    LOG(WARNING) << "LoadYAMLConfigFile: failed to load config: " << e.what();
    return false;
  }
}

bool LoadAppYAMLConfigAndSyncGoogleFlags(const string& app_path,
                                         YAML::Node* config,
                                         string* yaml_path_out) {
  CHECK_NOTNULL(config);
  try {
    *config = LoadAppYAMLConfigAndSyncGoogleFlags(app_path, yaml_path_out);
    Flag::GetNotExplicitlySet().clear();
    Flag::after_init_listener();
    Flag::InvokeAfterInit(true);
    return true;
  } catch (exception& e) {
    LOG(WARNING) << "LoadAppYAMLConfigAndSyncGoogleFlags: "
                    "failed to load config: "
                 << e.what();
    return false;
  }
}

void UpdateDefaultLogToStdErrFlags(bool logtostderr, bool colorlogtostderr) {
  FLAGS_alsologtostderr = logtostderr;
  FLAGS_colorlogtostderr = colorlogtostderr;
}

string ExportToString(const YAML::Node& node, bool flow_style) {
  YAML::Emitter emitter;
  if (flow_style) {
    YAML::Node export_node = node;
    export_node.SetStyle(YAML::EmitterStyle::Flow);
    emitter << export_node;
  } else {
    emitter << node;
  }
  // emitter << YAML::Flow << node;
  return string(emitter.c_str());
}

bool ExportToFile(const YAML::Node& node, const string& file_path) {
  ofstream ofs(file_path);
  if (!ofs.is_open()) {
    LOG(ERROR) << "failed to export yaml node to file " << file_path;
    return false;
  }
  const string yaml_str = ExportToString(node, false);
  ofs.write(&yaml_str[0], yaml_str.size());
  ofs.close();
  return true;
}
void Flag::InvokeAfterInit(bool check_explicitly) {
  after_init_listener();
  if (check_explicitly && !not_explicitly_set_.empty()) {
    for (const auto& flag_name : not_explicitly_set_) {
      LOG(ERROR) << "You should set " << flag_name << " explicitly.";
    }
    exit(0);
  }
}
}  // namespace rvl

//-----------------------------------------------------------------------------
