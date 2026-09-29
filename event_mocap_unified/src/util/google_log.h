// Copyright (c) 2025, Robot Vision Lab. SNU
// All rights reserved.
//
// Authors:
//    Jongwoo Lim  
//   

#ifndef _UTIL_GOOGLE_LOG_H_
#define _UTIL_GOOGLE_LOG_H_

#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include "util/yaml.h"

namespace rvl {


inline void InitGoogleLogging(const char* argv0) {
  google::InitGoogleLogging(argv0);
  Flag::InvokeAfterInit(false);
}

} // namespace rvl


#define REFINE_VARIABLE(type, shorttype, develname, deployname, value, help, explicitly) \
  namespace fL##shorttype {                                               \
    static const type FLAGS_nono##develname = value;                                    \
    /* We always want to export defined variables, dll or no */         \
    GFLAGS_DLL_DEFINE_FLAG type FLAGS_##develname = FLAGS_nono##develname;        \
    static type FLAGS_no##develname = FLAGS_nono##develname;                      \
    static GFLAGS_NAMESPACE::FlagRegisterer o_##develname(                   \
      #develname, MAYBE_STRIPPED_HELP(help), __FILE__,                       \
      &FLAGS_##develname, &FLAGS_no##develname);                                       \
    GFLAGS_DLL_DEFINE_FLAG type FLAGS_##deployname = FLAGS_nono##develname;        \
    static GFLAGS_NAMESPACE::FlagRegisterer o_##deployname(                   \
      #deployname, MAYBE_STRIPPED_HELP(help), __FILE__,                       \
      &FLAGS_##deployname, &FLAGS_no##develname);                                        \
    char FLAGS_nonono##deployname = []{                                      \
    rvl::Flag::after_init_listener.connect([]{                                               \
    if (!rvl::IsGoogleFlagDefaultValue(#develname)) {                                    \
         fL##shorttype::FLAGS_##deployname = fL##shorttype::FLAGS_##develname;           \
    }                                                                                     \
    if (explicitly && rvl::IsGoogleFlagDefaultValue(#develname) && rvl::IsGoogleFlagDefaultValue(#deployname)) \
    { rvl::Flag::GetNotExplicitlySet().emplace_back(rvl::FormattedString("%s or %s", #develname, #deployname));  \
    }\
    });                                                                                              \
    return char{}; }();             \
  }                                                                     \
  type& FLAGS_##develname = fL##shorttype::FLAGS_##deployname

#define REFINE_bool(develname, deployname, val, txt, explicitly)                     \
  namespace fLB {                                                        \
    typedef ::fLB::CompileAssert FLAG_##deployname##_value_is_not_a_bool[\
            (sizeof(::fLB::IsBoolFlag(val)) != sizeof(double))? 1: -1];  \
  }                                                                      \
  REFINE_VARIABLE(bool, B, develname, deployname, val, txt, explicitly)

#define REFINE_int32(develname, deployname, val, txt, explicitly) \
   REFINE_VARIABLE(GFLAGS_NAMESPACE::int32, I, \
                   develname, deployname, val, txt, explicitly)

#define REFINE_uint32(develname, deployname,val, txt, explicitly) \
   REFINE_VARIABLE(GFLAGS_NAMESPACE::uint32, U, \
                   develname, deployname, val, txt, explicitly)

#define REFINE_int64(develname, deployname, val, txt, explicitly) \
   REFINE_VARIABLE(GFLAGS_NAMESPACE::int64, I64, \
                   develname, deployname, val, txt, explicitly)

#define REFINE_uint64(develname, deployname,val, txt, explicitly) \
   REFINE_VARIABLE(GFLAGS_NAMESPACE::uint64, U64, \
                   develname, deployname, val, txt, explicitly)

#define REFINE_double(develname, deployname, val, txt, explicitly) \
   REFINE_VARIABLE(double, D, develname, deployname, val, txt, explicitly)


#define REFINE_string(develname, deployname, val, txt, explicitly)                      \
  namespace fLS {                                                           \
    using ::fLS::clstring;                                                  \
    using ::fLS::StringFlagDestructor;                                      \
    static union { void* align; char s[sizeof(clstring)]; } s_##develname[2];    \
    clstring* const FLAGS_no##develname = ::fLS::                                \
                                   dont_pass0toDEFINE_string(s_##develname[0].s, \
                                                             val);          \
    static GFLAGS_NAMESPACE::FlagRegisterer o_##develname(                       \
        #develname, MAYBE_STRIPPED_HELP(txt), __FILE__,                          \
        FLAGS_no##develname, new (s_##develname[1].s) clstring(*FLAGS_no##develname));     \
    static StringFlagDestructor d_##develname(s_##develname[0].s, s_##develname[1].s);     \
    extern GFLAGS_DLL_DEFINE_FLAG clstring& FLAGS_##develname;                   \
    using fLS::FLAGS_##develname;                                                \
    clstring& FLAGS_##develname = *FLAGS_no##develname;                   \
  static union { void* align; char s[sizeof(clstring)]; } s_##deployname[2];    \
    clstring* const FLAGS_no##deployname = ::fLS::                                \
                                   dont_pass0toDEFINE_string(s_##deployname[0].s, \
                                                             val);          \
    static GFLAGS_NAMESPACE::FlagRegisterer o_##deployname(                       \
        #deployname, MAYBE_STRIPPED_HELP(txt), __FILE__,                          \
        FLAGS_no##deployname, new (s_##deployname[1].s) clstring(*FLAGS_no##deployname));     \
    static StringFlagDestructor d_##deployname(s_##deployname[0].s, s_##deployname[1].s);     \
    extern GFLAGS_DLL_DEFINE_FLAG clstring& FLAGS_##deployname;                   \
    using fLS::FLAGS_##deployname;                                                \
    clstring& FLAGS_##deployname = *FLAGS_no##deployname;                   \
    char FLAGS_nonono##deployname = []{                                      \
    rvl::Flag::after_init_listener.connect([]{                                               \
    if (!rvl::IsGoogleFlagDefaultValue(#develname)) {                                    \
         fLS::FLAGS_##deployname = fLS::FLAGS_##develname;           \
    }                                                                              \
    if (explicitly && rvl::IsGoogleFlagDefaultValue(#develname) && rvl::IsGoogleFlagDefaultValue(#deployname)) \
    { rvl::Flag::GetNotExplicitlySet().emplace_back(rvl::FormattedString("%s or %s", #develname, #deployname));                                                                                                \
    }\
    });\
    return char{}; }();             \
  }                                                                         \
  fLS::clstring& FLAGS_##develname = fLS::FLAGS_##deployname

#define RECLARE_VARIABLE(type, shorttype, develname, deployname) \
  /* We always want to import declared variables, dll or no */ \
  namespace fL##shorttype { \
    extern GFLAGS_DLL_DECLARE_FLAG type FLAGS_##deployname; }  \
  extern type& FLAGS_##develname;\
  using fL##shorttype::FLAGS_##deployname

#define RECLARE_bool(develname, deployname) \
  RECLARE_VARIABLE(bool, B, develname, deployname)

#define RECLARE_int32(develname, deployname) \
  RECLARE_VARIABLE(::GFLAGS_NAMESPACE::int32, I, develname, deployname)

#define RECLARE_uint32(develname, deployname) \
  RECLARE_VARIABLE(::GFLAGS_NAMESPACE::uint32, U, develname, deployname)

#define RECLARE_int64(develname, deployname) \
  RECLARE_VARIABLE(::GFLAGS_NAMESPACE::int64, I64, develname, deployname)

#define RECLARE_uint64(develname, deployname) \
  RECLARE_VARIABLE(::GFLAGS_NAMESPACE::uint64, U64, develname, deployname)

#define RECLARE_double(develname, deployname) \
  RECLARE_VARIABLE(double, D, develname, deployname)

#define RECLARE_string(develname, deployname) \
  /* We always want to import declared variables, dll or no */ \
  namespace fLS { \
    extern GFLAGS_DLL_DECLARE_FLAG ::fLS::clstring& FLAGS_##deployname; } \
  extern fLS::clstring& FLAGS_##develname; \
  using fLS::FLAGS_##deployname

#endif  // _UTIL_GOOGLE_LOG_H_
