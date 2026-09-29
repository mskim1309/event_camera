cmake_minimum_required(VERSION 3.14)
project(event_mocap_unified)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# --- Required dependencies ---
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)

# --- Optional: Metavision SDK (live/file modes) ---
find_package(MetavisionSDK COMPONENTS base cv cv3d driver QUIET)
if(MetavisionSDK_FOUND)
  message(STATUS "MetavisionSDK found — live/file modes enabled")
else()
  message(STATUS "MetavisionSDK NOT found — simulation mode only")
endif()

# --- Optional: Rerun (visualization) ---
find_package(rerun_sdk QUIET)
if(rerun_sdk_FOUND)
  message(STATUS "Rerun SDK found — visualization enabled")
else()
  message(STATUS "Rerun SDK NOT found — visualization disabled (no-op)")
endif()

# --- Library sources ---
set(LIB_SOURCES
  src/config_loader.cpp
  src/pose_estimator.cpp
  src/marker_simulator.cpp
  src/visualizer.cpp
  src/data_source.cpp
  src/gflags.cpp
  src/util/yaml.cc
  src/util/file_io_util.cc
)

if(MetavisionSDK_FOUND)
  list(APPEND LIB_SOURCES src/active_marker_tracker.cpp)
endif()

add_library(event_mocap_lib ${LIB_SOURCES})

target_include_directories(event_mocap_lib PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(event_mocap_lib PUBLIC
  Eigen3::Eigen
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  glog::glog
  gflags
  yaml-cpp
)

if(MetavisionSDK_FOUND)
  target_link_libraries(event_mocap_lib PUBLIC
    MetavisionSDK::base
    MetavisionSDK::cv
    MetavisionSDK::cv3d
    MetavisionSDK::driver
  )
  target_compile_definitions(event_mocap_lib PUBLIC HAS_METAVISION)
endif()

if(rerun_sdk_FOUND)
  target_link_libraries(event_mocap_lib PUBLIC rerun_sdk)
  target_compile_definitions(event_mocap_lib PUBLIC RERUN_FOUND)
endif()

# --- Executables ---
add_executable(event_mocap_unified app/main.cpp)
target_link_libraries(event_mocap_unified event_mocap_lib)

# 2D tracking viewer (requires MetavisionSDK + UI)
if(MetavisionSDK_FOUND)
  find_package(MetavisionSDK COMPONENTS ui QUIET)
  if(MetavisionSDK_ui_FOUND OR TARGET MetavisionSDK::ui)
    add_executable(tracking_viewer app/tracking_viewer.cpp)
    target_link_libraries(tracking_viewer event_mocap_lib MetavisionSDK::ui)
    message(STATUS "tracking_viewer enabled")
  else()
    message(STATUS "MetavisionSDK::ui not found — tracking_viewer disabled")
  endif()
endif()
