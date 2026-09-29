#!/bin/bash
# 2D active marker tracking viewer (MTWindow)
cd "$(dirname "$0")" || exit 1

# Playback from file
./build/tracking_viewer \
  -config=config/default.yaml \
  -input=/media/yyi/data/emc_data/0819/multiLED.raw \
  -marker_json=config/marker/multi_led_marker.json \
  -realtime \
  -show_events # visualize events

# # Live streaming from camera
# ./build/tracking_viewer \
#   -config=config/default.yaml \
#   -marker_json=config/marker/multi_led_marker.json \
#   -camera_config=config/camera_config.json