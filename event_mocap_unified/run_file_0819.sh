#!/bin/bash
# Run file mode with 0819 multiLED dataset
cd "$(dirname "$0")" || exit 1

./build/event_mocap_unified \
  -config=config/default.yaml \
  -mode=file \
  -input=/media/yyi/data/emc_data/0819/multiLED.raw \
  -marker_json=config/marker/multi_led_marker.json \
  -calib_json=config/calibration_1017.json \
  -output=result/pose_0819_multiLED.csv \
  -visualize
