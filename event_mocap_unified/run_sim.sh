#!/bin/bash
# Run simulation mode
cd "$(dirname "$0")" || exit 1

./build/event_mocap_unified \
  -config=config/default.yaml \
  -mode=sim \
  -output=result/pose_sim.csv \
  -visualize
