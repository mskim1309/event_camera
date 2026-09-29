# mskim event-camera workspace

This repository keeps the source code and configuration used for the event-camera
active-marker pose-estimation work.

- `event_mocap/`: earlier standalone tracking, PnP, visualization, and camera-pose experiments.
- `event_mocap_unified/`: current unified active-marker tracker, geometry-aware PnP,
  causal temporal filter, and reproducible RAW replay/analysis scripts.
- `기시설1_2021-19515_김민성_3차제출.docx`: original thesis submission document.

The repository intentionally excludes sensor `.raw` recordings, generated RAW
indexes, build outputs, generated pose CSV/plots, and downloaded SDKs. Those
artifacts are large and are reproducible from the tracked source/configuration
or must be handled as experiment data outside Git.
