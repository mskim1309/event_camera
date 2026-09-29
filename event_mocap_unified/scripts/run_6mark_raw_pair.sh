#!/usr/bin/env bash
# Reproducible R0/R1 replay for the 2026 six-marker RAW recordings.
set -euo pipefail

recording="${1:?usage: $0 <6mark4|6mark5|6mark6|6mark7|6mark8>}"
case "${recording}" in
    6mark4|6mark5|6mark6|6mark7|6mark8) ;;
    *) echo "unsupported six-marker recording: ${recording}" >&2; exit 2 ;;
esac

binary="${EMOCAP_BIN:-/tmp/emocap-metavision-build/event_mocap_unified}"
raw_root="${EMOCAP_RAW_ROOT:-/workspace/mskim/event_mocap/metavision_active_marker_2d_tracking}"
raw="${raw_root}/${recording}.raw"
marker="config/6marker.json"
calibration="config/calibration_1017.json"
default_config="config/default.yaml"
result_dir="result/raw_replay_${recording}"

for required in "${binary}" "${raw}" "${marker}" "${calibration}" "${default_config}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required file: ${required}" >&2
        exit 2
    fi
done

mkdir -p "${result_dir}"
manifest="${result_dir}/run_manifest.txt"
{
    printf 'recording=%s\n' "${recording}"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'working_directory=%s\n' "$(pwd)"
    printf 'binary=%s\n' "${binary}"
    printf 'raw=%s\n' "${raw}"
    printf 'marker=%s\n' "${marker}"
    printf 'calibration=%s\n' "${calibration}"
    printf 'file_mode_note=camera_config_and_sidecar_bias_are_not_loaded_by_current_EventFileSource\n'
    git status --short || true
} > "${manifest}"

common=(
    "--config=${default_config}"
    "--mode=file"
    "--input=${raw}"
    "--marker_json=${marker}"
    "--calib_json=${calibration}"
    "--pnp_candidate_mode=auto"
    "--logtostderr"
)

"${binary}" "${common[@]}" \
    "--output=${result_dir}/r0_raw_pose.csv" \
    "--pnp_diagnostics_frame_output=${result_dir}/r0_frame.csv" \
    "--pnp_diagnostics_correspondence_output=${result_dir}/r0_correspondence.csv" \
    "--temporal_filter=false" \
    2>&1 | tee "${result_dir}/r0.log"

"${binary}" "${common[@]}" \
    "--output=${result_dir}/r1_filtered_pose.csv" \
    "--temporal_filter=true" \
    "--temporal_filter_diagnostics_output=${result_dir}/r1_temporal.csv" \
    2>&1 | tee "${result_dir}/r1.log"

hash_inputs=("${binary}" "${raw}" "${marker}" "${calibration}" "${default_config}" "$0")
if [[ -f "${raw_root}/${recording}.bias" ]]; then
    hash_inputs+=("${raw_root}/${recording}.bias")
fi
sha256sum "${hash_inputs[@]}" \
    "${result_dir}/r0_raw_pose.csv" \
    "${result_dir}/r0_frame.csv" \
    "${result_dir}/r0_correspondence.csv" \
    "${result_dir}/r1_filtered_pose.csv" \
    "${result_dir}/r1_temporal.csv" \
    > "${result_dir}/sha256sum.txt"

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${manifest}"
printf 'R0/R1 replay complete: %s\n' "${result_dir}"
