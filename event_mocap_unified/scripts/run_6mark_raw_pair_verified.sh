#!/usr/bin/env bash
# Reproduce a six-marker RAW R0/R1 comparison without mixing runs.
#
# Run inside the Metavision/Ceres container from any directory:
#   ./scripts/run_6mark_raw_pair_verified.sh 6mark5
#
# Environment overrides:
#   EMOCAP_BIN, EMOCAP_RAW_ROOT, EMOCAP_RUN_LABEL
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
cd "${project_root}"

recording="${1:-}"
case "${recording}" in
    6mark4|6mark5|6mark6|6mark7|6mark8) ;;
    *)
        echo "usage: $0 <6mark4|6mark5|6mark6|6mark7|6mark8>" >&2
        exit 2
        ;;
esac

binary="${EMOCAP_BIN:-/tmp/emocap-metavision-build/event_mocap_unified}"
raw_root="${EMOCAP_RAW_ROOT:-/workspace/mskim/event_mocap/metavision_active_marker_2d_tracking}"
raw="${raw_root}/${recording}.raw"
marker="${project_root}/config/6marker.json"
calibration="${project_root}/config/calibration_1017.json"
default_config="${project_root}/config/default.yaml"
analyzer="${script_dir}/analyze_runtime_replay.py"

for required in "${binary}" "${raw}" "${marker}" "${calibration}" "${default_config}" "${analyzer}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required file: ${required}" >&2
        exit 2
    fi
done
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }

run_label="${EMOCAP_RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}}"
if [[ ! "${run_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EMOCAP_RUN_LABEL may contain only A-Z, a-z, 0-9, dot, underscore, and hyphen" >&2
    exit 2
fi

result_dir="${project_root}/result/raw_replay_${recording}_${run_label}"
if [[ -e "${result_dir}" ]]; then
    echo "refusing to reuse an existing run directory: ${result_dir}" >&2
    exit 2
fi
mkdir -p "${result_dir}"

manifest="${result_dir}/run_manifest.txt"
{
    printf 'recording=%s\n' "${recording}"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'project_root=%s\n' "${project_root}"
    printf 'result_dir=%s\n' "${result_dir}"
    printf 'binary=%s\n' "${binary}"
    printf 'raw=%s\n' "${raw}"
    printf 'marker=%s\n' "${marker}"
    printf 'calibration=%s\n' "${calibration}"
    printf 'default_config=%s\n' "${default_config}"
    printf 'analyzer=%s\n' "${analyzer}"
    printf 'file_mode_note=EventFileSource uses Camera::from_file; recording sidecar bias is not replay-applied by this command\n'
    printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'git_status_begin\n'
    git status --short || true
    printf 'git_status_end\n'
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

# This checks that R0's poses and R1's raw-measurement fields are identical
# before it emits any comparative smoothness/repeatability metric.
python3 "${analyzer}" \
    --raw-pose-csv "${result_dir}/r0_raw_pose.csv" \
    --temporal-csv "${result_dir}/r1_temporal.csv" \
    --output-dir "${result_dir}/runtime_analysis" \
    > "${result_dir}/runtime_analysis_stdout.json"

hash_inputs=("${binary}" "${raw}" "${marker}" "${calibration}" "${default_config}" "${analyzer}" "${script_dir}/run_6mark_raw_pair_verified.sh")
if [[ -f "${raw_root}/${recording}.bias" ]]; then
    hash_inputs+=("${raw_root}/${recording}.bias")
fi
sha256sum "${hash_inputs[@]}" \
    "${result_dir}/r0_raw_pose.csv" \
    "${result_dir}/r0_frame.csv" \
    "${result_dir}/r0_correspondence.csv" \
    "${result_dir}/r1_filtered_pose.csv" \
    "${result_dir}/r1_temporal.csv" \
    "${result_dir}/runtime_analysis/summary.json" \
    "${result_dir}/runtime_analysis/summary.md" \
    "${result_dir}/runtime_analysis/runtime_pair_comparison.png" \
    > "${result_dir}/sha256sum.txt"

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${manifest}"
printf 'r0_r1_parity=enforced_by_runtime_analysis\n' >> "${manifest}"
printf 'complete\n' > "${result_dir}/COMPLETE"

printf 'Verified R0/R1 replay complete: %s\n' "${result_dir}"
printf 'Report: %s\n' "${result_dir}/runtime_analysis/summary.md"
