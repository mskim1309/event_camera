#!/usr/bin/env bash
# Reproduce independent R0/R1 comparisons for one or more six-marker RAW files.
#
# Run inside the Metavision/Ceres container:
#   ./scripts/run_6mark_raw_batch_verified.sh
#   ./scripts/run_6mark_raw_batch_verified.sh 6mark5 6mark6
#
# A batch only receives BATCH_COMPLETE after every child run passes its R0/R1
# parity check, its input/output hashes, and its JSON summary validation.
# It deliberately writes one row per recording; recordings are not pooled into
# a pseudo-replicated aggregate metric.
#
# Environment overrides: EMOCAP_BIN, EMOCAP_RAW_ROOT, EMOCAP_BATCH_LABEL
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
cd "${project_root}"

if (($# == 0)); then
    recordings=(6mark4 6mark5 6mark6 6mark7 6mark8)
else
    recordings=("$@")
fi

for recording in "${recordings[@]}"; do
    case "${recording}" in
        6mark4|6mark5|6mark6|6mark7|6mark8) ;;
        *)
            echo "usage: $0 [6mark4|6mark5|6mark6|6mark7|6mark8 ...]" >&2
            exit 2
            ;;
    esac
done

binary="${EMOCAP_BIN:-/tmp/emocap-metavision-build/event_mocap_unified}"
raw_root="${EMOCAP_RAW_ROOT:-/workspace/mskim/event_mocap/metavision_active_marker_2d_tracking}"
runner="${script_dir}/run_6mark_raw_pair_verified.sh"
analyzer="${script_dir}/analyze_runtime_replay.py"
marker="${project_root}/config/6marker.json"
calibration="${project_root}/config/calibration_1017.json"
default_config="${project_root}/config/default.yaml"

for required in "${binary}" "${runner}" "${analyzer}" "${marker}" "${calibration}" "${default_config}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required file: ${required}" >&2
        exit 2
    fi
done
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 2; }

for recording in "${recordings[@]}"; do
    raw="${raw_root}/${recording}.raw"
    if [[ ! -f "${raw}" ]]; then
        echo "missing RAW file: ${raw}" >&2
        exit 2
    fi
done

batch_label="${EMOCAP_BATCH_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}}"
if [[ ! "${batch_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EMOCAP_BATCH_LABEL may contain only A-Z, a-z, 0-9, dot, underscore, and hyphen" >&2
    exit 2
fi

batch_dir="${project_root}/result/raw_replay_batch_${batch_label}"
if [[ -e "${batch_dir}" ]]; then
    echo "refusing to reuse an existing batch directory: ${batch_dir}" >&2
    exit 2
fi

for recording in "${recordings[@]}"; do
    child_dir="${project_root}/result/raw_replay_${recording}_${batch_label}"
    if [[ -e "${child_dir}" ]]; then
        echo "refusing to reuse an existing child run directory: ${child_dir}" >&2
        exit 2
    fi
done

mkdir -p "${batch_dir}"
manifest="${batch_dir}/batch_manifest.txt"
{
    printf 'batch_label=%s\n' "${batch_label}"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'project_root=%s\n' "${project_root}"
    printf 'binary=%s\n' "${binary}"
    printf 'raw_root=%s\n' "${raw_root}"
    printf 'marker=%s\n' "${marker}"
    printf 'calibration=%s\n' "${calibration}"
    printf 'default_config=%s\n' "${default_config}"
    printf 'runner=%s\n' "${runner}"
    printf 'recordings=%s\n' "${recordings[*]}"
    printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'git_status_begin\n'
    git status --short || true
    printf 'git_status_end\n'
} > "${manifest}"

fail_batch() {
    local recording="$1"
    local reason="$2"
    printf 'failed_utc=%s\nrecording=%s\nreason=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${recording}" "${reason}" > "${batch_dir}/FAILED"
    echo "batch failed at ${recording}: ${reason}" >&2
    exit 1
}

for recording in "${recordings[@]}"; do
    child_dir="${project_root}/result/raw_replay_${recording}_${batch_label}"
    child_log="${batch_dir}/${recording}.log"

    printf 'Running verified R0/R1 replay: %s\n' "${recording}"
    if ! EMOCAP_BIN="${binary}" EMOCAP_RAW_ROOT="${raw_root}" EMOCAP_RUN_LABEL="${batch_label}" \
        bash "${runner}" "${recording}" 2>&1 | tee "${child_log}"; then
        fail_batch "${recording}" "child_runner_failed"
    fi

    [[ -f "${child_dir}/COMPLETE" ]] || fail_batch "${recording}" "missing_child_COMPLETE"
    [[ -f "${child_dir}/sha256sum.txt" ]] || fail_batch "${recording}" "missing_child_sha256sum"
    if ! sha256sum --check --status "${child_dir}/sha256sum.txt"; then
        fail_batch "${recording}" "child_hash_verification_failed"
    fi

    summary="${child_dir}/runtime_analysis/summary.json"
    row="${batch_dir}/${recording}.json"
    [[ -f "${summary}" ]] || fail_batch "${recording}" "missing_runtime_summary"
    if ! python3 - "${summary}" "${row}" "${recording}" "${child_dir}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, row_path, recording, child_dir = map(Path, sys.argv[1:])
recording = str(recording)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
pairing = summary.get("raw_pnp_pairing", {})
if pairing.get("identical") is not True:
    raise SystemExit("raw_pnp_pairing.identical is not true")

continuity = summary["continuity"]
counts = continuity["output_source_counts"]
dynamic = summary["dynamic_local_linear"]
row = {
    "recording": recording,
    "run_directory": str(child_dir),
    "summary_path": str(summary_path),
    "raw_pnp_pairing_identical": True,
    "callbacks": continuity["callbacks"],
    "raw_measurements": continuity["raw_measurements"],
    "output_poses": continuity["output_poses"],
    "filtered_measurements": counts.get("filtered_measurement", 0),
    "predictions": counts.get("prediction", 0),
    "no_output": counts.get("none", 0),
    "dynamic_local_linear": {
        "raw_position_rms_mm": dynamic["raw"]["position"]["rms"],
        "filtered_position_rms_mm": dynamic["filtered"]["position"]["rms"],
        "raw_rotation_rms_deg": dynamic["raw"]["rotation"]["rms"],
        "filtered_rotation_rms_deg": dynamic["filtered"]["rotation"]["rms"],
    },
}
row_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
    then
        fail_batch "${recording}" "invalid_runtime_summary"
    fi
done

python3 - "${batch_dir}" "${batch_label}" "${recordings[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

batch_dir = Path(sys.argv[1])
batch_label = sys.argv[2]
recordings = sys.argv[3:]
rows = [json.loads((batch_dir / f"{recording}.json").read_text(encoding="utf-8")) for recording in recordings]
index = {
    "schema_version": 1,
    "batch_label": batch_label,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "recordings": rows,
    "interpretation_limit": (
        "Each RAW is retained as an independent row. Do not pool callback-level "
        "metrics into a single accuracy claim; no external ground truth is present."
    ),
}
(batch_dir / "batch_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Verified multi-RAW R0/R1 batch",
    "",
    f"- batch label: `{batch_label}`",
    "- Each row is one recording. Values are not pooled across callbacks or recordings.",
    "- `dynamic_local_linear` is a high-frequency variation proxy, not absolute pose accuracy.",
    "- Prediction is counted only for output continuity; paired variation metrics use filtered measurements.",
    "",
    "| RAW | R0/R1 parity | callbacks | raw measurement | output pose | prediction | no output | position RMS raw → filtered (mm) | rotation RMS raw → filtered (deg) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    dynamic = row["dynamic_local_linear"]
    lines.append(
        f"| {row['recording']} | {str(row['raw_pnp_pairing_identical']).lower()} | "
        f"{row['callbacks']:,} | {row['raw_measurements']:,} | {row['output_poses']:,} | "
        f"{row['predictions']:,} | {row['no_output']:,} | "
        f"{dynamic['raw_position_rms_mm']:.4f} → {dynamic['filtered_position_rms_mm']:.4f} | "
        f"{dynamic['raw_rotation_rms_deg']:.4f} → {dynamic['filtered_rotation_rms_deg']:.4f} |"
    )
(batch_dir / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${manifest}"
printf 'all_child_runs_verified=true\n' >> "${manifest}"
printf 'complete\n' > "${batch_dir}/BATCH_COMPLETE"

printf 'Verified multi-RAW batch complete: %s\n' "${batch_dir}"
printf 'Report: %s\n' "${batch_dir}/batch_summary.md"
