"""Run Stage-5 quantization-exclusion ablations for YOLO INT8 model.22.

Stage 5 starts from the Stage-3 accuracy winner: model.22 cv2 + cv3 + dfl are
excluded from ModelOpt quantization while model.22/other remains quantized.
Each candidate returns exactly one cv2 scale branch (cv2.0, cv2.1, or cv2.2)
to Q/DQ quantization while keeping the other two cv2 branches plus cv3 and dfl
excluded. Accuracy is measured only; latency is benchmarked later only for an
accuracy-surviving candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.yolo_int8.generate_calibration_data import sha256
from examples.yolo_int8.inspect_yolo_node_groups import exact_nodes_for_blocks, inspect_onnx
from examples.yolo_int8.run_calibration_matrix import (
    BUILDER_SETTINGS,
    build_accuracy_command,
    build_engine_command,
    expected_builder_metadata,
    metadata_matches,
    query_modelopt_version,
    validate_python_interpreter,
)
from examples.yolo_int8.run_selective_fp16_sensitivity import (
    METRICS,
    SMOKE_LIMIT,
    build_quantize_command,
    exclusion_patterns,
)
from examples.yolo_int8.run_selective_fp16_stage3 import partition_block22

CV2_BRANCH_PREFIXES = {
    "cv2_0": "/model.22/cv2.0/",
    "cv2_1": "/model.22/cv2.1/",
    "cv2_2": "/model.22/cv2.2/",
}


def partition_cv2_branches(cv2_names: Sequence[str]) -> dict[str, list[str]]:
    """Partition exact model.22/cv2 node names into the three detection scales."""
    result = {name: [] for name in CV2_BRANCH_PREFIXES}
    unmatched: list[str] = []
    for node in cv2_names:
        matches = [name for name, prefix in CV2_BRANCH_PREFIXES.items() if node.startswith(prefix)]
        if len(matches) != 1:
            unmatched.append(node)
            continue
        result[matches[0]].append(node)
    if unmatched:
        raise ValueError(f"cv2 nodes did not map to exactly one scale branch: {unmatched}")
    if any(not names for names in result.values()):
        raise ValueError("every Stage-5 cv2 branch must contain at least one exact node")
    if set().union(*(set(names) for names in result.values())) != set(cv2_names):
        raise ValueError("Stage-5 cv2 branch partition is not exact")
    return {name: sorted(names) for name, names in result.items()}


def stage5_variants(
    block22_groups: dict[str, list[str]], cv2_branches: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Return exclusion sets after one cv2 branch is returned to Q/DQ quantization."""
    required = {"cv2", "cv3", "dfl", "other"}
    if set(block22_groups) != required:
        raise ValueError(f"Stage 5 requires groups {sorted(required)}; found {sorted(block22_groups)}")
    parent_excluded = set(block22_groups["cv2"] + block22_groups["cv3"] + block22_groups["dfl"])
    variants: dict[str, list[str]] = {}
    for branch_name, branch_nodes in cv2_branches.items():
        excluded = sorted(parent_excluded - set(branch_nodes))
        if set(branch_nodes) & set(excluded):
            raise ValueError(f"Stage-5 branch {branch_name} overlaps its exclusion set")
        if set(excluded) | set(branch_nodes) != parent_excluded:
            raise ValueError(f"Stage-5 branch {branch_name} does not reconstruct the Stage-3 parent exclusion set")
        variants[f"quantize_{branch_name}"] = excluded
    return variants


def run_command(command: Sequence[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")
    return time.perf_counter() - started


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def artifact_identity(
    source_sha: str,
    modelopt_version: str,
    calibration_metadata: dict[str, Any],
    exact_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "source_fp32_onnx_sha256": source_sha,
        "modelopt_version": modelopt_version,
        "calibration_method": "entropy",
        "calibration_metadata": calibration_metadata,
        "calibration_count": calibration_metadata.get("count", calibration_metadata.get("calibration_count")),
        "calibration_seed": calibration_metadata.get("seed"),
        "excluded_exact_node_names": list(exact_names),
        "builder_settings": BUILDER_SETTINGS,
    }


def evaluation_identity(artifact: dict[str, Any], scope: str) -> dict[str, Any]:
    return {
        **artifact,
        "scope": scope,
        "evaluation_settings": {
            "conf_threshold": 0.001,
            "iou_threshold": 0.7,
            "limit": None if scope == "full" else SMOKE_LIMIT,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--calibration-data-dir", type=Path, required=True)
    parser.add_argument("--eval-images-dir", type=Path, required=True)
    parser.add_argument("--eval-annotation-path", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--modelopt-python", type=Path, required=True)
    parser.add_argument("--int8-baseline-accuracy-json", type=Path, required=True)
    parser.add_argument("--stage3-parent-accuracy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    validate_python_interpreter(args.runtime_python, "runtime")
    validate_python_interpreter(args.modelopt_python, "ModelOpt")
    for path, description, directory in (
        (args.onnx_path, "source ONNX", False),
        (args.calibration_data_dir, "calibration data", True),
        (args.eval_images_dir, "evaluation images", True),
        (args.eval_annotation_path, "evaluation annotations", False),
        (args.int8_baseline_accuracy_json, "INT8 baseline accuracy", False),
        (args.stage3_parent_accuracy_json, "Stage-3 parent accuracy", False),
    ):
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(f"{description} not found: {path}")

    calibration_metadata = json.loads((args.calibration_data_dir / "metadata.json").read_text(encoding="utf-8"))
    calibration_count = calibration_metadata.get("count", calibration_metadata.get("calibration_count"))
    calibration_seed = calibration_metadata.get("seed")
    if calibration_count != 128 or calibration_seed != 0:
        raise ValueError(
            "Stage 5 requires entropy_128 calibration metadata with count=128 and seed=0; "
            f"found count={calibration_count!r}, seed={calibration_seed!r}"
        )

    graph_report = inspect_onnx(args.onnx_path)
    block22_names = exact_nodes_for_blocks(graph_report, [22])
    if not block22_names:
        raise ValueError("model.22 resolved to zero named ONNX nodes")
    groups = partition_block22(block22_names)
    cv2_branches = partition_cv2_branches(groups["cv2"])
    variants = stage5_variants(groups, cv2_branches)

    source_sha = sha256(args.onnx_path)
    modelopt_version = query_modelopt_version(args.modelopt_python)
    artifact_root = args.output_dir / "artifacts"
    scope_root = args.output_dir / "results" / args.scope
    if args.force:
        if scope_root.exists():
            shutil.rmtree(scope_root)
        for label in variants:
            candidate = artifact_root / label
            if candidate.exists():
                shutil.rmtree(candidate)
    scope_root.mkdir(parents=True, exist_ok=True)

    int8_baseline = json.loads(args.int8_baseline_accuracy_json.read_text(encoding="utf-8"))
    stage3_parent = json.loads(args.stage3_parent_accuracy_json.read_text(encoding="utf-8"))
    baseline_ap = int8_baseline.get("AP50:95")
    parent_ap = stage3_parent.get("AP50:95")

    manifest: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage 5 model.22 cv2 branch quantization-exclusion ablation",
        "scope": args.scope,
        "source_fp32_onnx_sha256": source_sha,
        "modelopt_version": modelopt_version,
        "calibration_method": "entropy",
        "calibration_metadata": calibration_metadata,
        "builder_settings": BUILDER_SETTINGS,
        "evaluation_settings": {
            "conf_threshold": 0.001,
            "iou_threshold": 0.7,
            "limit": None if args.scope == "full" else SMOKE_LIMIT,
        },
        "performance_benchmarking": False,
        "parent_exclusion_groups": ["cv2", "cv3", "dfl"],
        "always_quantized_group": "other",
        "cv2_branches": [
            {"name": name, "exact_node_count": len(nodes), "exact_node_names": nodes}
            for name, nodes in cv2_branches.items()
        ],
        "variants": [],
    }
    rows: list[dict[str, Any]] = []

    for label, names in variants.items():
        returned_branch = label.removeprefix("quantize_")
        returned_nodes = cv2_branches[returned_branch]
        artifact_dir = artifact_root / label
        result_dir = scope_root / label
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        patterns = exclusion_patterns(names)
        artifact_meta = artifact_identity(source_sha, modelopt_version, calibration_metadata, names)
        eval_meta = evaluation_identity(artifact_meta, args.scope)
        row: dict[str, Any] = {
            "label": label,
            "status": "success",
            "failure_stage": None,
            "scope": args.scope,
            "smoke_result": args.scope == "smoke",
            "returned_to_quantization_branch": returned_branch,
            "returned_to_quantization_exact_node_count": len(returned_nodes),
            "returned_to_quantization_exact_node_names": returned_nodes,
            "always_quantized_group": "other",
            "excluded_exact_node_count": len(names),
            "excluded_exact_node_names": names,
        }
        try:
            onnx_out = artifact_dir / "yolov8n_int8_qdq.onnx"
            qmeta = Path(str(onnx_out) + ".conversion.json")
            engine = artifact_dir / "yolov8n_int8.engine"
            logs = result_dir / "logs"
            quant_expected = {
                "source_sha256": source_sha,
                "modelopt_version": modelopt_version,
                "calibration_method": "entropy",
                "calibration_metadata": calibration_metadata,
                "calibration_count": calibration_count,
                "calibration_seed": calibration_seed,
                "calibration_image_ids": calibration_metadata.get("image_ids", calibration_metadata.get("calibration_image_ids")),
                "nodes_to_exclude_patterns": patterns,
                "resolved_excluded_node_names": names,
            }
            row["failure_stage"] = "quantization"
            if not args.resume or not onnx_out.is_file() or not metadata_matches(qmeta, quant_expected):
                row["quantization_duration_seconds"] = run_command(
                    build_quantize_command(args.onnx_path, onnx_out, args.calibration_data_dir, args.modelopt_python, patterns),
                    logs / "quantize.log",
                )
            row["failure_stage"] = "engine_build"
            engine_meta_path = Path(str(engine) + ".json")
            if not args.resume or not engine.is_file() or not metadata_matches(engine_meta_path, expected_builder_metadata(sha256(onnx_out))):
                row["engine_build_duration_seconds"] = run_command(
                    build_engine_command(onnx_out, engine, args.runtime_python), logs / "build_engine.log"
                )
            accuracy_path = result_dir / "accuracy.json"
            identity_path = result_dir / "evaluation_metadata.json"
            row["failure_stage"] = "accuracy_evaluation"
            accuracy_current = args.resume and accuracy_path.is_file() and metadata_matches(identity_path, eval_meta)
            if not accuracy_current:
                run_command(
                    build_accuracy_command(
                        engine,
                        args.eval_images_dir,
                        args.eval_annotation_path,
                        result_dir / "predictions.json",
                        accuracy_path,
                        None if args.scope == "full" else SMOKE_LIMIT,
                        args.runtime_python,
                    ),
                    logs / "accuracy.log",
                )
            accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
            row.update({metric: accuracy.get(metric) for metric in METRICS})
            row["ONNX SHA-256"] = sha256(onnx_out)
            row["engine SHA-256"] = sha256(engine)
            row["delta_AP50:95_vs_int8_baseline"] = (
                None if baseline_ap is None or row.get("AP50:95") is None else row["AP50:95"] - baseline_ap
            )
            row["delta_AP50:95_vs_stage3_parent"] = (
                None if parent_ap is None or row.get("AP50:95") is None else row["AP50:95"] - parent_ap
            )
            write_json(identity_path, eval_meta)
            row["failure_stage"] = None
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc)})
        rows.append(row)
        manifest["variants"] = rows
        write_json(scope_root / "sensitivity_manifest.json", manifest)

    ordered = sorted(rows, key=lambda row: (row.get("AP50:95") is None, -(row.get("AP50:95") or 0), row["label"]))
    best = next((row["label"] for row in ordered if row.get("AP50:95") is not None), None)
    summary = {
        "stage": "Stage 5 model.22 cv2 branch quantization-exclusion ablation",
        "scope": args.scope,
        "smoke_warning": "Correctness-only subset; not final accuracy." if args.scope == "smoke" else None,
        "int8_baseline": {metric: int8_baseline.get(metric) for metric in METRICS},
        "stage3_parent": {metric: stage3_parent.get(metric) for metric in METRICS},
        "best_variant": best,
        "winner_note": "Latency remains deferred until full-COCO accuracy identifies a candidate that retains the Stage-3 parent recovery.",
        "variants": ordered,
    }
    write_json(scope_root / "sensitivity_summary.json", summary)
    fields = list(dict.fromkeys(key for row in ordered for key in row)) if ordered else ["label"]
    with (scope_root / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["variants"] = rows
    write_json(scope_root / "sensitivity_manifest.json", manifest)
    return summary


if __name__ == "__main__":
    main()
