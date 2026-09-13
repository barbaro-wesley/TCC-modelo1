"""Create or validate a traceable double-annotation batch for S10 news pressure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.news_annotation import (
    ANNOTATION_COLUMNS,
    build_annotation_batch,
    merge_annotation_slots,
    validate_annotations,
)
from vs_epl_krls.news_pressure import load_news_corpus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    safe = frame.copy()
    for column in ("title", "summary", "canonical_url", "evidence_text", "rationale"):
        safe[column] = safe[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
            else value
        )
    safe.to_csv(temporary, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.merge_completed:
        frames = [
            pd.read_csv(path, dtype=str, keep_default_na=False)
            for path in args.merge_completed
        ]
        combined = merge_annotation_slots(frames)
        _atomic_csv(combined, args.completed_output)
        result = validate_annotations(combined, require_complete=True)
        return {
            "mode": "merge_and_validate",
            "completed_path": str(args.completed_output),
            "completed_sha256": _sha256(args.completed_output),
            **result,
        }
    if args.validate_completed is not None:
        frame = pd.read_csv(args.validate_completed, dtype=str, keep_default_na=False)
        result = validate_annotations(frame, require_complete=True)
        return {
            "mode": "validate",
            "annotations_sha256": _sha256(args.validate_completed),
            **result,
        }

    pressure_manifest = json.loads(args.pressure_manifest.read_text(encoding="utf-8"))
    if pressure_manifest.get("holdout_evaluated") is not False:
        raise RuntimeError("pressure experiment does not prove holdout isolation")
    corpus = load_news_corpus(args.news_records)
    if corpus.dataset_sha256 != pressure_manifest.get("news_dataset_sha256"):
        raise RuntimeError("annotation corpus differs from the pressure experiment lineage")
    batch = build_annotation_batch(
        corpus,
        item_count=args.item_count,
        relevant_fraction=args.relevant_fraction,
        annotators_per_item=args.annotators_per_item,
        random_state=args.random_state,
        batch_name=args.batch_name,
    )
    validate_annotations(batch.frame, require_complete=False)
    _atomic_csv(batch.frame, args.queue_output)
    queue_hash = _sha256(args.queue_output)
    slot_files: dict[str, dict[str, str]] = {}
    for slot in range(1, batch.rows_per_item + 1):
        slot_path = args.queue_output.with_name(
            f"{args.queue_output.stem}.annotator_{slot}{args.queue_output.suffix}"
        )
        _atomic_csv(
            batch.frame[batch.frame["annotation_slot"] == slot].reset_index(drop=True),
            slot_path,
        )
        slot_files[str(slot)] = {"path": str(slot_path), "sha256": _sha256(slot_path)}
    payload: dict[str, object] = {
        "schema_version": "s10-news-annotation.v1",
        "batch_id": batch.batch_id,
        "news_dataset_sha256": batch.news_dataset_sha256,
        "pressure_manifest_sha256": _sha256(args.pressure_manifest),
        "queue_sha256": queue_hash,
        "queue_path": str(args.queue_output),
        "annotator_files": slot_files,
        "item_count": batch.item_count,
        "annotation_rows": len(batch.frame),
        "annotators_per_item": batch.rows_per_item,
        "random_state": args.random_state,
        "relevant_fraction_requested": args.relevant_fraction,
        "selection_pool_counts": batch.selection_pool_counts,
        "source_counts": batch.source_counts,
        "year_min": int(pd.to_datetime(batch.frame["published_date"]).dt.year.min()),
        "year_max": int(pd.to_datetime(batch.frame["published_date"]).dt.year.max()),
        "allowed_labels": {
            "relevance_label": [0, 1, 2, 3],
            "direction_label": ["down", "neutral", "up", "uncertain"],
            "intensity_label": [0, 1, 2, 3],
            "horizon_label": ["1w", "2w", "4w", "long", "unknown"],
        },
        "columns": list(ANNOTATION_COLUMNS),
        "status": "awaiting_two_independent_human_annotations",
        "training_allowed": False,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest_output.with_name(args.manifest_output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.manifest_output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-records", type=Path)
    parser.add_argument(
        "--pressure-manifest",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_news_pressure"
        / "manifest.json",
    )
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=ROOT / "data" / "annotations" / "s10_news_pressure_v1.csv",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_news_annotation"
        / "manifest.json",
    )
    parser.add_argument("--item-count", type=int, default=300)
    parser.add_argument("--relevant-fraction", type=float, default=2 / 3)
    parser.add_argument("--annotators-per-item", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=20260823)
    parser.add_argument("--batch-name", default="s10-pressure-v1")
    parser.add_argument("--validate-completed", type=Path)
    parser.add_argument("--merge-completed", type=Path, nargs="+")
    parser.add_argument(
        "--completed-output",
        type=Path,
        default=ROOT / "data" / "annotations" / "s10_news_pressure_v1.completed.csv",
    )
    args = parser.parse_args()
    if (
        args.validate_completed is None
        and not args.merge_completed
        and args.news_records is None
    ):
        parser.error("--news-records is required when creating a batch")
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
