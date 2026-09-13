"""Simulate two independent S10 annotators without creating human ground truth."""

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

from vs_epl_krls.news_annotation import merge_annotation_slots, validate_annotations
from vs_epl_krls.news_annotation_simulation import simulate_annotation_slot


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


def _disagreements(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    left = first.set_index("item_id")
    right = second.set_index("item_id")
    if set(left.index) != set(right.index):
        raise ValueError("simulated slot files do not contain the same items")
    rows: list[dict[str, object]] = []
    label_fields = (
        "relevance_label",
        "direction_label",
        "intensity_label",
        "horizon_label",
    )
    for item_id in sorted(left.index):
        a = left.loc[item_id]
        b = right.loc[item_id]
        differing = [field for field in label_fields if str(a[field]) != str(b[field])]
        if differing:
            rows.append(
                {
                    "item_id": item_id,
                    "record_id": a["record_id"],
                    "title": a["title"],
                    "differing_fields": ",".join(differing),
                    **{f"a_{field}": a[field] for field in label_fields},
                    **{f"b_{field}": b[field] for field in label_fields},
                    "requires_human_adjudication": True,
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    source_manifest = json.loads(args.annotation_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("training_allowed") is not False:
        raise RuntimeError("source annotation manifest must keep training blocked")
    expected_files = source_manifest.get("annotator_files", {})
    if _sha256(args.slot_1) != expected_files.get("1", {}).get("sha256"):
        raise RuntimeError("slot 1 does not match the immutable blank queue")
    if _sha256(args.slot_2) != expected_files.get("2", {}).get("sha256"):
        raise RuntimeError("slot 2 does not match the immutable blank queue")
    first_blank = pd.read_csv(args.slot_1, dtype=str, keep_default_na=False)
    second_blank = pd.read_csv(args.slot_2, dtype=str, keep_default_na=False)
    first = simulate_annotation_slot(
        first_blank, persona="supply_cost", annotated_at_utc=args.timestamp
    )
    second = simulate_annotation_slot(
        second_blank, persona=args.persona_2, annotated_at_utc=args.timestamp
    )
    merged = merge_annotation_slots([first, second])
    agreement = validate_annotations(merged, require_complete=True)
    disagreements = _disagreements(first, second)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_path = args.output_dir / "simulated_supply_cost.csv"
    second_path = args.output_dir / "simulated_procurement_risk.csv"
    merged_path = args.output_dir / "simulated_merged.csv"
    disagreements_path = args.output_dir / "simulated_disagreements.csv"
    _atomic_csv(first, first_path)
    _atomic_csv(second, second_path)
    _atomic_csv(merged, merged_path)
    disagreements.to_csv(disagreements_path, index=False, encoding="utf-8-sig")

    relevance_kappa = float(agreement["relevance_cohen_kappa"])
    direction_kappa = float(agreement["direction_cohen_kappa"])
    agreement_gate = relevance_kappa >= 0.60 and direction_kappa >= 0.60
    independent_policy_gate = not disagreements.empty
    minimum_directional_items = max(3, round(0.01 * first["item_id"].nunique()))
    directional_counts = {
        "supply_cost": first["direction_label"].value_counts().to_dict(),
        args.persona_2: second["direction_label"].value_counts().to_dict(),
    }
    directional_coverage_gate = all(
        counts.get("up", 0) >= minimum_directional_items
        and counts.get("down", 0) >= minimum_directional_items
        for counts in directional_counts.values()
    )
    intensity_coverage_gate = all(
        int((pd.to_numeric(frame["intensity_label"]) >= 2).sum())
        >= minimum_directional_items
        for frame in (first, second)
    )
    simulation_gate = (
        agreement_gate
        and independent_policy_gate
        and directional_coverage_gate
        and intensity_coverage_gate
    )
    label_columns = (
        "relevance_label",
        "direction_label",
        "intensity_label",
        "horizon_label",
    )
    distributions = {
        "supply_cost": {
            field: first[field].value_counts().sort_index().to_dict()
            for field in label_columns
        },
        args.persona_2: {
            field: second[field].value_counts().sort_index().to_dict()
            for field in label_columns
        },
    }
    payload: dict[str, object] = {
        "schema_version": "s10-news-annotation-simulation.v1",
        "simulation": True,
        "human_annotations": False,
        "production_training_allowed": False,
        "model_promotion_allowed": False,
        "purpose": "workflow validation and weak-label research only",
        "source_annotation_manifest_sha256": _sha256(args.annotation_manifest),
        "news_dataset_sha256": source_manifest["news_dataset_sha256"],
        "batch_id": source_manifest["batch_id"],
        "timestamp": args.timestamp,
        "personas": {
            "slot_1": "simulated_supply_cost_v1",
            "slot_2": f"simulated_{args.persona_2}_v1",
        },
        "item_count": int(first["item_id"].nunique()),
        "agreement": agreement,
        "agreement_gate_kappa_060": agreement_gate,
        "independence_gate_nonidentical_labels": independent_policy_gate,
        "directional_coverage_gate": directional_coverage_gate,
        "intensity_coverage_gate": intensity_coverage_gate,
        "minimum_items_per_direction_and_high_intensity": minimum_directional_items,
        "simulation_gate_passed": simulation_gate,
        "disagreement_items": int(disagreements["item_id"].nunique())
        if not disagreements.empty
        else 0,
        "disagreement_rate": float(
            disagreements["item_id"].nunique() / first["item_id"].nunique()
        )
        if not disagreements.empty
        else 0.0,
        "label_distributions": distributions,
        "outputs": {
            "supply_cost": {"path": str(first_path), "sha256": _sha256(first_path)},
            args.persona_2: {
                "path": str(second_path),
                "sha256": _sha256(second_path),
            },
            "merged": {"path": str(merged_path), "sha256": _sha256(merged_path)},
            "disagreements": {
                "path": str(disagreements_path),
                "sha256": _sha256(disagreements_path),
            },
        },
        "next_required_step": "human review of disagreements and real double annotation",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Simulação de dois anotadores — Diesel S10",
        "",
        "Esta execução simula duas políticas determinísticas. Não são pessoas, não são ground truth e os rótulos não autorizam treino de produção ou promoção de modelo.",
        "",
        f"- Itens: {payload['item_count']}",
        f"- Kappa de relevância: {relevance_kappa:.4f}",
        f"- Kappa de direção: {direction_kappa:.4f}",
        f"- Kappa de intensidade: {float(agreement['intensity_cohen_kappa']):.4f}",
        f"- Kappa de horizonte: {float(agreement['horizon_cohen_kappa']):.4f}",
        f"- Itens com alguma divergência: {payload['disagreement_items']}",
        f"- Gate simulado de concordância: {agreement_gate}",
        f"- Gate de independência (saídas não idênticas): {independent_policy_gate}",
        f"- Gate de cobertura direcional: {directional_coverage_gate}",
        f"- Gate de cobertura de intensidade: {intensity_coverage_gate}",
        f"- Gate combinado da simulação: {simulation_gate}",
        "",
        "As divergências foram preservadas para adjudicação humana. O lote humano original permanece intacto e em branco.",
    ]
    (args.report_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-manifest",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_news_annotation"
        / "manifest.json",
    )
    parser.add_argument(
        "--slot-1",
        type=Path,
        default=ROOT
        / "data"
        / "annotations"
        / "s10_news_pressure_v1.annotator_1.csv",
    )
    parser.add_argument(
        "--slot-2",
        type=Path,
        default=ROOT
        / "data"
        / "annotations"
        / "s10_news_pressure_v1.annotator_2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "annotations" / "simulated",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_news_annotation_simulation",
    )
    parser.add_argument("--timestamp", default="2026-08-23T12:00:00Z")
    parser.add_argument(
        "--persona-2",
        choices=("procurement_risk", "procurement_risk_calibrated"),
        default="procurement_risk",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
