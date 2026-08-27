#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def safe_filename(value: str) -> str:
    cleaned = str(value).replace("/", "__").replace("\\", "__").replace(" ", "_")
    return cleaned + ".json"


def normalize_axis_data(record: dict) -> dict:
    frequencies = {}
    for freq, freq_payload in record.get("frequencies", {}).items():
        cycles = freq_payload.get("cycles", []) if isinstance(freq_payload, dict) else []
        frequencies[freq] = {
            "cycle_count": len(cycles),
            "cycles": cycles,
        }
    return {
        "axis": record.get("axis"),
        "frequencies": frequencies,
    }


def build_dataset(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_counts = {}
    total_visits = 0

    for group_dir in output_dir.iterdir():
        if group_dir.is_dir():
            for child in group_dir.iterdir():
                if child.is_file():
                    child.unlink()
            group_dir.rmdir()

    grouped: dict[tuple[str, str, str], dict] = {}

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {input_path} at line {line_number}: {exc}") from exc

            label = record.get("pathology_class", "unknown")
            patient_name = record.get("patient_name")
            visit_date = record.get("visit_date")
            axis = record.get("axis")

            if not patient_name or not visit_date or not axis:
                continue

            key = (label, patient_name, visit_date)
            visit = grouped.setdefault(key, {
                "visit_id": f"{label}/{patient_name}/{visit_date}",
                "label": label,
                "patient_name": patient_name,
                "visit_date": visit_date,
                "axes": {},
            })

            axis_payload = normalize_axis_data(record)
            visit["axes"][axis] = axis_payload["frequencies"]
            label_counts[label] = label_counts.get(label, 0) + 1

    for (label, patient_name, visit_date), visit in grouped.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        safe_name = safe_filename(f"{patient_name}__{visit_date}")
        out_path = label_dir / safe_name
        out_path.write_text(json.dumps(visit, ensure_ascii=False, indent=2), encoding="utf-8")
        total_visits += 1

    summary = {
        "source": str(input_path),
        "output_dir": str(output_dir),
        "total_visits": total_visits,
        "labels": label_counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Format dataset.jsonl into grouped JSON files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dataset/dataset.jsonl"),
        help="Path to source JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset/formatted"),
        help="Folder where formatted JSON files will be saved.",
    )
    args = parser.parse_args()

    summary = build_dataset(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
