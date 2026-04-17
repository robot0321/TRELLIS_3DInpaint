#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


METHODS = ("optinit", "repaint", "sdedit", "blendeddiffusion", "ilvr", "dps", "md")


def to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def detect_method(path: Path) -> Optional[str]:
    for part in path.parts:
        if part in METHODS:
            return part
    return None


def read_numeric_rows(csv_path: Path) -> Tuple[int, List[Dict[str, float]]]:
    total_rows = 0
    numeric_rows: List[Dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return total_rows, numeric_rows
        for row in reader:
            total_rows += 1
            numeric = {}
            for k, v in row.items():
                fv = to_float(v if v is not None else "")
                if fv is not None:
                    numeric[k] = fv
            if numeric:
                numeric_rows.append(numeric)
    return total_rows, numeric_rows


def mean_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        for k, v in row.items():
            sums[k] += v
            counts[k] += 1
    out = {}
    for k in sums:
        out[k] = sums[k] / counts[k]
    return out


def collect(logs_root: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    method_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    method_file_count: Dict[str, int] = defaultdict(int)
    method_total_rows: Dict[str, int] = defaultdict(int)
    method_numeric_rows: Dict[str, int] = defaultdict(int)
    metric_keys = set()

    for csv_path in logs_root.rglob("evaluation_metrics.csv"):
        method = detect_method(csv_path)
        if method is None:
            continue

        total_rows, numeric_rows = read_numeric_rows(csv_path)
        method_file_count[method] += 1
        method_total_rows[method] += total_rows
        method_numeric_rows[method] += len(numeric_rows)
        if numeric_rows:
            method_rows[method].extend(numeric_rows)
            for row in numeric_rows:
                metric_keys.update(row.keys())

    metric_cols = sorted(metric_keys)
    out_rows: List[Dict[str, object]] = []

    for method in METHODS:
        if method_file_count.get(method, 0) == 0:
            continue
        m = mean_rows(method_rows.get(method, []))
        item: Dict[str, object] = {
            "method": method,
            "num_files": method_file_count[method],
            "num_rows": method_total_rows[method],
            "num_numeric_rows": method_numeric_rows[method],
        }
        for col in metric_cols:
            item[col] = m.get(col, "")
        out_rows.append(item)

    return out_rows, metric_cols


def write_csv(out_path: Path, rows: List[Dict[str, object]], metric_cols: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "num_files", "num_rows", "num_numeric_rows"] + metric_cols
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gather evaluation_metrics.csv under logs_root and compute method-level means "
            "for numeric columns."
        )
    )
    parser.add_argument("--logs_root", type=Path, required=True, help="Root directory containing per-sample folders.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <logs_root_parent>/eval_reconstruction.csv)",
    )
    args = parser.parse_args()

    logs_root = args.logs_root
    if not logs_root.exists():
        raise FileNotFoundError(f"logs_root does not exist: {logs_root}")

    out_path = args.output or (logs_root / "eval_reconstruction.csv")
    rows, metric_cols = collect(logs_root)
    if not rows:
        raise FileNotFoundError(
            f"No method-tagged evaluation_metrics.csv files found under: {logs_root}"
        )

    write_csv(out_path, rows, metric_cols)
    print(f"Saved: {out_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
