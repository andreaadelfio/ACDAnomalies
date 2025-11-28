"""Utility per unire i CSV *_in_catalog.csv nel dataset finale."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def collect_catalog_files(root: Path) -> List[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*_in_catalog.csv") if path.is_file())


def build_frame(csv_path: Path, root: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if df.empty:
		df = pd.DataFrame(columns=df.columns)

	return df


def merge_catalogs(root: Path, output: Path) -> Path:
	files = collect_catalog_files(root)
	if not files:
		raise FileNotFoundError(f"Nessun file *_in_catalog.csv trovato in {root}")

	frames = [build_frame(path, root) for path in files]
	merged = pd.concat(frames, ignore_index=True)

	sort_columns = [
		col
		for col in ["results_day", "anomaly_id", "detection_id", "start_datetime"]
		if col in merged.columns
	]
	if sort_columns:
		merged.sort_values(by=sort_columns, inplace=True, ignore_index=True)

	output.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(output, index=False)
	return output


def main(cli_args) -> None:
	output = merge_catalogs(Path(cli_args), Path(cli_args + '/total_detections_in_catalog.csv'))
	print(f"Merge completato: {output}")


if __name__ == "__main__":
	main('/home/scutini/ACDAnomalies/acd/results/2025-11-10/')
