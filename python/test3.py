#!/usr/bin/env python3
"""
Synthetic dataset generator for metacat/hypot stress testing.

Features:
- CLI interface
- Multiple schema templates
- Parquet + JSON sidecar generation
- Metacat batch registration file output
"""

import argparse
import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------------------------------------------------
# Schema template interface
# ----------------------------------------------------------------------

@dataclass
class SchemaTemplate:
    name: str
    description: str
    # Generate one JSON metadata dict for a dataset
    metadata_generator: Callable[[], Dict[str, Any]]
    # Generate a pandas DataFrame for the Parquet table
    parquet_generator: Callable[[Dict[str, Any], int], pd.DataFrame]


# ----------------------------------------------------------------------
# Schema: simple event summary
# ----------------------------------------------------------------------

def simple_metadata_generator() -> Dict[str, Any]:
    return {
        "schema": "simple",
        "run_number": random.randint(1000, 2000),
        "subrun": random.randint(0, 50),
        "detector_id": random.choice(range(16)),
        "timestamp": random.randint(1700000000, 1700500000),
        "energy_estimate": float(np.random.normal(50, 10)),
        "quality_flag": random.choice(["good", "suspect", "bad"]),
        "uuid": str(uuid.uuid4()),
    }


def simple_parquet_generator(metadata: Dict[str, Any], nrows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": np.arange(nrows),
        "detector_id": np.full(nrows, metadata["detector_id"]),
        "timestamp": np.random.randint(1700000000, 1700500000, size=nrows),
        "charge": np.random.normal(100, 15, size=nrows),
        "energy": np.random.normal(50, 10, size=nrows),
        "quality_flag": np.random.choice(["good", "suspect", "bad"], size=nrows),
    })


# ----------------------------------------------------------------------
# Schema: calibration-style
# ----------------------------------------------------------------------

def calibration_metadata_generator() -> Dict[str, Any]:
    return {
        "schema": "calibration",
        "run_number": random.randint(2000, 3000),
        "calib_cycle": random.randint(0, 20),
        "detector_id": random.choice(range(8)),
        "timestamp": random.randint(1700000000, 1700500000),
        "temperature": float(np.random.normal(20, 2)),
        "voltage": float(np.random.normal(3.3, 0.1)),
        "uuid": str(uuid.uuid4()),
    }


def calibration_parquet_generator(metadata: Dict[str, Any], nrows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "channel_id": np.arange(nrows),
        "detector_id": np.full(nrows, metadata["detector_id"]),
        "gain": np.random.normal(1.0, 0.05, size=nrows),
        "offset": np.random.normal(0.0, 0.01, size=nrows),
        "noise_rms": np.random.normal(0.5, 0.1, size=nrows),
        "temp_c": np.full(nrows, metadata["temperature"]),
        "voltage_v": np.full(nrows, metadata["voltage"]),
    })


# ----------------------------------------------------------------------
# Schema: beam / spill summary
# ----------------------------------------------------------------------

def beam_metadata_generator() -> Dict[str, Any]:
    return {
        "schema": "beam",
        "run_number": random.randint(3000, 4000),
        "spill_id": random.randint(0, 1000),
        "beam_energy": float(np.random.normal(120, 5)),  # GeV
        "protons_on_target": int(abs(np.random.normal(1e13, 1e12))),
        "timestamp": random.randint(1700000000, 1700500000),
        "uuid": str(uuid.uuid4()),
    }


def beam_parquet_generator(metadata: Dict[str, Any], nrows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "spill_event_index": np.arange(nrows),
        "run_number": np.full(nrows, metadata["run_number"]),
        "spill_id": np.full(nrows, metadata["spill_id"]),
        "beam_energy": np.random.normal(metadata["beam_energy"], 0.5, size=nrows),
        "intensity": np.random.normal(1.0, 0.1, size=nrows),
        "timestamp": np.random.randint(1700000000, 1700500000, size=nrows),
    })


# ----------------------------------------------------------------------
# Available schema templates
# ----------------------------------------------------------------------

SCHEMA_TEMPLATES: Dict[str, SchemaTemplate] = {
    "simple": SchemaTemplate(
        name="simple",
        description="Generic event summary: event_id, detector_id, energy, quality_flag.",
        metadata_generator=simple_metadata_generator,
        parquet_generator=simple_parquet_generator,
    ),
    "calibration": SchemaTemplate(
        name="calibration",
        description="Calibration-style data: channel gains, offsets, noise, temperature, voltage.",
        metadata_generator=calibration_metadata_generator,
        parquet_generator=calibration_parquet_generator,
    ),
    "beam": SchemaTemplate(
        name="beam",
        description="Beam/spill summary: beam energy, protons on target, intensity.",
        metadata_generator=beam_metadata_generator,
        parquet_generator=beam_parquet_generator,
    ),
}


# ----------------------------------------------------------------------
# Core generation logic
# ----------------------------------------------------------------------

def generate_dataset(
    out_root: Path,
    template: SchemaTemplate,
    min_rows: int,
    max_rows: int,
    hierarchical_layout: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Generate one dataset:
    - Parquet file
    - JSON metadata sidecar

    Returns:
        parquet_path, metadata_dict
    """
    metadata = template.metadata_generator()

    # Determine directory layout
    if hierarchical_layout:
        run_str = f"run_{metadata.get('run_number', 0)}"
        subrun = metadata.get("subrun", None)
        calib_cycle = metadata.get("calib_cycle", None)
        spill_id = metadata.get("spill_id", None)

        # pick one secondary key if available
        if subrun is not None:
            subdir = f"subrun_{subrun}"
        elif calib_cycle is not None:
            subdir = f"calib_{calib_cycle}"
        elif spill_id is not None:
            subdir = f"spill_{spill_id}"
        else:
            subdir = "subrun_0"

        dataset_dir = out_root / run_str / subdir
    else:
        dataset_dir = out_root

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Parquet file
    nrows = random.randint(min_rows, max_rows)
    df = template.parquet_generator(metadata, nrows)
    parquet_filename = f"data_{metadata['uuid']}.parquet"
    parquet_path = dataset_dir / parquet_filename

    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)

    # JSON metadata sidecar
    json_path = parquet_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return parquet_path, metadata


def generate_datasets(
    output_dir: Path,
    schema_name: str,
    num_datasets: int,
    min_rows: int,
    max_rows: int,
    flat_layout: bool,
) -> List[Tuple[Path, Dict[str, Any]]]:
    template = SCHEMA_TEMPLATES[schema_name]
    hierarchical_layout = not flat_layout

    results: List[Tuple[Path, Dict[str, Any]]] = []
    for _ in range(num_datasets):
        parquet_path, metadata = generate_dataset(
            output_dir,
            template,
            min_rows,
            max_rows,
            hierarchical_layout=hierarchical_layout,
        )
        results.append((parquet_path, metadata))
    return results


# ----------------------------------------------------------------------
# Metacat batch file generation
# ----------------------------------------------------------------------

def build_metacat_batch(
    datasets: List[Tuple[Path, Dict[str, Any]]],
    root_dir: Path,
) -> Dict[str, Any]:
    """
    Build a generic metacat batch registration structure.

    Each entry includes:
    - logical_name: path relative to root_dir
    - file_path: absolute path
    - size_bytes: file size
    - metadata: JSON metadata dict
    """
    entries = []
    for parquet_path, metadata in datasets:
        stat = parquet_path.stat()
        logical_name = str(parquet_path.relative_to(root_dir))
        entry = {
            "logical_name": logical_name,
            "file_path": str(parquet_path.resolve()),
            "size_bytes": stat.st_size,
            "metadata": metadata,
        }
        entries.append(entry)

    batch = {"datasets": entries}
    return batch


def write_metacat_batch(batch: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(batch, f, indent=2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic dataset generator for metacat/hypot stress testing."
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Root directory for generated data.",
    )
    parser.add_argument(
        "-n", "--num-datasets",
        type=int,
        default=1000,
        help="Number of synthetic datasets to generate.",
    )
    parser.add_argument(
        "-s", "--schema",
        choices=sorted(SCHEMA_TEMPLATES.keys()),
        default="simple",
        help="Schema template to use.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=5,
        help="Minimum number of rows per Parquet file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum number of rows per Parquet file.",
    )
    parser.add_argument(
        "--flat-layout",
        action="store_true",
        help="Use a flat directory layout (no run/subrun hierarchy).",
    )
    parser.add_argument(
        "--metacat-batch",
        type=Path,
        help="Path to write a metacat batch registration JSON file.",
    )
    parser.add_argument(
        "--list-schemas",
        action="store_true",
        help="List available schema templates and exit.",
    )

    return parser.parse_args()


def list_schemas() -> None:
    print("Available schema templates:")
    for name, tmpl in SCHEMA_TEMPLATES.items():
        print(f"  {name:12s} - {tmpl.description}")


def main() -> None:
    args = parse_args()

    if args.list_schemas:
        list_schemas()
        return

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_datasets} datasets")
    print(f"  Schema:        {args.schema}")
    print(f"  Output dir:    {out_dir}")
    print(f"  Row range:     {args.min_rows}–{args.max_rows}")
    print(f"  Layout:        {'flat' if args.flat_layout else 'hierarchical'}")

    datasets = generate_datasets(
        output_dir=out_dir,
        schema_name=args.schema,
        num_datasets=args.num_datasets,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        flat_layout=args.flat_layout,
    )

    if args.metacat_batch:
        print(f"Building metacat batch file: {args.metacat_batch}")
        batch = build_metacat_batch(datasets, root_dir=out_dir)
        write_metacat_batch(batch, args.metacat_batch)

    print("Done.")


if __name__ == "__main__":
    main()
