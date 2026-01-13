#!/usr/bin/env python3
"""
Synthetic dataset generator for metacat/hypot stress testing.

Features:
- CLI interface
- Multiple schema templates
- Parquet + JSON sidecar generation
- Metacat batch registration file output
- Pathology knobs (nulls, NaNs, extreme sizes, duplicate logical names)
- Schema corruption knobs (missing/extra/wrong-type/mixed/reordered/empty/duplicate columns)
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
# Configuration dataclasses
# ----------------------------------------------------------------------

@dataclass
class PathologyConfig:
    null_rate: float = 0.0
    nan_rate: float = 0.0
    extreme_size_rate: float = 0.0
    duplicate_rate: float = 0.0


@dataclass
class SchemaCorruptionConfig:
    missing_col_rate: float = 0.0
    extra_col_rate: float = 0.0
    wrong_type_rate: float = 0.0
    mixed_type_rate: float = 0.0
    reorder_col_rate: float = 0.0
    empty_table_rate: float = 0.0
    duplicate_col_rate: float = 0.0


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
# Pathology helpers (metadata + rows + logical-name duplicates)
# ----------------------------------------------------------------------

def apply_metadata_pathologies(metadata: Dict[str, Any], cfg: PathologyConfig) -> Dict[str, Any]:
    # Null injection
    if cfg.null_rate > 0:
        for k in list(metadata.keys()):
            if random.random() < cfg.null_rate:
                metadata[k] = None

    # NaN injection
    if cfg.nan_rate > 0:
        for k, v in list(metadata.items()):
            if isinstance(v, (int, float)) and random.random() < cfg.nan_rate:
                metadata[k] = float("nan")

    return metadata


def choose_row_count(min_rows: int, max_rows: int, cfg: PathologyConfig) -> int:
    if random.random() < cfg.extreme_size_rate:
        # 50% tiny, 50% huge
        if random.random() < 0.5:
            return random.randint(0, 1)  # tiny
        else:
            return random.randint(5000, 20000)  # huge
    return random.randint(min_rows, max_rows)


# ----------------------------------------------------------------------
# Schema corruption helpers (Parquet DataFrame)
# ----------------------------------------------------------------------

def corrupt_dataframe(df: pd.DataFrame, cfg: SchemaCorruptionConfig) -> pd.DataFrame:
    # Empty table
    if random.random() < cfg.empty_table_rate:
        return pd.DataFrame()

    # Missing columns
    if cfg.missing_col_rate > 0 and len(df.columns) > 1 and random.random() < cfg.missing_col_rate:
        drop_col = random.choice(df.columns.tolist())
        df = df.drop(columns=[drop_col])

    # Extra columns
    if cfg.extra_col_rate > 0 and random.random() < cfg.extra_col_rate:
        df["extra_col_" + str(uuid.uuid4())[:8]] = np.random.normal(0, 1, size=len(df))

    # Wrong types: force one column to string
    if cfg.wrong_type_rate > 0 and len(df.columns) > 0 and random.random() < cfg.wrong_type_rate:
        col = random.choice(df.columns.tolist())
        df[col] = df[col].astype(str)

    # Mixed types: put a dict into a random row of a random column
    if cfg.mixed_type_rate > 0 and len(df.columns) > 0 and len(df) > 0 and random.random() < cfg.mixed_type_rate:
        col = random.choice(df.columns.tolist())
        row_idx = random.randint(0, len(df) - 1)
        df.at[row_idx, col] = {"weird": "object"}

    # Reorder columns
    if cfg.reorder_col_rate > 0 and len(df.columns) > 1 and random.random() < cfg.reorder_col_rate:
        cols = df.columns.tolist()
        random.shuffle(cols)
        df = df[cols]

    # Duplicate column names
    if cfg.duplicate_col_rate > 0 and len(df.columns) > 0 and random.random() < cfg.duplicate_col_rate:
        dup_col = random.choice(df.columns.tolist())
        df[dup_col + "_dup"] = df[dup_col]
        df.rename(columns={dup_col + "_dup": dup_col}, inplace=True)

    return df


# ----------------------------------------------------------------------
# Core generation logic
# ----------------------------------------------------------------------

def generate_dataset(
    out_root: Path,
    template: SchemaTemplate,
    min_rows: int,
    max_rows: int,
    hierarchical_layout: bool,
    patho_cfg: PathologyConfig,
    corrupt_cfg: SchemaCorruptionConfig,
    duplicate_pool: List[str],
) -> Tuple[Path, Dict[str, Any], str]:
    """
    Generate one dataset:
    - Parquet file (possibly corrupted)
    - JSON metadata sidecar (possibly pathological)
    - logical_name used for metacat

    Returns:
        parquet_path, metadata_dict, logical_name
    """
    metadata = template.metadata_generator()
    metadata = apply_metadata_pathologies(metadata, patho_cfg)

    # Determine directory layout
    if hierarchical_layout:
        run_str = f"run_{metadata.get('run_number', 0)}"
        subrun = metadata.get("subrun", None)
        calib_cycle = metadata.get("calib_cycle", None)
        spill_id = metadata.get("spill_id", None)

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

    # Decide row count (with extreme size pathologies)
    nrows = choose_row_count(min_rows, max_rows, patho_cfg)

    # Parquet DataFrame + corruption
    df = template.parquet_generator(metadata, nrows)
    df = corrupt_dataframe(df, corrupt_cfg)

    parquet_filename = f"data_{metadata.get('uuid', str(uuid.uuid4()))}.parquet"
    parquet_path = dataset_dir / parquet_filename

    # Decide logical_name (possibly duplicate)
    if patho_cfg.duplicate_rate > 0 and duplicate_pool and random.random() < patho_cfg.duplicate_rate:
        logical_name = random.choice(duplicate_pool)
    else:
        logical_name = str(parquet_path.relative_to(out_root))
        duplicate_pool.append(logical_name)

    # Write Parquet file
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)

    # JSON metadata sidecar
    json_path = parquet_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return parquet_path, metadata, logical_name


def generate_datasets(
    output_dir: Path,
    schema_name: str,
    num_datasets: int,
    min_rows: int,
    max_rows: int,
    flat_layout: bool,
    patho_cfg: PathologyConfig,
    corrupt_cfg: SchemaCorruptionConfig,
) -> List[Tuple[Path, Dict[str, Any], str]]:
    template = SCHEMA_TEMPLATES[schema_name]
    hierarchical_layout = not flat_layout

    results: List[Tuple[Path, Dict[str, Any], str]] = []
    duplicate_pool: List[str] = []

    for _ in range(num_datasets):
        parquet_path, metadata, logical_name = generate_dataset(
            out_root=output_dir,
            template=template,
            min_rows=min_rows,
            max_rows=max_rows,
            hierarchical_layout=hierarchical_layout,
            patho_cfg=patho_cfg,
            corrupt_cfg=corrupt_cfg,
            duplicate_pool=duplicate_pool,
        )
        results.append((parquet_path, metadata, logical_name))
    return results


# ----------------------------------------------------------------------
# Metacat batch file generation
# ----------------------------------------------------------------------

def build_metacat_batch(
    datasets: List[Tuple[Path, Dict[str, Any], str]],
) -> Dict[str, Any]:
    """
    Build a generic metacat batch registration structure.

    Each entry includes:
    - logical_name: path (possibly duplicated) used as dataset name
    - file_path: absolute path
    - size_bytes: file size
    - metadata: JSON metadata dict
    """
    entries = []
    for parquet_path, metadata, logical_name in datasets:
        stat = parquet_path.stat()
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

    # Core
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
        help="Minimum number of rows per (non-pathological) Parquet file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum number of rows per (non-pathological) Parquet file.",
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

    # Pathology knobs
    parser.add_argument(
        "--null-rate",
        type=float,
        default=0.0,
        help="Probability of replacing each metadata field with null.",
    )
    parser.add_argument(
        "--nan-rate",
        type=float,
        default=0.0,
        help="Probability of injecting NaN into individual numeric metadata fields.",
    )
    parser.add_argument(
        "--extreme-size-rate",
        type=float,
        default=0.0,
        help="Probability of generating extremely small (0–1 rows) or large (5k–20k rows) tables.",
    )
    parser.add_argument(
        "--duplicate-rate",
        type=float,
        default=0.0,
        help="Probability that a dataset reuses a previous logical name (duplicate).",
    )

    # Schema corruption knobs
    parser.add_argument(
        "--missing-col-rate",
        type=float,
        default=0.0,
        help="Probability of dropping a column from the Parquet table.",
    )
    parser.add_argument(
        "--extra-col-rate",
        type=float,
        default=0.0,
        help="Probability of adding an unexpected column to the Parquet table.",
    )
    parser.add_argument(
        "--wrong-type-rate",
        type=float,
        default=0.0,
        help="Probability of forcing a Parquet column to string dtype.",
    )
    parser.add_argument(
        "--mixed-type-rate",
        type=float,
        default=0.0,
        help="Probability of inserting a non-scalar object into a column.",
    )
    parser.add_argument(
        "--reorder-col-rate",
        type=float,
        default=0.0,
        help="Probability of shuffling column order.",
    )
    parser.add_argument(
        "--empty-table-rate",
        type=float,
        default=0.0,
        help="Probability of creating a Parquet file with an empty table (no columns).",
    )
    parser.add_argument(
        "--duplicate-col-rate",
        type=float,
        default=0.0,
        help="Probability of creating duplicate column names in the Parquet table.",
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

    patho_cfg = PathologyConfig(
        null_rate=args.null_rate,
        nan_rate=args.nan_rate,
        extreme_size_rate=args.extreme_size_rate,
        duplicate_rate=args.duplicate_rate,
    )
    corrupt_cfg = SchemaCorruptionConfig(
        missing_col_rate=args.missing_col_rate,
        extra_col_rate=args.extra_col_rate,
        wrong_type_rate=args.wrong_type_rate,
        mixed_type_rate=args.mixed_type_rate,
        reorder_col_rate=args.reorder_col_rate,
        empty_table_rate=args.empty_table_rate,
        duplicate_col_rate=args.duplicate_col_rate,
    )

    print(f"Generating {args.num_datasets} datasets")
    print(f"  Schema:        {args.schema}")
    print(f"  Output dir:    {out_dir}")
    print(f"  Row range:     {args.min_rows}–{args.max_rows}")
    print(f"  Layout:        {'flat' if args.flat_layout else 'hierarchical'}")
    print(f"  Pathologies:   null={patho_cfg.null_rate}, nan={patho_cfg.nan_rate}, "
          f"extreme_size={patho_cfg.extreme_size_rate}, duplicate_ln={patho_cfg.duplicate_rate}")
    print(f"  Corruptions:   missing={corrupt_cfg.missing_col_rate}, extra={corrupt_cfg.extra_col_rate}, "
          f"wrong_type={corrupt_cfg.wrong_type_rate}, mixed={corrupt_cfg.mixed_type_rate}, "
          f"reorder={corrupt_cfg.reorder_col_rate}, empty={corrupt_cfg.empty_table_rate}, "
          f"dup_col={corrupt_cfg.duplicate_col_rate}")

    datasets = generate_datasets(
        output_dir=out_dir,
        schema_name=args.schema,
        num_datasets=args.num_datasets,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        flat_layout=args.flat_layout,
        patho_cfg=patho_cfg,
        corrupt_cfg=corrupt_cfg,
    )

    if args.metacat_batch:
        print(f"Building metacat batch file: {args.metacat_batch}")
        batch = build_metacat_batch(datasets)
        write_metacat_batch(batch, args.metacat_batch)

    print("Done.")


if __name__ == "__main__":
    main()
