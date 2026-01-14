#!/usr/bin/env python3
"""
Minimal synthetic dataset generator for metacat/hypot ingestion testing.

Refactored version using MinimalFileBuilder:
  - Clean separation of concerns
  - Realistic hypotpro metadata schema
  - Parquet + JSON sidecar
  - Optional metacat batch file
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------------------------------------------------
# MinimalSchemaTemplate — matches real hypotpro metadata conventions
# ----------------------------------------------------------------------

class MinimalSchemaTemplate:
    """
    Schema template for synthetic_generator_minimal.py.
    Matches the metadata and attribute structure observed in the
    hypotpro namespace on Fermilab metacat.
    """

    def __init__(
        self,
        fn_description,
        fn_configuration,
        rs_runs,
        fn_format="txt",
        fn_owner="hypotraw",
        fn_tier="etc",
        dh_type="other",
        checksum=None,
        checksum_type="adler32",
        size=None,
        datasets=None,
    ):
        self.metadata = {
            "dh.type": dh_type,
            "fn.configuration": fn_configuration,
            "fn.description": fn_description,
            "fn.format": fn_format,
            "fn.owner": fn_owner,
            "fn.tier": fn_tier,
            "rs.runs": rs_runs if isinstance(rs_runs, list) else [rs_runs],
        }

        self.attributes = {
            "checksum_type": checksum_type,
            "checksum": checksum,
            "size": size,
        }

        self.datasets = datasets if datasets is not None else []

    def to_metacat_dict(self, namespace, logical_name):
        return {
            "namespace": namespace,
            "name": logical_name,
            "metadata": self.metadata,
            "attributes": self.attributes,
            "datasets": self.datasets,
        }


# ----------------------------------------------------------------------
# MinimalFileBuilder — core file creation engine
# ----------------------------------------------------------------------

class MinimalFileBuilder:
    """
    High-level builder for creating a synthetic file + metadata + attributes
    that match the hypotpro metacat namespace conventions.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_parquet(self, nrows: int) -> Tuple[Path, pd.DataFrame]:
        df = pd.DataFrame({
            "event_id": np.arange(nrows),
            "timestamp": np.random.randint(1700000000, 1700500000, size=nrows),
            "charge": np.random.normal(100, 15, size=nrows),
            "energy": np.random.normal(50, 10, size=nrows),
            "quality_flag": np.random.choice(["good", "suspect", "bad"], size=nrows),
        })

        uuid_str = str(uuid.uuid4())
        parquet_filename = f"data_{uuid_str}.parquet"
        parquet_path = self.output_dir / parquet_filename

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, parquet_path)

        return parquet_path, df

    def _generate_metadata_fields(self) -> Dict[str, Any]:
        return {
            "fn_description": f"declad_test_{random.randint(1, 5)}",
            "fn_configuration": "c20240226",
            "rs_runs": [random.randint(1000000, 1000010)],
            "fn_format": "txt",
            "fn_owner": "hypotraw",
            "fn_tier": "etc",
            "dh_type": "other",
        }

    def build_file(self, min_rows: int, max_rows: int):
        nrows = random.randint(min_rows, max_rows)
        parquet_path, df = self._generate_parquet(nrows)

        stat = parquet_path.stat()
        checksum = f"{random.getrandbits(32):08x}"

        md_fields = self._generate_metadata_fields()

        schema = MinimalSchemaTemplate(
            fn_description=md_fields["fn_description"],
            fn_configuration=md_fields["fn_configuration"],
            rs_runs=md_fields["rs_runs"],
            fn_format=md_fields["fn_format"],
            fn_owner=md_fields["fn_owner"],
            fn_tier=md_fields["fn_tier"],
            dh_type=md_fields["dh_type"],
            checksum=checksum,
            size=stat.st_size,
            datasets=[
                "hypotpro:declad_test1",
                "hypotpro:declad_test3",
            ],
        )

        json_path = parquet_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(schema.metadata, f, indent=2)

        logical_name = parquet_path.name
        return parquet_path, schema, logical_name

    def to_metacat_entry(self, parquet_path: Path, schema: MinimalSchemaTemplate, logical_name: str):
        entry = schema.to_metacat_dict(
            namespace="hypotpro",
            logical_name=logical_name,
        )
        entry["file_path"] = str(parquet_path.resolve())
        return entry


# ----------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------

def generate_datasets(
    output_dir: Path,
    num_datasets: int,
    min_rows: int,
    max_rows: int,
) -> List[Tuple[Path, MinimalSchemaTemplate, str]]:

    builder = MinimalFileBuilder(output_dir)
    results = []

    for _ in range(num_datasets):
        results.append(
            builder.build_file(
                min_rows=min_rows,
                max_rows=max_rows,
            )
        )

    return results


# ----------------------------------------------------------------------
# Metacat batch file
# ----------------------------------------------------------------------

def build_metacat_batch(
    datasets: List[Tuple[Path, MinimalSchemaTemplate, str]]
) -> Dict[str, Any]:

    entries = []
    builder = None  # will be replaced per-entry

    for parquet_path, schema, logical_name in datasets:
        if builder is None:
            builder = MinimalFileBuilder(parquet_path.parent)
        entry = builder.to_metacat_entry(parquet_path, schema, logical_name)
        entries.append(entry)

    return {"files": entries}


def write_metacat_batch(batch: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(batch, f, indent=2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal synthetic dataset generator."
    )

    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True,
        help="Directory for generated data."
    )
    parser.add_argument(
        "-n", "--num-datasets", type=int, default=100,
        help="Number of datasets to generate."
    )
    parser.add_argument(
        "--min-rows", type=int, default=5,
        help="Minimum rows per Parquet file."
    )
    parser.add_argument(
        "--max-rows", type=int, default=20,
        help="Maximum rows per Parquet file."
    )
    parser.add_argument(
        "--metacat-batch", type=Path,
        help="Write a metacat batch registration JSON file."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_datasets} minimal datasets")
    datasets = generate_datasets(
        output_dir=out_dir,
        num_datasets=args.num_datasets,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
    )

    if args.metacat_batch:
        batch = build_metacat_batch(datasets)
        write_metacat_batch(batch, args.metacat_batch)
        print(f"Wrote batch file: {args.metacat_batch}")

    print("Done.")


if __name__ == "__main__":
    main()
