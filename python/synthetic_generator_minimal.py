#!/usr/bin/env python3
"""
Minimal synthetic dataset generator for metacat ingestion testing.

Features:
  - Parquet + JSON sidecar generation
  - MetaCat v4.1.2-compatible batch files
  - Namespace, owner, dataset prefix configurable
  - NEW: --num-datasets controls dataset grouping
  - Emits one batch file per dataset
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
# MinimalSchemaTemplate
# ----------------------------------------------------------------------

class MinimalSchemaTemplate:
    """
    Metadata + attributes container matching hypotpro conventions.
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

    def to_flat_metacat_entry(self, namespace, logical_name, file_path):
        """
        Produce a MetaCat v4.1.2-compatible flat entry.
        """
        return {
            "namespace": namespace,
            "name": logical_name,
            "size": self.attributes["size"],
            "checksum": self.attributes["checksum"],
            "checksum_type": self.attributes["checksum_type"],
            "metadata": self.metadata,
            "parents": [],
            "file_path": file_path,
        }


# ----------------------------------------------------------------------
# MinimalFileBuilder
# ----------------------------------------------------------------------

class MinimalFileBuilder:
    """
    Creates Parquet files, metadata, and batch entries.
    """

    def __init__(self, output_dir: Path, namespace: str, owner: str, dataset_prefix: str):
        self.output_dir = output_dir
        self.namespace = namespace
        self.owner = owner
        self.dataset_prefix = dataset_prefix
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
            "fn_description": f"{self.dataset_prefix}_{random.randint(1, 5)}",
            "fn_configuration": "c20240226",
            "rs_runs": [random.randint(1000000, 1000010)],
            "fn_format": "txt",
            "fn_owner": self.owner,
            "fn_tier": "etc",
            "dh_type": "other",
        }

    def _dataset_name(self, dataset_index: int) -> str:
        return f"{self.namespace}:{self.dataset_prefix}_{dataset_index}"

    def build_file(self, min_rows: int, max_rows: int, dataset_index: int):
        nrows = random.randint(min_rows, max_rows)
        parquet_path, df = self._generate_parquet(nrows)

        stat = parquet_path.stat()
        checksum = f"{random.getrandbits(32):08x}"

        md_fields = self._generate_metadata_fields()
        dataset_name = self._dataset_name(dataset_index)

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
            datasets=[dataset_name],
        )

        json_path = parquet_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(schema.metadata, f, indent=2)

        logical_name = parquet_path.name
        return parquet_path, schema, logical_name, dataset_index

    def to_metacat_entry(self, parquet_path: Path, schema: MinimalSchemaTemplate, logical_name: str):
        return schema.to_flat_metacat_entry(
            namespace=self.namespace,
            logical_name=logical_name,
            file_path=str(parquet_path.resolve()),
        )


# ----------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------

def generate_datasets(
    output_dir: Path,
    namespace: str,
    owner: str,
    dataset_prefix: str,
    num_datasets: int,
    num_files: int,
    min_rows: int,
    max_rows: int,
) -> List[Tuple[Path, MinimalSchemaTemplate, str, int]]:

    builder = MinimalFileBuilder(output_dir, namespace, owner, dataset_prefix)
    results = []

    for i in range(num_files):
        dataset_index = (i % num_datasets) + 1
        results.append(
            builder.build_file(
                min_rows=min_rows,
                max_rows=max_rows,
                dataset_index=dataset_index,
            )
        )

    return results


# ----------------------------------------------------------------------
# Batch file creation
# ----------------------------------------------------------------------

def build_metacat_batches(
    datasets: List[Tuple[Path, MinimalSchemaTemplate, str, int]],
    namespace: str,
    owner: str,
    dataset_prefix: str,
    output_dir: Path,
) -> Dict[int, List[Dict[str, Any]]]:

    batches = {}

    for parquet_path, schema, logical_name, dataset_index in datasets:
        entry = schema.to_flat_metacat_entry(
            namespace=namespace,
            logical_name=logical_name,
            file_path=str(parquet_path.resolve()),
        )

        if dataset_index not in batches:
            batches[dataset_index] = []

        batches[dataset_index].append(entry)

    return batches


def write_metacat_batch(batch: List[Dict[str, Any]], output_path: Path) -> None:
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

    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("-n", "--num-files", type=int, default=100)
    parser.add_argument("--num-datasets", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--namespace", type=str, default="hypotpro")
    parser.add_argument("--owner", type=str, default="hypotraw")
    parser.add_argument("--dataset-prefix", type=str, default="declad_test")
    parser.add_argument("--metacat-batch", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.num_files} files across {args.num_datasets} datasets "
        f"in namespace '{args.namespace}' with owner '{args.owner}' "
        f"and dataset prefix '{args.dataset_prefix}'"
    )

    datasets = generate_datasets(
        output_dir=out_dir,
        namespace=args.namespace,
        owner=args.owner,
        dataset_prefix=args.dataset_prefix,
        num_datasets=args.num_datasets,
        num_files=args.num_files,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
    )

    if args.metacat_batch:
        batches = build_metacat_batches(
            datasets,
            args.namespace,
            args.owner,
            args.dataset_prefix,
            out_dir,
        )

        for idx, entries in batches.items():
            batch_path = out_dir / f"batch_{args.dataset_prefix}_{idx}.json"
            write_metacat_batch(entries, batch_path)
            print(f"Wrote batch file: {batch_path}")

    print("Done.")


if __name__ == "__main__":
    main()
