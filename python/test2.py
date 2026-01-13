import os, json, random, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

N = 5000  # number of synthetic datasets
OUT = Path("synthetic_data")
OUT.mkdir(exist_ok=True)

def random_metadata():
    return {
        "run_number": random.randint(1000, 2000),
        "subrun": random.randint(0, 50),
        "detector_id": random.choice(range(16)),
        "timestamp": random.randint(1700000000, 1700500000),
        "energy_estimate": float(np.random.normal(50, 10)),
        "quality_flag": random.choice(["good", "suspect", "bad"]),
        "uuid": str(uuid.uuid4())
    }

def generate_parquet(path, detector_id):
    nrows = random.randint(5, 20)
    df = pd.DataFrame({
        "event_id": np.arange(nrows),
        "detector_id": np.full(nrows, detector_id),
        "timestamp": np.random.randint(1700000000, 1700500000, size=nrows),
        "charge": np.random.normal(100, 15, size=nrows),
        "energy": np.random.normal(50, 10, size=nrows),
        "quality_flag": np.random.choice(["good", "suspect", "bad"], size=nrows)
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

for i in range(N):
    md = random_metadata()
    run_dir = OUT / f"run_{md['run_number']}" / f"subrun_{md['subrun']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fname = f"data_{md['uuid']}.parquet"
    fpath = run_dir / fname

    generate_parquet(fpath, md["detector_id"])

    with open(fpath.with_suffix(".json"), "w") as f:
        json.dump(md, f, indent=2)

