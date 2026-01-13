import os, json, random, uuid
from pathlib import Path
import numpy as np

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

def generate_file(path, size_kb=10):
    with open(path, "wb") as f:
        f.write(os.urandom(size_kb * 1024))

for i in range(N):
    md = random_metadata()
    run_dir = OUT / f"run_{md['run_number']}" / f"subrun_{md['subrun']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fname = f"data_{md['uuid']}.bin"
    fpath = run_dir / fname

    generate_file(fpath, size_kb=random.randint(5, 200))

    with open(fpath.with_suffix(".json"), "w") as f:
        json.dump(md, f, indent=2)
