# MetacatStressTest
Python code for creating a dataset for a stress test of Metacat, etc.

This package was developed using Microsoft Copilot.

***Set up on fifeutilgpvm03:***

1. Install miniforge3:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
```

2. Setup conda:

```
source ~/miniforge3/etc/profile.d/conda.sh
```

3. Create metacat-stress environment:

Place the following into a metacat-stress.yml file:

```
name: metacat-stress
channels:
  - conda-forge
dependencies:
  # Core Python
  - python=3.11

  # Data formats
  - pyarrow
  - fastparquet
  - pandas
  - numpy

  # JSON/YAML metadata handling
  - pyyaml
  - orjson

  # CLI + tooling
  - click
  - rich
  - typer

  # Randomized/adversarial data generation
  - faker
  - hypothesis
  - python-rapidjson

  # File generation + corruption knobs
  - pillow
  - zstandard

  # Performance + concurrency
  - tqdm
  - aiofiles
  - anyio

  # Optional: FITS support if you want astronomy‑style payloads
  - astropy

  # Optional: HDF5/ROOT payloads for DUNE/Mu2E realism
  - h5py
  - uproot

  # Testing + validation
  - pytest
  - jsonschema

  # Install mamba for faster package ops
  - mamba
```

Then run:

```
mamba env create -f metacat-stress.yml
```

4. Activate metacat-stress environment:

```   
conda activate metacat-stress
```

***Example runs for synthetic_generator_minimal.py:***

1.  Generate 10 synthetic files:

```
./synthetic_generator_minimal.py -o ./synthetic_minimal -n 10
```

2. Generate 50 files, each with 20–40 rows:

```
./synthetic_generator_minimal.py \
    -o ./synthetic_minimal \
    -n 50 \
    --min-rows 20 \
    --max-rows 40
```

3.  Generate 25 files and write a batch JSON:

```
./synthetic_generator_minimal.py \
    -o ./synthetic_minimal \
    -n 25 \
    --metacat-batch ./synthetic_minimal/batch.json
```

4.  Generate 200 files, larger row counts, and a batch file:

```
./synthetic_generator_minimal.py \
    -o ./synthetic_minimal \
    -n 200 \
    --min-rows 50 \
    --max-rows 200 \
    --metacat-batch ./synthetic_minimal/hypotpro_batch.json
```

5.  Same as #4, but explictly providing additinal metacat namespace/dataset info:

```
./synthetic_generator_minimal.py \
        -o ./synthetic_minimal \
        -n 200 \
        --min-rows 50 \
        --max-rows 200 \
        --metacat-batch ./synthetic_minimal/batch_synthetic_minimal.json \
        --namespace dtucker_metacat_tests_2 \
        --owner dtucker \
        --dataset-prefix test_20260129 \
        --num-datasets 5
```


***Example runs for synthetic_generator.py (still does not conform to standard schemas):***

1. Basic:

```
python synthetic_generator.py \
  -o synthetic_clean \
  -n 5000 \
  -s simple \
  --metacat-batch metacat_batch_clean.json
```

2. More advanced:

```
python synthetic_generator.py \
  -o synthetic_stress \
  -n 10000 \
  -s simple \
  --min-rows 5 \
  --max-rows 20 \
  --null-rate 0.02 \
  --nan-rate 0.02 \
  --extreme-size-rate 0.01 \
  --missing-col-rate 0.02 \
  --extra-col-rate 0.02 \
  --reorder-col-rate 0.02 \
  --empty-table-rate 0.005 \
  --metacat-batch metacat_batch_stress.json

```
