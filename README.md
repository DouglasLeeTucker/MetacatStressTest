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
  # Match Spack's Python major/minor version
  - python=3.9

  # Core scientific stack (Python 3.9 compatible)
  - numpy
  - pandas
  - pyarrow
  - fastparquet
  - scipy

  # JSON/YAML metadata handling
  - pyyaml
  - orjson
  - jsonschema

  # CLI + tooling
  - click
  - rich
  - typer

  # Randomized/adversarial data generation
  - faker
  - hypothesis
  - python-rapidjson

  # File generation + compression
  - pillow
  - zstandard

  # Performance + concurrency
  - tqdm
  - aiofiles
  - anyio

  # Optional: astronomy formats
  - astropy

  # Optional: HDF5/ROOT payloads
  - h5py
  - uproot

  # Testing
  - pytest

  # Faster package operations
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
        --metacat-batch \
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

***Example run of end-to-end MetaCat stress test:***

The script `metacat_stress_pipeline.sh` will:
1. Generate N synthetic files.
2. Declare all N synthetic files to MetaCat.
3. Create a child-of-N-parents JSON file.
4. Declare the child-of-N-parents JSON file to MetaCat.

One can optionally choose any (or all) of the above steps to run in a given run of `metacat_stress_pipeline.sh`.

Here are the arguments to `metacat_stress_pipeline.sh`:

```
$ metacat_stress_pipeline.sh --help
Usage: /home/dtucker/bin/metacat_stress_pipeline.sh [options] [stage flags]

Options:
  --namespace <ns>
  --dataset <name>
  --workdir <path>
  --synthetic-dir <path>
  --child-name <name>
  --num-parents <N>
  --verbose
  --dry-run
  --logfile <path>
  --debug-env        Print environment diagnostics and exit

Stage flags:
  --run-generate
  --run-declare-synthetic
  --run-create-child
  --run-declare-child
```

***NOTE:  Currently, the namespace used must have already been created in MetaCat!!!***

Here is an example run of `metacat_stress_pipeline.sh`.  
Here, the namespace `dtucker_metacat_stress_tests` has been previously created.
This command:
1. creates a set of N=100K small parquet files, their metadata JSON files, and a combined batch JSON file in the subdirectory `synthetic_minimal_n100000` in `~/WORK/GitHub/MetacatStressTest/python/`;
2. declares these 100K parquet files to the hypot experment in MetaCat (via the combined batch JSON file);
3. creates a child file (called `metacat_stress_test_20260212_child_of_100000_parents`) whose parents consist of all 100K files whose metadata have just been declared;
4. declares that child file to the hypot experiment in MetaCat.

It also provides a log file (with verbose output) in `~/WORK/GitHub/MetacatStressTest/python/metacat_stress_test_20260212.log`.

```
./metacat_stress_pipeline.sh \
    --namespace dtucker_metacat_stress_tests \
    --dataset metacat_stress_test_20260212 \
    --workdir ~/WORK/GitHub/MetacatStressTest/python \
    --synthetic-dir synthetic_minimal_n100000 \
    --child-name metacat_stress_test_20260212_child_of_100000_parents \
    --num-parents 100000 \
    --verbose \
    --logfile ~/WORK/GitHub/MetacatStressTest/python/metacat_stress_test_20260212.log \
    --run-generate \
    --run-declare-synthetic \
    --run-create-child \
    --run-declare-child
```
