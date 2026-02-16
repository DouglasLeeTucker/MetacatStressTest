date

# MetaCat setup
source /cvmfs/fermilab.opensciencegrid.org/packages/common/setup-env.sh
spack load r-m-dd-config@1.8 experiment=hypot
htgettoken -i hypot -a htvaultprod.fnal.gov
metacat auth login -m token $USER

# Setup metacat-stress environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate metacat-stress

# Change directory to location of the child file and JSON batch file to be created
cd ~/WORK/GitHub/MetacatStressTest/python/synthetic_minimal_n100000

date

# Run python3 script to create child file and its JSON file
# Prepared by Microsoft CoPilot+GPT-5.1, 2026-02-06
python3 - <<'EOF'
import pyarrow as pa, pyarrow.parquet as pq, zlib, os, json, subprocess, sys

# 1. Create a tiny Parquet child file
table = pa.table({"value": [1, 2, 3]})
pq.write_table(table, "child_of_100000_parents.parquet")

# 2. Compute size and checksum
size = os.path.getsize("child_of_100000_parents.parquet")
checksum = format(zlib.adler32(open("child_of_100000_parents.parquet","rb").read()) & 0xffffffff, "08x")

# 3. Query MetaCat for all parents
cmd = ["metacat", "query", "files from dtucker_metacat_stress_tests:test_20260211"]
parents_raw = subprocess.check_output(cmd).decode().strip().split("\n")
parents = [p for p in parents_raw if p.strip()]

# 4. Build the MetaCat declaration
entry = {
    "namespace": "dtucker_metacat_stress_tests",
    "name": "test_20260211_child_of_100000_parents",
    "size": size,
    "checksum": checksum,
    "checksum_type": "adler32",
    "metadata": {
        "fn.description": "Child of 10,000 synthetic files",
        "fn.owner": "dtucker"
    },
    "parents": parents,
    "file_path": os.path.abspath("child_of_100000_parents.parquet")
}

# 5. Write JSON
with open("child_of_100000_parents_entry.json", "w") as f:
    json.dump(entry, f, indent=2)

print("Created child_of_100000_parents_entry.json with", len(parents), "parents.")
EOF

date

# Fix formatting error
jq '.parents = (.parents | map({did: .}))' child_of_100000_parents_entry.json > child_of_100000_parents_entry_fixed.json

echo "Finis!"

date

