# MetacatStressTest
Python code for creating a dataset for a stress test of Metacat, etc.

Example runs for synthetic_generator.py:

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
  --duplicate-rate 0.02 \
  --missing-col-rate 0.02 \
  --extra-col-rate 0.02 \
  --wrong-type-rate 0.02 \
  --mixed-type-rate 0.02 \
  --reorder-col-rate 0.02 \
  --empty-table-rate 0.005 \
  --duplicate-col-rate 0.01 \
  --metacat-batch metacat_batch_stress.json

```
