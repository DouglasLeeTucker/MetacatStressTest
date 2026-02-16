date

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metacat-stress

cd ~/WORK/GitHub/MetacatStressTest/python

./synthetic_generator_minimal.py \
    -o ./synthetic_minimal_n100000 \
    -n 100000 \
    --min-rows 50 \
    --max-rows 200 \
    --metacat-batch dummy.json \
    --namespace dtucker_metacat_stress_tests \
    --owner dtucker \
    --dataset-prefix test_20260211 \
    --num-datasets 1

date

