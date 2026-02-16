date

# MetaCat setup
source /cvmfs/fermilab.opensciencegrid.org/packages/common/setup-env.sh
spack load r-m-dd-config@1.8 experiment=hypot
htgettoken -i hypot -a htvaultprod.fnal.gov
metacat auth login -m token $USER

# Create new dataset
metacat dataset create dtucker_metacat_stress_tests:test_20260211
metacat dataset list  dtucker_metacat_stress_tests:* -cl

# Change directory to location of the batch JSON file
cd ~/WORK/GitHub/MetacatStressTest/python/synthetic_minimal_n100000

date

# Declare files in JSON batch script to metacat
metacat file declare-many batch_test_20260211_1.json dtucker_metacat_stress_tests:test_20260211

date

# Count how many files were declared to the new dataset
metacat dataset files dtucker_metacat_stress_tests:test_20260211 -m | wc

date
