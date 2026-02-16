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

# Declare child to MetaCat
metacat file declare -f child_of_100000_parents_entry_fixed.json dtucker_metacat_stress_tests:test_20260211

date

## Run validation
#metacat file show dtucker_metacat_stress_tests:test_20260211_child_of_100000_parents -l |wc
#metacat file show dtucker_metacat_stress_tests:test_20260211_child_of_100000_parents -l | less


echo "Finis!"

date

