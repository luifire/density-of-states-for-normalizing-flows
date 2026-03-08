echo ==========================
echo Run Unit Test

source activate training

mkdir /dummy

export data_path=/master-thesis-work/datasets/
export dummy_path=/dummy

export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"


cd /master-thesis-work/src/unit_test
python -m unittest test_training | tee big_unit_test_log.txt

echo Unit Test ended
echo ==========================
