echo ==========================
echo Run Bug Unit Test

source activate training

mkdir /dummy

export data_path=/master-thesis-work/datasets/
export dummy_path=/dummy
export training_class=$1

export PYTHONPATH="${PYTHONPATH}:/master-thesis-work/src/models/:/master-thesis-work/src/common"

cd /master-thesis-work/src/unit_test
python -m unittest test_training test_model | tee big_unit_test_log.txt
#python -m unittest test_model.TestModels.test_survae_big_model

echo Big Unit Test ended
echo ==========================
