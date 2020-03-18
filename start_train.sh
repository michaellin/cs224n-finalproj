export PYTHONPATH=`pwd`
python training_ptr_gen/train.py 2>&1 | tee log/training_log

