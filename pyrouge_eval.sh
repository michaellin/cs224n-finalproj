export PYTHONPATH=`pwd`
MODEL=$1
python training_ptr_gen/rouge_test.py $MODEL
