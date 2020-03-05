export PYTHONPATH=`pwd`
MODEL_PATH=$1
MODEL=$(basename $MODEL_PATH)
python training_ptr_gen/decode.py $MODEL_PATH 2>&1 | tee "${MODEL}_decode_log.txt"

