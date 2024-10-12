
#!/bin/bash
source path.sh
set -e

# log_root="logs"
log_root="commit1.0_entropy0.4"
# .lst save the wav path.
input_training_file="/mnt/lynx4/users/cjs/dataset/LibriTTS_16khz/labels/train.lst"
input_validation_file="/mnt/lynx4/users/cjs/dataset/LibriTTS_16khz/labels/test-clean.lst"

# mode=debug
mode=train

if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  log_root=${log_root}_debug
  export CUDA_VISIBLE_DEVICES=0
  python ${BIN_DIR}/train.py \
    --config config_16k_320d.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 100 \
    --summary_interval 10 \
    --validation_interval 100 \

elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export OMP_NUM_THREADS=1
  python ${BIN_DIR}/train.py \
    --config config_16k_320d.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 5000 \
    --summary_interval 100 \
    --validation_interval 5000
fi
