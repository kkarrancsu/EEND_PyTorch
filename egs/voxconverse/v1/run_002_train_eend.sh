#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
exp_root=${1:-"/expscratch/kkarra/diarization/voxconverse"}

eend_code_root=${SCRIPT_DIR}/../../../

conf_dir=${SCRIPT_DIR}/conf/
train_dir=${exp_root}/mixture_sim/data/train_segments_ns2_beta2_100000
dev_dir=${exp_root}/mixture_sim/data/dev_segments_ns2_beta2_1000
model_dir=${exp_root}/models
train_conf=${conf_dir}/train_base.yaml

module load ffmpeg
source activate ovad
pushd . 
cd ${eend_code_root}

num_gpus="auto-detect"   # can be an integer, or auto-detect to use all gpus available
# the --gpu auto-detect option overrides configuration in the yaml file
~/.conda/envs/ovad/bin/python eend/bin/train.py -c $train_conf --gpu $num_gpus --num-workers 8 \
    $train_dir $dev_dir $model_dir

popd
