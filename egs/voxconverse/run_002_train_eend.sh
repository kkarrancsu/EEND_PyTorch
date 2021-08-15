#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
exp_root=${1:-"/exp/kkarra/diarization/voxconverse"}

eend_code_root=${SCRIPT_DIR}/../../

conf_dir=${eend_code_root}/conf/voxconverse/base
train_dir=${exp_root}/mixture_sim/data/train_segments_ns2_beta2_100000
dev_dir=${exp_root}/mixture_sim/data/dev_segments_ns2_beta2_1000
model_dir=${exp_root}/models
train_conf=${conf_dir}/train.yaml

module load ffmpeg
source activate ovad
pushd . 
cd ${eend_code_root}
~/.conda/envs/ovad/bin/python eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir
popd
