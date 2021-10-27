#!/bin/bash
set +x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
exp_root=${1:-"/expscratch/kkarra/diarization/callhome"}
export LC_ALL=C

# TODO: change to the conda environment name!
source activate ovad

# derived
eend_code_root=${SCRIPT_DIR}/../../../

if [[ -z "${KALDI_ROOT}" ]]; then
    echo "KALDI_ROOT undefined!"
    exit 1;
fi

. ./cmd.sh
. ./path.sh
module load ffmpeg

model_dir=${exp_root}/models

# Model averaging options
average_start=400
average_end=500

ave_id=avg${average_start}-${average_end}
echo "averaging model parameters into $model_dir/$ave_id.nnet.npz"
if [ -s $model_dir/$ave_id.nnet.npz ]; then
    echo "$model_dir/$ave_id.nnet.npz already exists. "
    echo " if you want to retry, please remove it."
    exit 1
fi
python $eend_code_root/eend/bin/model_averaging.py $model_dir/$ave_id.nnet.npz $model_dir $average_start $average_end || exit 1
