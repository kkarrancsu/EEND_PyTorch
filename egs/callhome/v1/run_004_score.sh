#!/bin/bash
set +x

stage=1

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

train_set=$exp_root/mixture_sim/data/swb_sre_tr_ns2_beta2_100000
valid_set=$exp_root/mixture_sim/data/swb_sre_cv_ns2_beta2_500
train_id=$(basename $train_set)
valid_id=$(basename $valid_set)

train_config=conf/train_base.yaml
infer_config=conf/infer_base.yaml
infer_args=

eval `$eend_code_root/eend/bin/yaml2bash.py --prefix infer $train_config`
eval `$eend_code_root/eend/bin/yaml2bash.py --prefix infer $infer_config`

if [ $infer_gpu -ne 0 ]; then
    infer_cmd+=" --gpu 1"
fi

infer_config_id=$(echo $infer_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')

model_id=$train_id.$valid_id.$train_config_id
model_dir=$exp_root/models/
# from 003 --> make a common config
average_start=400
average_end=500
ave_id=avg${average_start}-${average_end}

infer_dir=$exp_root/exp/diarize/infer/$model_id.$ave_id.$infer_config_id
if [ $stage -le 1 ]; then
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
        echo " if you want to retry, please remove it."
    else
    for dset in callhome2_spk2; do
        work=$infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            $eend_code_root/eend/bin/infer.py \
            -c $infer_config \
            $infer_args \
            $exp_root/data/eval/$dset \
            $model_dir/$ave_id.nnet.npz \
            $infer_dir/$dset \
            || exit 1
    done
    fi
fi

scoring_dir=$exp_root/exp/diarize/scoring/$model_id.$ave_id.$infer_config_id
if [ $stage -le 2 ]; then
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
        echo " if you want to retry, please remove it."
    else
    for dset in callhome2_spk2; do
        work=$scoring_dir/$dset/.work
        mkdir -p $work
        find $infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        $eend_code_root/eend/bin/make_rttm.py --median=$med --threshold=$th \
            --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
            $work/file_list_$dset $scoring_dir/$dset/hyp_${th}_$med.rttm
        $KALDI_ROOT/tools/sctk-2.4.10/src/md-eval/md-eval.pl -c 0.25 \
            -r $exp_root/data/eval/$dset/rttm \
            -s $scoring_dir/$dset/hyp_${th}_$med.rttm > $scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
    done
    fi
fi

