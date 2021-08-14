#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
exp_root=${1:-"/exp/kkarra/diarization/voxconverse"}
dataprep_root=${2:-"/home/hltcoe/kkarra/dataprep/voxconverse"}
audio_root=/export/common/data/corpora/voxconverse/audio
nj=40
voxceleb1_root=/export/common/data/corpora/voxceleb1
voxceleb2_root=/export/common/data/corpora/voxceleb2
stage=0 

if [[ -z "${KALDI_ROOT}" ]]; then
    echo "KALDI_ROOT undefined!"
    exit 1;
fi
mkdir -p $exp_root

# create training dataset
if [ $stage -le 0 ]; then
    module load ffmpeg
    pushd .
    cd $KALDI_ROOT/egs/voxceleb/v1
    # from: https://github.com/kaldi-asr/kaldi/blob/master/egs/voxceleb/v2/run.sh
    . ./cmd.sh
    # use voxceleb data as training data
    local/make_voxceleb2.pl $voxceleb2_root dev $exp_root/data/voxceleb2_train
    local/make_voxceleb2.pl $voxceleb2_root "test" $exp_root/data/voxceleb2_test
    # This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
    # Our evaluation set is the test portion of VoxCeleb1.
    local/make_voxceleb1_v2.pl $voxceleb1_root dev $exp_root/data/voxceleb1_train
    local/make_voxceleb1_v2.pl $voxceleb1_root "test" $exp_root/dev  
    ## We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
    ## This should give 7,323 speakers and 1,276,888 utterances.
    utils/combine_data.sh $exp_root/train/ $exp_root/data/voxceleb2_train \
        $exp_root/data/voxceleb2_test $exp_root/data/voxceleb1_train

    # Make MFCCs and compute the energy-based VAD for each dataset
    for name in train dev; do
        steps/make_mfcc.sh --write-utt2num-frames true \
          --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
          ${exp_root}/${name} ${exp_root}/log/make_mfcc ${exp_root}/${name}/data/mfcc
        utils/fix_data_dir.sh ${exp_root}/${name}
        sid/compute_vad_decision.sh --nj 80 --cmd "$train_cmd" \
          ${exp_root}/${name} ${exp_root}/log/make_vad ${exp_root}/${name}/data/vad
        utils/fix_data_dir.sh ${exp_root}/${name}
    done
    popd
fi

# create kaldi-data-dir for voxconverse for inference
if [ $stage -le 1 ]; then
    # run the dataprep script from dataprep repo, to generate wav.scp etc
    pushd .
    cd $dataprep_root
    voxconverse_inference_dir=$exp_root/vc_infer
    make_kaldi_dir.sh $audio_root $voxconverse_inference_dir
    popd
    
    pushd .
    wsj_dir=${KALDI_ROOT}/egs/wsj/s5/
    cd $wsj_dir
    . ./cmd.sh

    # only if vad.scp doesn't exist
    if [ ! -f $data_root/vad.scp ]; then
        steps/make_mfcc.sh --nj $nj --mfcc-config ${SCRIPT_DIR}/conf/mfcc.conf --cmd "$train_cmd" $voxconverse_inference_dir 
        steps/compute_vad_decision.sh --nj $nj --vad-config ${SCRIPT_DIR}/conf/vad.conf --cmd "$train_cmd" $voxconverse_inference_dir
        utils/data/get_reco2dur.sh $voxconverse_inference_dir
    fi

    # TODO: run rvad also
    rvad_code_dir=/home/hltcoe/kkarra/rVAD/rVADfast_py_2.0/

    popd
fi
