#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
exp_root=${1:-"/exp/kkarra/diarization/voxconverse"}
dataprep_root=${2:-"/home/hltcoe/kkarra/dataprep/voxconverse"}
audio_root="/export/common/data/corpora/voxconverse/audio"
nj=40
voxceleb1_root="/export/common/data/corpora/voxceleb1"
voxceleb2_root="/export/common/data/corpora/voxceleb2"
musan_root="/exp/kkarra/musan/data/local"  # NOTE: this isn't the full MUSAN, a subset for EEND
                                           # TODO: make sure the repo has this tar data
begin_stage=2
end_stage=2

# simulation options for creating mixture dataset
simu_opts_overlap=yes
simu_opts_num_speaker=2
simu_opts_sil_scale=2
simu_opts_rvb_prob=0.5
simu_opts_num_train=100000  # of mixtures for training
simu_opts_min_utts=10
simu_opts_max_utts=20

# derived
data_augment_root=${exp_root}/aug
eend_code_root=${SCRIPT_DIR}/../../

if [[ -z "${KALDI_ROOT}" ]]; then
    echo "KALDI_ROOT undefined!"
    exit 1;
fi
mkdir -p $exp_root

wsj_dir=${KALDI_ROOT}/egs/wsj/s5/
diar_dir=${KALDI_ROOT}/egs/callhome_diarization/v1
voxv1_dir=${KALDI_ROOT}/egs/voxceleb/v1

# setup sym links for steps & utils w/ wsj 
cd $SCRIPT_DIR
ln -s $wsj_dir/steps steps
ln -s $wsj_dir/utils utils

module load ffmpeg

# create training dataset
if [ $begin_stage -le 0 ] && [ $end_stage -ge 0 ]; then
    pushd .
    cd $voxv1_dir
    # from: https://github.com/kaldi-asr/kaldi/blob/master/egs/voxceleb/v2/run.sh
    . ./cmd.sh

    # use voxceleb data as training data
    #local/make_voxceleb2.pl $voxceleb2_root dev $exp_root/data/voxceleb2_train
    #local/make_voxceleb2.pl $voxceleb2_root "test" $exp_root/data/voxceleb2_test
    # This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
    # Our evaluation set is the test portion of VoxCeleb1.
    local/make_voxceleb1_v2.pl $voxceleb1_root dev $exp_root/data/voxceleb1_train
    local/make_voxceleb1_v2.pl $voxceleb1_root "test" $exp_root/data/voxceleb1_dev
    ## We will train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
    ## This should give 7,323 speakers and 1,276,888 utterances.
    #utils/combine_data.sh $exp_root/train $exp_root/data/voxceleb2_train \
    #    $exp_root/data/voxceleb2_test $exp_root/data/voxceleb1_train
    # WARNING!!: we only use vox1 for training right now to speed up training, we dont want piped commands
    #  as that slows down straining.  once the pipeline is verified, we can try w/ piped data also
    utils/combine_data.sh $exp_root/train $exp_root/data/voxceleb1_train
    utils/combine_data.sh $exp_root/dev $exp_root/data/voxceleb1_dev

    utils/data/get_reco2dur.sh $exp_root/train/
    utils/data/get_reco2dur.sh $exp_root/dev/

    # Make MFCCs and compute the energy-based VAD for each dataset
    for name in train dev; do
        if [ "$name" == "train" ]; then
            nj=80
        else
            nj=40
        fi
        steps/make_mfcc.sh --write-utt2num-frames true \
          --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
          ${exp_root}/${name} ${exp_root}/log/make_mfcc ${exp_root}/${name}/data/mfcc
        utils/fix_data_dir.sh ${exp_root}/${name}
        sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
          ${exp_root}/${name} ${exp_root}/log/make_vad ${exp_root}/${name}/data/vad
        utils/fix_data_dir.sh ${exp_root}/${name}
    done
    popd

    # make segments file, which is needed for training
    pushd .
    cd $diar_dir
    diarization/vad_to_segments.sh $exp_root/train $exp_root/train_segments
    diarization/vad_to_segments.sh $exp_root/dev $exp_root/dev_segments
    popd
fi

if [ $begin_stage -le 1 ] && [ $end_stage -ge 1 ]; then
    pushd .
    cd $wsj_dir

    # musan
    if ! utils/validate_data_dir.sh --no-text --no-feats $data_augment_root/musan; then
        steps/data/make_musan.sh --sampling-rate 16000 $musan_root/ $data_augment_root
        #utils/copy_data_dir.sh data/musan_noise data/musan_noise_bg
        #awk '{if(NR>1) print $1,$1}'  $musan_root/noise/free-sound/ANNOTATIONS > data/musan_noise_bg/utt2spk
        utils/fix_data_dir.sh $data_augment_root/musan
    fi

    # rirs
    if ! utils/validate_data_dir.sh --no-text --no-feats $data_augment_root/data/sim_rirs_16k; then
        if [ ! -e $data_augment_root/sim_rir_16k.zip ]; then
            wget -O $data_augment_root/sim_rir_16k.zip --no-check-certificate http://www.openslr.org/resources/26/sim_rir_16k.zip
        fi
        if [ ! -d $data_augment_root/data/sim_rirs_16k ]; then
            mkdir -p $data_augment_root/data/sim_rirs_16k
            unzip $data_augment_root/sim_rir_16k.zip -d $data_augment_root/data/sim_rirs_16k
        fi
        mkdir -p $data_augment_root/sim_rirs_16k
        find $data_augment_root/data/sim_rirs_16k -iname "*.wav" \
            | awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' \
            | sort > $data_augment_root/sim_rirs_16k/wav.scp
        awk '{print $1, $1}' $data_augment_root/sim_rirs_16k/wav.scp > $data_augment_root/sim_rirs_16k/utt2spk
        utils/fix_data_dir.sh $data_augment_root/sim_rirs_16k
    fi

    popd
fi

if [ $begin_stage -le 2 ] && [ $end_stage -ge 2 ]; then
    . ./cmd.sh
    pushd .
    cd $wsj_dir
    # simulate mixtures w/ the dataset
    # from: https://github.com/hitachi-speech/EEND/blob/master/egs/mini_librispeech/v1/run_prepare_shared.sh#L67
    echo "simulation of mixture"
    simudir=$exp_root/mixture_sim
    simu_actual_dirs=(
        $exp_root/diarization_data
    )
    mkdir -p $exp_root/.work
    random_mixture_cmd=$eend_code_root/eend/bin/random_mixture_nooverlap.py
    make_mixture_cmd=$eend_code_root/eend/bin/make_mixture_nooverlap.py
    if [ "$simu_opts_overlap" == "yes" ]; then
        random_mixture_cmd=$eend_code_root/eend/bin/random_mixture.py
        make_mixture_cmd=$eend_code_root/eend/bin/make_mixture.py
    fi

    for simu_opts_sil_scale in 2; do
        #for dset in train_segments dev_segments; do
        for dset in dev_segments; do
            if [ "$dset" == "train_segments" ]; then
                n_mixtures=$simu_opts_num_train
            else
                n_mixtures=1000
            fi
            simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
            # check if you have the simulation
            if ! utils/validate_data_dir.sh --no-text --no-feats $simudir/data/$simuid; then
                # random mixture generation
                $simu_cmd $simudir/.work/random_mixture_$simuid.log \
                    $random_mixture_cmd --n_speakers $simu_opts_num_speaker --n_mixtures $n_mixtures \
                    --speech_rvb_probability $simu_opts_rvb_prob \
                    --sil_scale $simu_opts_sil_scale \
                    $exp_root/$dset $data_augment_root/musan $data_augment_root/sim_rirs_16k \
                    \> $simudir/.work/mixture_$simuid.scp
                nj=100
                mkdir -p $simudir/wav/$simuid
                # distribute simulated data to $simu_actual_dir
                split_scps=
                for n in $(seq $nj); do
                    split_scps="$split_scps $simudir/.work/mixture_$simuid.$n.scp"
                    mkdir -p $simudir/.work/data_$simuid.$n
                    actual=${simu_actual_dirs[($n-1)%${#simu_actual_dirs[@]}]}/$simudir/wav/$simuid/$n
                    mkdir -p $actual
                    ln -nfs $actual $simudir/wav/$simuid/$n
                done
                utils/split_scp.pl $simudir/.work/mixture_$simuid.scp $split_scps || exit 1

                $simu_cmd --max-jobs-run 32 JOB=1:$nj $simudir/.work/make_mixture_$simuid.JOB.log \
                    $make_mixture_cmd --rate=16000 \
                    $simudir/.work/mixture_$simuid.JOB.scp \
                    $simudir/.work/data_$simuid.JOB $simudir/wav/$simuid/JOB
                utils/combine_data.sh $simudir/data/$simuid $simudir/.work/data_$simuid.*
                steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    $simudir/data/$simuid/utt2spk $simudir/data/$simuid/segments \
                    $simudir/data/$simuid/rttm
                utils/data/get_reco2dur.sh $simudir/data/$simuid
            fi
        done
    done
    popd
fi

# create kaldi-data-dir for voxconverse for inference
if [ $begin_stage -le 3 ] && [ $end_stage -ge 3 ]; then
    # run the dataprep script from dataprep repo, to generate wav.scp etc
    pushd .
    cd $dataprep_root
    voxconverse_inference_dir=$exp_root/vc_infer
    make_kaldi_dir.sh $audio_root $voxconverse_inference_dir
    popd
    
    pushd .
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
