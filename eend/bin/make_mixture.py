#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This script generates simulated multi-talker mixtures for diarization
#
# common/make_mixture.py \
#     mixture.scp \
#     data/mixture \
#     wav/mixture


import argparse
import os
from eend import kaldi_data
import numpy as np
import math
import soundfile as sf
import json

#### FOR DEBUGGING ONLY ####
DISABLE_AUDIO_GEN=False # set to False for normal operation!
############################

def gen_kaldi_dataset(script, out_data_dir, out_wav_dir, sample_rate=16000):
    # open output data files
    segments_fp = os.path.join(out_data_dir, 'segments')
    utt2spk_fp = os.path.join(out_data_dir, 'utt2spk')
    wav_scp_fp = os.path.join(out_data_dir, 'wav.scp')
    script_lines = open(script, 'r').readlines()

    try:
        os.makedirs(out_data_dir)
    except IOError:
        pass

    # "-R" forces the default random seed for reproducibility
    resample_cmd = "sox -q -V0 -R -t wav - -t wav - rate {}".format(sample_rate)

    global_ctr=0
    with open(segments_fp, 'w', buffering=1) as segments_f, open(utt2spk_fp, 'w', buffering=1) as utt2spk_f, open(wav_scp_fp, 'w', buffering=1) as wav_scp_f:
        for line_num, line in enumerate(script_lines):
            recid, jsonstr = line.strip().split(None, 1)
            indata = json.loads(jsonstr)
            wavfn = indata['recid']
            ########
            # recid now include out_wav_dir
            #recid = os.path.join(args.out_wav_dir, wavfn).replace('/','_')
            ########
            recid = wavfn.replace('/', '_')
            global_ctr += 1
            noise = indata['noise']
            noise_snr = indata['snr']
            mixture = []
            for speaker in indata['speakers']:
                spkid = speaker['spkid']
                utts = speaker['utts']
                intervals = speaker['intervals']
                rir = speaker['rir']
                data = []
                pos = 0
                for interval, utt in zip(intervals, utts):
                    if DISABLE_AUDIO_GEN:
                        speech = np.zeros(5)
                        silence = np.zeros(int(interval * sample_rate)) 
                    else:
                        # append silence interval data
                        silence = np.zeros(int(interval * sample_rate))
                        data.append(silence)
                        # utterance is reverberated using room impulse response
                        preprocess = "wav-reverberate --print-args=false " \
                                     " --impulse-response={} - -".format(rir)
                        if isinstance(utt, list):
                            rec, st, et = utt
                            st = np.rint(st * sample_rate).astype(int)
                            et = np.rint(et * sample_rate).astype(int)
                        else:
                            rec = utt
                            st = 0
                            et = None
                        if rir is not None:
                            wav_rxfilename = kaldi_data.process_wav(rec, preprocess)
                        else:
                            wav_rxfilename = rec
                        wav_rxfilename = kaldi_data.process_wav(
                                wav_rxfilename, resample_cmd)
                        speech, _ = kaldi_data.load_wav(wav_rxfilename, st, et)
                        data.append(speech)
                    # calculate start/end position in samples
                    startpos = pos + len(silence)
                    endpos = startpos + len(speech)
                    # write segments and utt2spk
                    uttid = '{}-{}_{:07d}_{:07d}'.format(
                            spkid, recid, int(startpos / sample_rate * 100),
                            int(endpos / sample_rate * 100))
                    segments_f.write('%s %s %0.07d %0.07d\n' % (uttid, recid, startpos / sample_rate, endpos / sample_rate))
                    utt2spk_f.write('%s %s\n' % (uttid, spkid))
                    # update position for next utterance
                    pos = endpos

                if not DISABLE_AUDIO_GEN:
                    data = np.concatenate(data)
                    mixture.append(data)

            outfname = '{}.wav'.format(wavfn)
            outpath = os.path.join(out_wav_dir, outfname)
 
            if not DISABLE_AUDIO_GEN:
                # fitting to the maximum-length speaker data, then mix all speakers
                maxlen = max(len(x) for x in mixture)
                mixture = [np.pad(x, (0, maxlen - len(x)), 'constant') for x in mixture]
                mixture = np.sum(mixture, axis=0)
                # noise is repeated or cutted for fitting to the mixture data length
                noise_resampled = kaldi_data.process_wav(noise, resample_cmd)
                noise_data, _ = kaldi_data.load_wav(noise_resampled)
                if maxlen > len(noise_data):
                    noise_data = np.pad(noise_data, (0, maxlen - len(noise_data)), 'wrap')
                else:
                    noise_data = noise_data[:maxlen]
                # noise power is scaled according to selected SNR, then mixed
                signal_power = np.sum(mixture**2) / len(mixture)
                noise_power = np.sum(noise_data**2) / len(noise_data)
                scale = math.sqrt(
                            math.pow(10, - noise_snr / 10) * signal_power / noise_power)
                mixture += noise_data * scale
                # output the wav file and write wav.scp
                sf.write(outpath, mixture, sample_rate)
            wav_scp_f.write('%s %s\n' % (recid, os.path.abspath(outpath)))
            print('Progress: %d/%d' % (line_num+1, len(script_lines)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('script',
                        help='list of json')
    parser.add_argument('out_data_dir',
                        help='output data dir of mixture')
    parser.add_argument('out_wav_dir',
                        help='output mixture wav files are stored here')
    parser.add_argument('--rate', type=int, default=16000,
                        help='sampling rate')
    args = parser.parse_args()
    gen_kaldi_dataset(args.script, args.out_data_dir, args.out_wav_dir, args.rate)


