#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
# averaging serialized models

import torch
from collections import OrderedDict
import argparse

import glob
import os
import re

def average_model(idir, avg_start, avg_stop, ofile):
    omodel = OrderedDict()

    model_flist = glob.glob(os.path.join(idir, '*.th'))
    
    for ifile in model_flist:
        fname_no_path=os.path.basename(ifile)
        saved_epoch=re.findall(r'\d+', fname_no_path)
        assert len(saved_epoch)==1, "Unable to parse epoch from saved models, please rename files to <name>_<digit>.th"
        saved_epoch_int = int(saved_epoch[0])
        if saved_epoch_int >= avg_start and saved_epoch_int <= avg_stop:
            tmpmodel = torch.load(ifile)
            for k, v in tmpmodel.items():
                omodel[k] = omodel.get(k, 0) + v

    for k, v in omodel.items():
        omodel[k] = v / len(model_flist)

    torch.save(omodel, ofile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ofile")
    parser.add_argument("idir")
    parser.add_argument('avg_start', type=int)
    parser.add_argument('avg_stop', type=int)
    args = parser.parse_args()

    print(str(args))
    average_model(args.idir, args.avg_start, args.avg_stop, args.ofile)
    print("Finished averaging")

