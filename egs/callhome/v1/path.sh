# KALDI_ROOT is checked for in the main runner script (run_001_prepdata.sh)

export PATH=$PATH:$KALDI_ROOT/src/featbin:$KALDI_ROOT/src/nnet3bin/:$KALDI_ROOT/src/bin/:$KALDI_ROOT/tools/openfst/bin/
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
