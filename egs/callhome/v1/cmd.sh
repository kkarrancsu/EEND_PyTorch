# Modify this file according to a job scheduling system in your cluster.
# For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
#
# If you use your local machine, use "run.pl".
# export train_cmd="run.pl"
# export infer_cmd="run.pl"
# export simu_cmd="run.pl"

# If you use Grid Engine, use "queue.pl"
#export cmd="utils/queue.pl --mem 32G -l h_rt=200:00:00"  # L166 of steps/segmentation/detect_speech_activity.sh" uses "cmd" variable
export train_cmd="utils/queue.pl --mem 32G -l 'hostname=c*'"  # TODO: update this to a GPU node
export infer_cmd="utils/queue.pl --mem 32G -l 'hostname=c*'"  # TODO: update this to a GPU node
export simu_cmd="utils/queue.pl --mem 4G"

# If you use SLURM, use "slurm.pl".
# export train_cmd="slurm.pl"
# export infer_cmd="slurm.pl"
# export simu_cmd="slurm.pl"
