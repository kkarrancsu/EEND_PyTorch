# Modify this file according to a job scheduling system in your cluster.
# For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
#
# If you use your local machine, use "run.pl".
# export train_cmd="run.pl"
# export infer_cmd="run.pl"
# export simu_cmd="run.pl"

# If you use Grid Engine, use "queue.pl"
#export cmd="utils/queue.pl --mem 32G -l h_rt=200:00:00"  # L166 of steps/segmentation/detect_speech_activity.sh" uses "cmd" variable
#export train_cmd="qsub -l gpu=4,num_proc=20,mem_free=80G,h_rt=400:00:00 -q gpu.q"  
export train_cmd="utils/queue.pl --mem 80G -q gpu.q -l gpu=4 -l h_rt=500:00:00,num_proc=20"
export infer_cmd="utils/queue.pl --mem 32G"
export simu_cmd="utils/queue.pl --mem 4G -l h_rt=48:00:00"  # simulation creation takes a long time!

# If you use SLURM, use "slurm.pl".
# export train_cmd="slurm.pl"
# export infer_cmd="slurm.pl"
# export simu_cmd="slurm.pl"
