#!/bin/bash
scenarios="collide"
# scenarios="contain drop dominoes link roll support"
for val in $scenarios; do
    tmux kill-session -t ${val}
    tmux new-session -s ${val} -d 
    tmux send-keys -t ${val} "source /ccn2/u/haw027/miniconda3/etc/profile.d/conda.sh" Enter 
    tmux send-keys -t ${val} "conda activate uncos" Enter 
    # tmux send-keys -t ${val} "python demo.py -p ${val}" Enter 
    tmux send-keys -t ${val} "CUDA_VISIBLE_DEVICES=0 python /home/haw027/code/uncos/scripts/test_copy.py --scenario ${val}" Enter 
    echo $val
done
