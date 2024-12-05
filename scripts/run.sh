#!/bin/bash
# scenarios="collide contain drop dominoes link roll support"
# scenarios="test"
scenarios="problematic"
for val in $scenarios; do
    tmux kill-session -t ${val}
    tmux new-session -s ${val} -d
    tmux send-keys -t ${val} "conda activate uncos" Enter 
    tmux send-keys -t ${val} "export LD_LIBRARY_PATH=/opt/conda/envs/uncos/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH" Enter 
    tmux send-keys -t ${val} "python test.py -p ${val}" Enter 
    # tmux send-keys -t ${val} "python extract.py --scenario ${val}" Enter 
    echo $val
done
