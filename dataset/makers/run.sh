#!/bin/bash
scenarios="collide contain drop dominoes link roll support"
for val in $scenarios; do
    tmux kill-session -t ${val}
    tmux new-session -s ${val} -d
    tmux send-keys -t ${val} "source /ccn2/u/haw027/miniconda3/etc/profile.d/conda.sh" Enter 
    tmux send-keys -t ${val} "conda activate physion" Enter 
    tmux send-keys -t ${val} "cd ~/code/ADEPT-Model/dataset/makers/" Enter 
    tmux send-keys -t ${val} "python test.py --scenario ${val}" Enter 
    echo $val
done
