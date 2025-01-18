#!/bin/bash
scenarios="collide contain drop dominoes link roll support"
for val in $scenarios; do
    tmux kill-session -t ${val}
    tmux new-session -s ${val} -d
    # tmux send-keys -t ${val} "source /mnt/fs0/haw027/conda/etc/profile.d/conda.sh" Enter 
    tmux send-keys -t ${val} "source /ccn2/u/haw027/miniconda3/etc/profile.d/conda.sh" Enter 
    tmux send-keys -t ${val} "conda activate physion" Enter 
    # tmux send-keys -t ${val} "conda activate foundationpose" Enter 
    tmux send-keys -t ${val} "cd ~/code/ADEPT-Model/dataset/makers/" Enter 
    tmux send-keys -t ${val} "python test_single.py --scenario ${val}" Enter 
    echo $val
done


# #!/bin/bash
# scenarios="collide contain drop dominoes link roll support"
# for val in $scenarios; do
#     tmux kill-session -t ${val}
#     tmux new-session -s ${val} -d
#     # tmux send-keys -t ${val} "source /mnt/fs0/haw027/conda/etc/profile.d/conda.sh" Enter 
#     tmux send-keys -t ${val} "source /ccn2/u/haw027/miniconda3/etc/profile.d/conda.sh" Enter 
#     # tmux send-keys -t ${val} "conda activate physion" Enter 
#     tmux send-keys -t ${val} "conda activate foundationpose" Enter 
#     tmux send-keys -t ${val} "cd ~/code/ADEPT-Model/dataset/makers/" Enter 
#     tmux send-keys -t ${val} "python find_num_obj.py --scenario ${val}" Enter 
#     echo $val
# done


# #!/bin/bash
# scenarios="collide contain drop dominoes link roll support"
# for val in $scenarios; do
#     tmux kill-session -t ${val}_2
#     tmux new-session -s ${val}_2 -d
#     tmux send-keys -t ${val}_2 "source /ccn2/u/haw027/miniconda3/etc/profile.d/conda.sh" Enter 
#     tmux send-keys -t ${val}_2 "conda activate foundationpose" Enter 
#     tmux send-keys -t ${val}_2 "cd /home/haw027/code/clip" Enter 
#     tmux send-keys -t ${val}_2 "python kalman.py --scenario ${val}" Enter 
#     echo $val
# done

# #!/bin/bash
# scenarios="collide contain drop dominoes link roll support"
# for val in $scenarios; do
#     tmux kill-session -t ${val}_1
#     tmux new-session -s ${val}_1 -d
#     tmux send-keys -t ${val}_1 "source /ccn2/u/haw027/miniconda3/etc/profile.d/conda.sh" Enter 
#     tmux send-keys -t ${val}_1 "conda activate foundationpose" Enter 
#     tmux send-keys -t ${val}_1 "cd /home/haw027/code/clip" Enter 
#     tmux send-keys -t ${val}_1 "python kalman.py --scenario ${val}" Enter 
#     echo $val
# done