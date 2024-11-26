import io
import numpy as np
from PIL import Image
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from os.path import isfile, join
from os import listdir
import json


parser = argparse.ArgumentParser(description='Download all stimuli from S3.')
parser.add_argument('--scenario', type=str, default='Dominoes', help='name of the scenarios')
args = parser.parse_args()
scenario = args.scenario
print(scenario)

source_path = '/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/test_humans_consolidated/lf_0/'
save_path = '/ccn2/u/haw027/b3d_ipe/num_obj'
scenario_path = join(source_path, scenario+'_all_movies')
onlyhdf5 = [f for f in listdir(scenario_path) if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith('.hdf5')]

all_data = {}
fig = plt.figure()
for hdf5_file in onlyhdf5:
    trial_name = hdf5_file[:-5]
    print('\t', trial_name)
    hdf5_file_path = join(scenario_path, hdf5_file)
    num_objs = []
    with h5py.File(hdf5_file_path, "r") as f:
        im_seg = np.array(Image.open(io.BytesIO(f['frames']['0000']['images']['_id_cam0'][:])))
        counter = np.unique(im_seg.reshape(-1, im_seg.shape[2]), axis=0)
        init_num = counter.shape[0]-1
        for key in f['frames'].keys():
            im_seg = np.array(Image.open(io.BytesIO(f['frames'][key]['images']['_id_cam0'][:])))
            counter = np.unique(im_seg.reshape(-1, im_seg.shape[2]), axis=0)
            num = counter.shape[0]-1-init_num
            num_objs.append(num)
    # plt.plot(range(len(num_objs)), num_objs)
    all_data[trial_name] = num_objs

plt.xlabel("frame")
plt.ylabel("num obj")
plt.title("num obj over time")
fig.savefig(join(save_path, f'{scenario}.png'))
with open(f'/ccn2/u/haw027/b3d_ipe/num_obj/{scenario}.json', "w") as f:
    json.dump(all_data, f)