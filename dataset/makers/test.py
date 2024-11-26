from PIL import Image
from os import listdir
import h5py
import io
import os
import argparse
import numpy as np
from os.path import isfile, join
import pycocotools.mask as mask_util
import json
import yaml
from glob import glob


def makedir(path):
    if not os.path.exists(path):
        # print(f"making dir {path}")
        os.makedirs(path)

def get_mask_area(color, seg_img):
    arr = seg_img == color
    arr = arr.min(-1).astype('float')
    arr = arr.reshape((arr.shape[-1], arr.shape[-1]))
    return np.asfortranarray(arr.astype(bool))

def write_serialized(var, file_name):
    """Write json and yaml file"""
    assert file_name is not None
    with open(file_name, "w") as f:
        if file_name.endswith(".json"):
            json.dump(var, f, indent=4)
        elif file_name.endswith(".yaml"):
            yaml.safe_dump(var, f, indent=4)
        else:
            raise FileNotFoundError
        
parser = argparse.ArgumentParser(description='Download all stimuli from S3.')
parser.add_argument('--scenario', type=str, default='Dominoes', help='name of the scenarios')
args = parser.parse_args()
scenario = args.scenario
print(scenario)

# source_path = '/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/test_humans_consolidated/lf_0'
source_path = '/mnt/fs5/rahul/lf_0/'
# save_path = '/ccn2/u/haw027/b3d_ipe/train/'
save_path = '/mnt/fs0/haw027/b3d_ipe/train/'

width = 350
height = 350

file = {}
scenario_path = join(source_path, scenario+'_all_movies')
# onlyhdf5 = [f for f in listdir(scenario_path) if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith('.hdf5')]
onlyhdf5 = [y for x in os.walk(scenario_path) for y in glob(os.path.join(x[0], '*.hdf5'))]
for hdf5_file in onlyhdf5:
    trial_name = '_'.join(hdf5_file.split('/')[-2:])[:-5]
    if trial_name.endswith('temp'):
        continue
    makedir(join(save_path, trial_name, "imgs"))
    file[trial_name] = {}
    file[trial_name]['scene'] = []
    print('\t', trial_name)
    hdf5_file_path = join(scenario_path, hdf5_file)
    with h5py.File(hdf5_file_path, "r") as f:
        object_ids = np.array(f['static']['object_ids'])
        model_names = np.array(f['static']['model_names'])
        distractors = np.array(f['static']['distractors']) if np.array(f['static']['distractors']).size != 0 else None
        occluders = np.array(f['static']['occluders']) if np.array(f['static']['occluders']).size != 0 else None
        distractor_ids = np.concatenate([np.where(model_names==distractor)[0] for distractor in distractors], axis=0).tolist() if distractors else []
        occluder_ids = np.concatenate([np.where(model_names==occluder)[0] for occluder in occluders], axis=0).tolist() if occluders else []
        excluded_model_ids = distractor_ids+occluder_ids
        included_model_ids = [idx for idx in range(len(object_ids)) if idx not in excluded_model_ids]

        object_segmentation_colors = np.array(f['static']['object_segmentation_colors'])
        scales = np.array(f['static']['scale'])
        for key in f['frames'].keys():
            if int(key)>10:
                continue
            im = np.array(Image.open(io.BytesIO(f['frames'][key]['images']['_img_cam0'][:])))
            im = Image.fromarray(im)
            downsampled_im = im.resize((width, height), Image.BICUBIC)
            downsampled_im.save(join(save_path, trial_name, "imgs", f"{key}.png"))

            locations = np.array(f['frames'][key]['objects']['positions_cam0'])
            rotations = np.array(f['frames'][key]['objects']['rotations_cam0'])
            im_seg = np.array(Image.open(io.BytesIO(f['frames'][key]['images']['_id_cam0'][:])).resize((width, height), Image.BICUBIC))

            this_frame = {}
            this_frame['objects'] = []
            for i in included_model_ids:
                obj_info = {}
                obj_info['type'] = model_names[i].decode('UTF-8')
                obj_info['location'] = locations[i].tolist()
                obj_info['rotation'] = rotations[i].tolist()
                obj_info['scale'] = scales[i].tolist()
                obj_mask = mask_util.encode(get_mask_area(im_seg, object_segmentation_colors[i]))
                obj_info['mask'] = {'size': obj_mask['size'], 'counts': obj_mask['counts'].decode('UTF-8')}
                this_frame['objects'].append(obj_info)
            file[trial_name]['scene'].append(this_frame)

write_serialized(file, join('/'.join(save_path.split('/')[:-2]), f'annotated_ann_{scenario}.json'))