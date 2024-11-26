from PIL import Image
from os import listdir
import h5py
import io
import numpy as np
from os.path import isfile, join
from copy import deepcopy
from functools import reduce
import argparse
import json


def color_image(depth_map, area, color):
    _depth_map = deepcopy(depth_map)
    _depth_map[area] = color
    return _depth_map
        
def get_mask_area(seg_img, colors):
    arrs = []
    for color in colors:
        arr = seg_img == color
        arr = arr.min(-1).astype('float32')
        arr = arr.reshape((arr.shape[-1], arr.shape[-1])).astype(bool)
        arrs.append(arr)
    return reduce(np.logical_or, arrs)
        
parser = argparse.ArgumentParser(description='Download all stimuli from S3.')
parser.add_argument('--scenario', type=str, default='Dominoes', help='name of the scenarios')
args = parser.parse_args()
scenario = args.scenario
print(scenario)
other_scenarios = ['collide', 'contain', 'dominoes', 'drop', 'roll', 'support']

source_path = '/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/test_humans_consolidated/lf_0/'
save_path = '/ccn2/u/haw027/b3d_ipe/unified_seg_masks'

bad_list = []

scenario_path = join(source_path, scenario+'_all_movies')
onlyhdf5 = [f for f in listdir(scenario_path) if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith('.hdf5')]
for hdf5_file in onlyhdf5:
    trial_name = hdf5_file[:-5]
    print('\t', trial_name)
    hdf5_file_path = join(scenario_path, hdf5_file)
    base_id, base_type, attachment_id, attachent_type, attachment_fixed, use_attachment, use_base, use_cap, cap_id = None, None, None, None, None, None, None, None, None
    interesting_ids = []
    with h5py.File(hdf5_file_path, "r") as f:
        im_seg_0 = np.array(Image.open(io.BytesIO(f['frames']['0000']['images']['_id_cam0'][:])))
        im_seg_1 = np.array(Image.open(io.BytesIO(f['frames']['0001']['images']['_id_cam0'][:])))

        # extract object info
        object_ids = np.array(f['static']['object_ids'])
        model_names = np.array(f['static']['model_names'])
        assert len(object_ids) == len(model_names)
        distractors = np.array(f['static']['distractors']) if np.array(f['static']['distractors']).size != 0 else None
        occluders = np.array(f['static']['occluders']) if np.array(f['static']['occluders']).size != 0 else None
        if scenario == 'link':
            assert distractors==None and occluders==None
        
        object_segmentation_colors = np.array(f['static']['object_segmentation_colors'])
        if "base_id" in np.array(f['static']) and "attachment_id" in np.array(f['static']):
            base_id = np.array(f['static']['base_id'])
            base_type = np.array(f['static']['base_type'])
            attachment_id = np.array(f['static']['attachment_id'])
            attachent_type = np.array(f['static']['attachent_type'])
            attachment_fixed = np.array(f['static']['attachment_fixed'])
            use_attachment = np.array(f['static']['use_attachment'])
            link_type = np.array(f['static']['link_type'])
            use_base = np.array(f['static']['use_base'])
            use_cap = np.array(f['static']['use_cap'])
            assert attachment_id.size==1
            assert base_id.size==1
            attachment_id = attachment_id.item()
            base_id = base_id.item()
            interesting_ids.append(base_id)
            interesting_ids.append(attachment_id)
            if use_cap:
                cap_id = attachment_id+1
                interesting_ids.append(cap_id)

    if scenario in other_scenarios:
        assert len(interesting_ids)==0 or attachment_fixed==False
        if len(interesting_ids)!=0 and attachment_fixed==False:
            counter_0 = np.unique(im_seg_0.reshape(-1, im_seg_0.shape[2]), axis=0)
            num_obj_0 = counter_0.shape[0]-1
            counter_1 = np.unique(im_seg_1.reshape(-1, im_seg_1.shape[2]), axis=0)
            num_obj_1 = counter_1.shape[0]-1
            assert num_obj_0 == num_obj_1
        continue

    flag = False
    invalid_ids = []
    for o_id in interesting_ids:
        try:
            object_ids.tolist().index(o_id)
        except:
            bad_list.append(trial_name)
            invalid_ids.append(o_id)
            flag = True

    for invalid_id in invalid_ids:
        interesting_ids.remove(invalid_id)

    if flag:
        im_seg_copy = deepcopy(im_seg_1)
    else:
        im_seg_copy = deepcopy(im_seg_0)

    im_seg_copy_copy = deepcopy(im_seg_copy)

    for i, o_id in enumerate(interesting_ids):
        color = object_segmentation_colors[object_ids.tolist().index(o_id)]
        
        area = get_mask_area(im_seg_copy_copy, [color])
        if i == 0:
            consistent_color = color
        else:
            im_seg_copy = color_image(im_seg_copy, area, consistent_color)
    
    
    counter = np.unique(im_seg_copy.reshape(-1, im_seg_copy.shape[2]), axis=0)
    num_obj = counter.shape[0]-1
    assert num_obj == len(object_ids)-len(interesting_ids)+1
    im = Image.fromarray(im_seg_copy)
    im.save(f"{save_path}/{trial_name}_seg.png")

bad_list = list(set(bad_list))
with open(f'{save_path}/bad_list_{scenario}.json', 'w') as f:
    json.dump(bad_list, f)