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


buggy_stims = "pilot-containment-cone-plate_0017 \
pilot-containment-cone-plate_0022 \
pilot-containment-cone-plate_0029 \
pilot-containment-cone-plate_0034 \
pilot-containment-multi-bowl_0042 \
pilot-containment-multi-bowl_0048 \
pilot-containment-vase_torus_0031 \
pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0005 \
pilot_it2_collision_non-sphere_box_0002 \
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0004 \
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0007 \
pilot_it2_drop_simple_box_0000 \
pilot_it2_drop_simple_box_0042 \
pilot_it2_drop_simple_tdw_1_dis_1_occ_0003 \
pilot_it2_rollingSliding_simple_collision_box_0008 \
pilot_it2_rollingSliding_simple_collision_box_large_force_0009 \
pilot_it2_rollingSliding_simple_collision_tdw_1_dis_1_occ_0002 \
pilot_it2_rollingSliding_simple_ledge_tdw_1_dis_1_occ_sphere_small_zone_0022 \
pilot_it2_rollingSliding_simple_ramp_box_small_zone_0006 \
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0004 \
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0017 \
pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom1_long_a_0022 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom1_0012 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0006 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0010 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0029 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0036 \
pilot_linking_nl6_aNone_bCone_occ1_dis1_boxroom_0028 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0000 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0002 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0003 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0010 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0013 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0017 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0018 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0032 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0036 \
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0021 \
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0041 \
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0006 \
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0009"

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
    if trial_name in buggy_stims:
        continue
    print('\t', trial_name)
    hdf5_file_path = join(scenario_path, hdf5_file)
    num_objs = []
    with h5py.File(hdf5_file_path, "r") as f:
        object_ids = np.array(f['static']['object_ids'])
        object_segmentation_colors = np.array(f['static']['object_segmentation_colors'])
        model_names = np.array(f["static"]["model_names"])
        distractors = np.array(f['static']['distractors']) if np.array(f['static']['distractors']).size != 0 else None
        occluders = np.array(f['static']['occluders']) if np.array(f['static']['occluders']).size != 0 else None
        distractor_ids = np.concatenate([np.where(model_names==distractor)[0] for distractor in distractors], axis=0).tolist() if distractors else []
        occluder_ids = np.concatenate([np.where(model_names==occluder)[0] for occluder in occluders], axis=0).tolist() if occluders else []
        excluded_model_ids = distractor_ids+occluder_ids
        included_model_ids = [idx for idx in range(len(object_ids)) if idx not in excluded_model_ids]
        object_segmentation_colors = [object_segmentation_colors[o_index].tolist() for o_index in included_model_ids]

        im_seg = np.array(Image.open(io.BytesIO(f['frames']['0000']['images']['_id_cam0'][:])))
        counter = np.unique(im_seg.reshape(-1, im_seg.shape[2]), axis=0)
        relevant_colors = [color.tolist() for color in counter if color.tolist() in object_segmentation_colors]
        init_num = len(relevant_colors)
        for key in f['frames'].keys():
            im_seg = np.array(Image.open(io.BytesIO(f['frames'][key]['images']['_id_cam0'][:])))
            counter = np.unique(im_seg.reshape(-1, im_seg.shape[2]), axis=0)
            relevant_colors = [color.tolist() for color in counter if color.tolist() in object_segmentation_colors]
            num = len(relevant_colors)-init_num
            num_objs.append(num)
    plt.plot(range(len(num_objs)), num_objs)
    all_data[trial_name] = num_objs

plt.xlabel("frame")
plt.ylabel("num obj")
plt.title("num obj over time")
fig.savefig(join(save_path, f'{scenario}.png'))
with open(f'/ccn2/u/haw027/b3d_ipe/num_obj/{scenario}.json', "w") as f:
    json.dump(all_data, f)