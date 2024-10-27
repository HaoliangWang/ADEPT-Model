import os
import glob
from collections import defaultdict
from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool

import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import default_collate
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as mask_util

from utils.io import read_serialized, write_serialized
from utils.constants import TERMS, TYPES
from utils.misc import to_torch, to_cpu

# def mkdir(path):
#     if not os.path.exists(path):
#         os.mkdir(path)
#     return path

class ObjectProposalDataset(torch.utils.data.dataset.Dataset):
    _split_point = .8

    _types = TYPES
    _terms = TERMS


    def __init__(self, dataset_cfg, args):
        self.root = dataset_cfg.ROOT
        self.split = dataset_cfg.SPLIT
        object_derender_file = read_serialized(dataset_cfg.OBJECT_DERENDER_FILE)

        self.img_files = []
        self.masks = []
        self.attributes = []
        self.basename2objects = defaultdict(list)

        self.case_names = sorted(os.listdir(os.path.join(self.root, 'images')))
        if "case_name" in args:
            self.case_names = [args.case_name]
        elif self.split == "TRAIN":
            self.case_names = self.case_names[:int(self._split_point * len(self.case_names))]
        elif self.split == "VAL":
            self.case_names = self.case_names[int(self._split_point * len(self.case_names)):]

        with Pool(cpu_count()) as p:
            with tqdm(total=len(self.case_names)) as bar:
                results = [p.apply_async(ObjectProposalDataset._prepare_case,
                                         (self.root, c, object_derender_file[c].scene),
                                         callback=lambda *a: bar.update())
                           for c in self.case_names]
                results = [r.get() for r in results]

        for img_files, masks, attributes in results:
            self.img_files.extend(img_files)
            self.masks.extend(masks)
            if self.split != "TEST":
                self.attributes.extend(to_torch(attributes))

    @staticmethod
    def _prepare_case(root, case_name, anns):
        img_files = sorted(glob.glob(os.path.join(root, "images", case_name, "imgs", "*")))
        n_imgs = len(img_files)
        assert n_imgs == len(anns), "{} has {} images but {} annotation".format(case_name, n_imgs, len(anns))
        img_tuple_files = []
        masks = []
        attributes = []
        for i in range(n_imgs):
            for object in anns[i].objects:
                attribute = {}
                for term in ObjectProposalDataset._terms:
                    if term == "type":
                        one_hot = [0] * len(ObjectProposalDataset._types)
                        one_hot[ObjectProposalDataset._types.index(object[term])] = 1
                        attribute[term] = one_hot
                    else:
                        attribute[term] = object[term]
                
                masks.append(object.mask)
                img_tuple_files.append(img_files[i])
                attributes.append(attribute)
        return img_tuple_files, masks, attributes

    def __getitem__(self, index):
        mask_code = self.masks[index]
        mask = torch.Tensor(mask_util.decode(mask_code)).float()
        img = self.img_files[index]
        basename = os.path.basename(img)
        # trial_name = os.path.dirname(img).split('/')[-2]

        img = TF.to_tensor(Image.open(img).convert("RGB"))

        segmented_image = img.clone()
        for i in range(segmented_image.shape[0]):
            segmented_image[i, :, :] *= mask
        img_tuple = torch.cat([segmented_image, img], dim=0)

        if len(self.attributes) > 0:
            attributes = self.attributes[index]
        else:
            attributes = None

        data = {"img_tuple": img_tuple,
                "basename": basename,
                "index": index}
        if attributes is not None:
            data["attributes"] = attributes
        
        # print("trial: ", trial_name, ' ', basename)
        # mkdir(os.path.join('/mnt/fs0/haw027/b3d_ipe/temp', trial_name))
        # save_image(segmented_image, os.path.join('/mnt/fs0/haw027/b3d_ipe/temp', trial_name, f"{basename}_{index}_seg.png"))
        # save_image(img, os.path.join('/mnt/fs0/haw027/b3d_ipe/temp', trial_name, f"{basename}_{index}_img.png"))
        # print("attributes: ", attributes)
        
        return data

    def __len__(self):
        return len(self.masks)

    def process_batch(self, inputs, outputs):
        basenames = inputs["basename"]
        indices = inputs["index"]
        attributes = to_cpu(outputs["output"])
        for i in range(attributes["type"].shape[0]):
            o = {}
            for term in self._terms:
                if term == "type":
                    _, idxs = torch.topk(attributes[term][i], len(ObjectProposalDataset._types))
                    pred_types = []
                    for idx in idxs:
                        pred_types.append(self._types[idx.item()])
                    o[term] = pred_types
                    # o[term] = self._types[torch.argmax(attributes[term][i]).item()]
                else:
                    o[term] = attributes[term][i].tolist()

            o["mask"] = self.masks[indices[i]]
            self.basename2objects[basenames[i]].append(o)

    def _process_result_case(self, case_name, output_dir):
        img_files = sorted(os.listdir(os.path.join(self.root, "images", case_name, "imgs")))
        scene = []
        for img_file in img_files:
            scene.append({"objects": self.basename2objects[img_file]})
        write_serialized({"scene": scene}, os.path.join(output_dir, "{}.json".format(case_name)))

    def process_results(self, output_dir):
        for case_name in tqdm(self.case_names):
            self._process_result_case(case_name, output_dir)
