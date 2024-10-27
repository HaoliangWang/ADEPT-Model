from dataset import build_dataset
from easydict import EasyDict


class DatasetCatalog(object):
    TRAIN_ROOT = "/mnt/fs0/haw027/b3d_ipe/train"
    HUMAN_ROOT = "/mnt/fs0/haw027/b3d_ipe/test"

    @staticmethod
    def get(name, args=None):
        if name.startswith("annotated_physics"):
            physics_map = {
                "annotated_physics_human": "/mnt/fs0/haw027/b3d_ipe/test/annotated_ann.json",
                "annotated_physics_train": "/mnt/fs0/haw027/b3d_ipe/train/annotated_ann.json",
                "annotated_physics_val": "/mnt/fs0/haw027/b3d_ipe/train/annotated_ann.json"
            }
            return build_dataset(
                EasyDict(NAME="OBJECT_PROPOSAL", OBJECT_DERENDER_FILE=physics_map[name],
                         SPLIT="TRAIN" if "train" in name else "VAL" if "val" in name else "TEST",
                         ROOT=DatasetCatalog.TRAIN_ROOT if "human" not in name else DatasetCatalog.HUMAN_ROOT,
                         ), args)