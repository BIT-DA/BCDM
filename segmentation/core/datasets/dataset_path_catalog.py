import os
from .cityscapes import cityscapesDataSet
from .gta5 import GTA5DataSet


class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "gta5_train": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None):
        if "gta5" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTA5DataSet(args["root"], args["data_list"],
                               max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return cityscapesDataSet(args["root"], args["data_list"],
                                     max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform)
        raise RuntimeError("Dataset not available: {}".format(name))
