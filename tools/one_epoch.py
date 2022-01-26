import numpy as np

from det3d.datasets.dataset_factory import get_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.datasets import build_dataset

from torch.utils.data import DataLoader
from det3d.torchie.parallel import collate, collate_kitti

import warnings
warnings.filterwarnings("ignore")

import time

def example_to_device(example, device, non_blocking=False) -> dict:
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm",
                "anno_box", "ind", "mask", 'cat']:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
            "img",
            "voxels_uv",
            "voxel_valid",
            "voxels_imgfeat",
            "bev_sparse"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


if __name__ == "__main__":
    # config_file = './configs/nusc/voxelnet/nusc_voxelnet.py'
    config_file = './configs/nusc/voxelnet/nusc_voxelnet_img.py'

    # config_file = './configs/nusc/pp/nusc_pp.py'
    # config_file = './configs/nusc/pp/nusc_pp_img.py'

    cfg = Config.fromfile(config_file)

    dataset = build_dataset(cfg.data.train)
    # dataset = build_dataset(cfg.data.val)
    print(dataset.__len__())

    for i in range(100):
        data = dataset.__getitem__(i)
        break
    print(11)

    data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_kitti,
            pin_memory=False,
        )

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = model.cuda()
    # model.eval()

    for i, data_batch in enumerate(data_loader):
        example = example_to_device(
            data_batch, 'cuda', non_blocking=False
        )
        losses = model(example, return_loss=True)
        print('loss', losses)
        break


