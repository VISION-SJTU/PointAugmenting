from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy

import torch
from torch.nn import functional as F

@DETECTORS.register_module
class PPFusion(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        img_backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PPFusion, self).__init__(
            reader, backbone, img_backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

        # img backbone pretrained
        for name, p in self.img_backbone.named_parameters():
            p.requires_grad = False
        self.img_backbone.eval()

        self.img_feat_num = 64
        self.max_points_in_voxel = 20

    def get_img_feat(self, img, pts_uv, voxels_valid):
        # pts_uv # B N 3 (u, v, cam_id)
        batch_size = img.shape[0]
        with torch.no_grad():
            img = img.view(-1, 3, img.shape[3], img.shape[4])  # B, 6, 3, H, W
            img_feat = self.img_backbone(img)
            # img_feat = torch.rand([batch_size * 6, 64, 112, 200]).to(img.device)
            img_feat = img_feat.view(batch_size, 6, -1, img_feat.shape[2], img_feat.shape[3]).transpose(2, 1)
            # print('img_feat', img_feat.shape)

            voxel_img_feat = F.grid_sample(img_feat, pts_uv, mode='bilinear', padding_mode='zeros')
            # print('voxel_img_feat', voxel_img_feat.view(-1))

            voxel_img_feat = voxel_img_feat.transpose(1, 4).contiguous()  # B * 1 * N * 10 * 64
            voxel_img_feat = voxel_img_feat.view(-1, self.max_points_in_voxel, self.img_feat_num).contiguous()  # (B * max_N) * 10 * C
            voxel_img_feat = voxel_img_feat[voxels_valid]
        return voxel_img_feat

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]  # Voxel_num * 10 * 5+3
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        voxels_uv = example["voxels_uv"]
        voxels_valid = example["voxel_valid"]

        batch_size = len(num_voxels)

        with torch.no_grad():
            voxels_feat = self.get_img_feat(example["img"], voxels_uv, voxels_valid)
            voxels_feat = voxels_feat * (voxels[:, :, -1].view(-1, self.max_points_in_voxel, 1))
            voxels_feat = torch.cat([voxels[:, :, :-4], voxels_feat], dim=2)

        data = dict(
            features=voxels_feat,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None



