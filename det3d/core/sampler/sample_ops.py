import copy
import pathlib
import pickle
import time
from functools import partial, reduce

import numpy as np
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.utils.check import shape_mergeable

def corners_to_bbox(info, corners, cam_name, calib, imsize=(900, 1600)):
    avail_new = info['avail_2d'].copy()
    bboxes_new = np.zeros([6, 4], dtype=np.int32)
    scale = np.zeros([6, 2], dtype=np.float32)
    for cam_id, cam_sensor in enumerate(cam_name):
        if info['avail_2d'][cam_id]:
            cam_from_global = calib["cams_from_global"][cam_sensor]
            cam_intrinsic = calib["cam_intrinsics"][cam_sensor]

            # lidar to global
            ref_to_global = calib["ref_to_global"]
            pts_hom = np.concatenate([corners[:, :3], np.ones([corners.shape[0], 1])], axis=1)
            pts_global = ref_to_global.dot(pts_hom.T)  # 4 * N

            # global to cam
            pts_cam = cam_from_global.dot(pts_global)[:3, :]  # 3 * N

            # cam to uv
            from det3d.datasets.pipelines.loading import view_points
            pts_uv = view_points(pts_cam, np.array(cam_intrinsic), normalize=True).T  # N * 3

            bbox = np.array([np.min(pts_uv[:, 0], axis=0), np.min(pts_uv[:, 1], axis=0),
                             np.max(pts_uv[:, 0], axis=0), np.max(pts_uv[:, 1], axis=0)], dtype=np.int32)
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(imsize[1], bbox[2])
            bbox[3] = min(imsize[0], bbox[3])
            bbox = bbox.astype(np.int32)
            if bbox[0] >= imsize[1] - 1 or bbox[1] >= imsize[0] - 1 or bbox[2] < 0 or bbox[3] < 0 or bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                avail_new[cam_id] = False
            else:
                bboxes_new[cam_id] = bbox.copy()
                # transform[cam_id, 0] = (bbox[0] + bbox[2] - info['bbox'][0] - info['bbox'][2]) / 2.
                # transform[cam_id, 1] = (bbox[1] + bbox[3] - info['bbox'][1] - info['bbox'][3]) / 2.
                scale[cam_id, 0] = (bbox[2] - bbox[0]) / (info['bbox'][cam_id, 2] - info['bbox'][cam_id, 0])
                scale[cam_id, 1] = (bbox[3] - bbox[1]) / (info['bbox'][cam_id, 3] - info['bbox'][cam_id, 1])

    return avail_new, bboxes_new, scale


class DataBaseSamplerV2:
    def __init__(
        self,
        db_infos,
        groups,
        db_prepor=None,
        rate=1.0,
        global_rot_range=None,
        logger=None,
    ):
        for k, v in db_infos.items():
            logger.info(f"load {len(v)} {k} database infos")

        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            logger.info("After filter database:")
            for k, v in db_infos.items():
                logger.info(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._use_group_sampling = False  # slower
        if any([len(g) > 1 for g in groups]):
            self._use_group_sampling = True
        if not self._use_group_sampling:
            self._group_db_infos = self.db_infos  # just use db_infos
            for group_info in groups:
                group_names = list(group_info.keys())
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
        else:
            for group_info in groups:
                group_dict = {}
                group_names = list(group_info.keys())
                group_name = ", ".join(group_names)
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
                self._group_name_to_names.append((group_name, group_names))
                # self._group_name_to_names[group_name] = group_names
                for name in group_names:
                    for item in db_infos[name]:
                        gid = item["group_id"]
                        if gid not in group_dict:
                            group_dict[gid] = [item]
                        else:
                            group_dict[gid] += [item]
                if group_name in self._group_db_infos:
                    raise ValueError("group must be unique")
                group_data = list(group_dict.values())
                self._group_db_infos[group_name] = group_data
                info_dict = {}
                if len(group_info) > 1:
                    for group in group_data:
                        names = [item["name"] for item in group]
                        names = sorted(names)
                        group_name = ", ".join(names)
                        if group_name in info_dict:
                            info_dict[group_name] += 1
                        else:
                            info_dict[group_name] = 1
                print(info_dict)

        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = prep.BatchSampler(v, k)
        self._enable_global_rot = False
        if global_rot_range is not None:
            if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
                global_rot_range = [-global_rot_range, global_rot_range]
            else:
                assert shape_mergeable(global_rot_range, [2])
            if np.abs(global_rot_range[0] - global_rot_range[1]) >= 1e-3:
                self._enable_global_rot = True
        self._global_rot_range = global_rot_range

    @property
    def use_group_sampling(self):
        return self._use_group_sampling

    def sample_all(
        self,
        root_path,
        gt_boxes,
        gt_names,
        num_point_features,
        random_crop=False,
        gt_group_ids=None,
        calib=None,
        road_planes=None,
    ):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self._sample_classes, self._sample_max_nums
        ):
            sampled_num = int(
                max_sample_num - np.sum([n == class_name for n in gt_names])
            )

            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        if self._use_group_sampling:
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for group_name, class_names in self._group_name_to_names:
                sampled_nums_group = [sampled_num_dict[n] for n in class_names]
                sampled_num = np.max(sampled_nums_group)
                sample_num_per_class.append(sampled_num)
                sampled_groups.append(group_name)
            total_group_ids = gt_group_ids
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
            if sampled_num > 0:
                if self._use_group_sampling:
                    sampled_cls = self.sample_group(
                        class_name, sampled_num, avoid_coll_boxes, total_group_ids
                    )
                else:
                    sampled_cls = self.sample_class_v2(
                        class_name, sampled_num, avoid_coll_boxes
                    )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )
                    if self._use_group_sampling:
                        if len(sampled_cls) == 1:
                            sampled_group_ids = np.array(sampled_cls[0]["group_id"])[
                                np.newaxis, ...
                            ]
                        else:
                            sampled_group_ids = np.stack(
                                [s["group_id"] for s in sampled_cls], axis=0
                            )
                        total_group_ids = np.concatenate(
                            [total_group_ids, sampled_group_ids], axis=0
                        )

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)

            num_sampled = len(sampled)
            s_points_list = []
            for info in sampled:
                try:
                    s_points = np.fromfile(
                        str(pathlib.Path(root_path) / info["path"]), dtype=np.float32
                    ).reshape(-1, num_point_features)

                    if "rot_transform" in info:
                        rot = info["rot_transform"]
                        s_points[:, :3] = box_np_ops.rotation_points_single_angle(
                            s_points[:, :4], rot, axis=2
                        )
                    s_points[:, :3] += info["box3d_lidar"][:3]
                    s_points_list.append(s_points)
                    # print(pathlib.Path(info["path"]).stem)
                except Exception:
                    print(str(pathlib.Path(root_path) / info["path"]))
                    continue
            if random_crop:
                s_points_list_new = []
                assert calib is not None
                rect = calib["rect"]
                Trv2c = calib["Trv2c"]
                P2 = calib["P2"]
                gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect, Trv2c, P2)
                crop_frustums = prep.random_crop_frustum(gt_bboxes, rect, Trv2c, P2)
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = prep.mask_points_in_corners(
                        s_points, crop_frustums[i : i + 1]
                    ).reshape(-1)
                    num_remove = np.sum(mask)
                    if num_remove > 0 and (s_points.shape[0] - num_remove) > 15:
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                s_points_list = s_points_list_new
            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones((num_sampled,), dtype=np.bool_),
            }
            if self._use_group_sampling:
                ret["group_ids"] = np.array([s["group_id"] for s in sampled])
            else:
                ret["group_ids"] = np.arange(
                    gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled)
                )
        else:
            ret = None
        return ret

    def sample_all_v2(
            self,
            root_path,
            gt_boxes,
            gt_names,
            gt_boxes_frustum,
            selected_feature,
            random_crop=False,
            revise_calib=False,
            gt_group_ids=None,
            calib=None,
            cam_name=None,
            road_planes=None,
    ):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
                self._sample_classes, self._sample_max_nums
        ):
            sampled_num = int(
                max_sample_num - np.sum([n == class_name for n in gt_names])
            )

            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        if self._use_group_sampling:
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for group_name, class_names in self._group_name_to_names:
                sampled_nums_group = [sampled_num_dict[n] for n in class_names]
                sampled_num = np.max(sampled_nums_group)
                sample_num_per_class.append(sampled_num)
                sampled_groups.append(group_name)
            total_group_ids = gt_group_ids
        sampled = []
        sampled_gt_boxes = []
        sampled_gt_frustums = []
        avoid_coll_boxes = gt_boxes
        avoid_coll_frustums = gt_boxes_frustum

        for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
            if sampled_num > 0:
                if self._use_group_sampling:
                    sampled_cls = self.sample_group(
                        class_name, sampled_num, avoid_coll_boxes, total_group_ids
                    )
                else:
                    # sampled_cls = self.sample_class_v2(
                    #     class_name, sampled_num, avoid_coll_boxes,
                    # )
                    sampled_cls = self.sample_class_v3(
                        class_name, sampled_num, avoid_coll_boxes, avoid_coll_frustums
                    )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                        sampled_gt_frustum = sampled_cls[0]["frustum"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )
                        sampled_gt_frustum = np.stack(
                            [s["frustum"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_boxes += [sampled_gt_box]
                    sampled_gt_frustums += [sampled_gt_frustum]
                    avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)
                    avoid_coll_frustums = np.concatenate([avoid_coll_frustums, sampled_gt_frustum], axis=0)

                    if self._use_group_sampling:
                        if len(sampled_cls) == 1:
                            sampled_group_ids = np.array(sampled_cls[0]["group_id"])[
                                np.newaxis, ...
                            ]
                        else:
                            sampled_group_ids = np.stack(
                                [s["group_id"] for s in sampled_cls], axis=0
                            )
                        total_group_ids = np.concatenate(
                            [total_group_ids, sampled_group_ids], axis=0
                        )

        # load sampled points
        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
            sampled_gt_frustums = np.concatenate(sampled_gt_frustums, axis=0)

            sampled_corners = box_np_ops.center_to_corner_box3d(sampled_gt_boxes[:, :3], sampled_gt_boxes[:, 3:6],
                                                                sampled_gt_boxes[:, -1])

            if road_planes is not None:
                # Only support KITTI
                # image plane
                assert False, "Not correct yet!"
                a, b, c, d = road_planes

                center = sampled_gt_boxes[:, :3]
                center[:, 2] -= sampled_gt_boxes[:, 5] / 2
                center_cam = box_np_ops.lidar_to_camera(center, calib["rect"], calib["Trv2c"])

                cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
                center_cam[:, 1] = cur_height_cam
                lidar_tmp_point = box_np_ops.camera_to_lidar(center_cam, calib["rect"], calib["Trv2c"])
                cur_lidar_height = lidar_tmp_point[:, 2]

                # botom to middle center
                # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
                sampled_gt_boxes[:, 2] = cur_lidar_height + sampled_gt_boxes[:, 5] / 2

                # mv_height = sampled_gt_boxes[:, 2] - cur_lidar_height
                # sampled_gt_boxes[:, 2] -= mv_height

            num_sampled = len(sampled)
            s_points_list = []
            for i, info in enumerate(sampled):
                try:
                    # TODO fix point read error
                    s_points = np.fromfile(
                        str(pathlib.Path(root_path) / info["path"]), dtype=np.float32
                    )
                    s_points = s_points.reshape(-1, 5 + 3)[:, selected_feature > 0.]
                    # if not add_rgb_to_points:
                    #     s_points = s_points[:, :4]
                    if "rot_transform" in info:
                        rot = info["rot_transform"]
                        s_points[:, :3] = box_np_ops.rotation_points_single_angle(
                            s_points[:, :4], rot, axis=2
                        )
                    s_points[:, :3] += info["box3d_lidar"][:3]

                    # print(pathlib.Path(info["path"]).stem)
                except Exception:
                    print(str(pathlib.Path(root_path) / info["path"]), 'error')
                    continue

                if revise_calib and calib is not None:
                    im_shape = (448*2, 1600)
                    # update bbox and pts
                    avail_new, bboxes_new, scale = corners_to_bbox(info, sampled_corners[i], cam_name, calib, im_shape)
                    changed_ids = np.logical_xor(avail_new, info['avail_2d'])
                    changed_ids = np.where(changed_ids)[0]
                    changed_ids = changed_ids / (6 - 1) * 2 - 1
                    # 原来有cam现在没有的->投影点cam_id设为无穷大
                    for changed_id in changed_ids:
                        s_points[s_points[:, -1] == changed_id, -1] = -41
                    update_ids = np.where(avail_new)[0]
                    for update_id in update_ids:
                        val = (s_points[:, -1] == (update_id / (6 - 1) * 2 - 1))
                        tmp1 = info['bbox'][update_id, 0] / (im_shape[1] - 1) * 2 - 1
                        tmp2 = info['bbox'][update_id, 1] / (im_shape[0] - 1) * 2 - 1
                        tmp3 = bboxes_new[update_id, 0] / (im_shape[1] - 1) * 2 - 1
                        tmp4 = bboxes_new[update_id, 1] / (im_shape[0] - 1) * 2 - 1
                        s_points[val, -3] = (s_points[val, -3] - tmp1) * scale[update_id, 0] + tmp3
                        s_points[val, -2] = (s_points[val, -2] - tmp2) * scale[update_id, 1] + tmp4
                    sampled[i]['bbox'] = bboxes_new
                    sampled[i]['avail_2d'] = avail_new
                s_points_list.append(s_points)

            # gt_bboxes = np.stack([s["bbox"] for s in sampled], axis=0)
            # if np.random.choice([False, True], replace=False, p=[0.3, 0.7]):
            # do random crop.
            if random_crop:
                s_points_list_new = []
                assert calib is not None
                rect = calib["rect"]
                Trv2c = calib["Trv2c"]
                P2 = calib["P2"]
                gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect, Trv2c, P2)
                crop_frustums = prep.random_crop_frustum(gt_bboxes, rect, Trv2c, P2)
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = prep.mask_points_in_corners(
                        s_points, crop_frustums[i: i + 1]
                    ).reshape(-1)
                    num_remove = np.sum(mask)
                    if num_remove > 0 and (s_points.shape[0] - num_remove) > 15:
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                s_points_list = s_points_list_new
            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "avail_2d": np.array([s["avail_2d"] for s in sampled]),
                "bboxes": np.array([s["bbox"] for s in sampled]),
                "depths": np.array([s["depth"] for s in sampled]),
                "patch_path": np.array([s["cam_paths"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "gt_frustums": sampled_gt_frustums,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones((num_sampled,), dtype=np.bool_),
            }
            if self._use_group_sampling:
                ret["group_ids"] = np.array([s["group_id"] for s in sampled])
            else:
                ret["group_ids"] = np.arange(
                    gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled)
                )
        else:
            ret = None
        return ret


    def sample(self, name, num):
        if self._use_group_sampling:
            group_name = name
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_v1(self, name, num):
        if isinstance(name, (list, tuple)):
            group_name = ", ".join(name)
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes, None, valid_mask, 0, 0, self._global_rot_range, num_try=100
            )

        sp_boxes_new = boxes[gt_boxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, -1]
        )

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot:
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (
                        boxes[i, -1] - sp_boxes[i - num_gt, -1]
                    )
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    def sample_group(self, name, num, gt_boxes, gt_group_ids):
        sampled, group_num = self.sample(name, num)
        sampled = copy.deepcopy(sampled)
        # rewrite sampled group id to avoid duplicated with gt group ids
        gid_map = {}
        max_gt_gid = np.max(gt_group_ids)
        sampled_gid = max_gt_gid + 1
        for s in sampled:
            gid = s["group_id"]
            if gid in gid_map:
                s["group_id"] = gid_map[gid]
            else:
                gid_map[gid] = sampled_gid
                s["group_id"] = sampled_gid
                sampled_gid += 1

        num_gt = gt_boxes.shape[0]
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_group_ids = np.stack([i["group_id"] for i in sampled], axis=0)
        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        group_ids = np.concatenate([gt_group_ids, sp_group_ids], axis=0)
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                group_ids=group_ids,
                num_try=100,
            )
        sp_boxes_new = boxes[gt_boxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, -1]
        )
        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False
        valid_samples = []
        idx = num_gt
        for num in group_num:
            if coll_mat[idx : idx + num].any():
                coll_mat[idx : idx + num] = False
                coll_mat[:, idx : idx + num] = False
            else:
                for i in range(num):
                    if self._enable_global_rot:
                        sampled[idx - num_gt + i]["box3d_lidar"][:2] = boxes[
                            idx + i, :2
                        ]
                        sampled[idx - num_gt + i]["box3d_lidar"][-1] = boxes[
                            idx + i, -1
                        ]
                        sampled[idx - num_gt + i]["rot_transform"] = (
                            boxes[idx + i, -1] - sp_boxes[idx + i - num_gt, -1]
                        )

                    valid_samples.append(sampled[idx - num_gt + i])
            idx += num
        return valid_samples

    def sample_class_v3(self, name, num, gt_boxes, gt_frustums):
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes, None, valid_mask, 0, 0, self._global_rot_range, num_try=100
            )

        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, -1]
        )

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv)

        sp_frustums = np.stack([i["frustum"] for i in sampled], axis=0)
        frustum_coll_mat = self.frustum_collision_test(gt_frustums, sp_frustums)
        coll_mat = np.logical_or(coll_mat, frustum_coll_mat)

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot:
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (
                            boxes[i, -1] - sp_boxes[i - num_gt, -1]
                    )
                valid_samples.append(sampled[i - num_gt])

        return valid_samples

    def frustum_collision_test(self, gt_frustums, sp_frustums, thresh=0.7):
        ## calculate iou
        N = gt_frustums.shape[0]
        K = sp_frustums.shape[0]
        gt_frustums_all = np.concatenate([gt_frustums, sp_frustums], axis=0)
        S = np.array([(cur_frus[1, 1, 0] - cur_frus[1, 0, 0]) * (cur_frus[2, 1, 0] - cur_frus[2, 0, 0] + cur_frus[2, 1, 1] - cur_frus[2, 0, 1]) \
                      for cur_frus in gt_frustums_all], dtype=np.float32)
        # assert S.any() > 0
        ret = np.zeros((N+K, N+K), dtype=np.float32)
        for i in range(N+K):
            for j in range(K):
                sp_frus = [sp_frustums[j, :, :, 0]] if sp_frustums[j, 2, 0, 1] < 0 else [sp_frustums[j, :, :, 0], sp_frustums[j, :, :, 1]]
                gt_frus = [gt_frustums_all[i, :, :, 0]] if gt_frustums_all[i, 2, 0, 1] < 0 else [gt_frustums_all[i, :, :, 0], gt_frustums_all[i, :, :, 1]]
                iou = 0
                for cur_sp_frus in sp_frus:
                    for cur_gt_frus in gt_frus:
                        coll = ((max(cur_sp_frus[2, 0], cur_gt_frus[2, 0]) < min(cur_sp_frus[2, 1], cur_gt_frus[2, 1]))
                            and (max(sp_frustums[j, 1, 0, 0], gt_frustums_all[i, 1, 0, 0]) < min(sp_frustums[j, 1, 1, 0], gt_frustums_all[i, 1, 1, 0])))
                        if coll:
                            iou += (min(cur_sp_frus[2, 1], cur_gt_frus[2, 1]) - max(cur_sp_frus[2, 0], cur_gt_frus[2, 0])) * \
                            (min(sp_frustums[j, 1, 1, 0], gt_frustums_all[i, 1, 1, 0]) - max(sp_frustums[j, 1, 0, 0], gt_frustums_all[i, 1, 0, 0]))
                            # assert iou > 0

                iou_per = iou / min(S[i], S[j+N])
                # assert iou_per <= 1.01
                ret[i, j + N] = iou_per
                ret[j + N, i] = iou_per

        ret = ret > thresh
        return ret

