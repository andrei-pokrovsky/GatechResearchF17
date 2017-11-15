import torch
import torch.utils.data as data
import numpy as np
import os, sys, subprocess, shlex
import scipy.io as sio
import imageio
from collections import namedtuple
import progressbar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../models"))

MAX_NUM_BBOX = 10

import json

with open(os.path.join(BASE_DIR, "classname_to_idx.txt"), "r") as f:
    classname_to_idx = json.loads(f.read())

with open(os.path.join(BASE_DIR, "idx_to_classname.txt"), "r") as f:
    idx_to_classname = json.loads(f.read())


def im_loader(path):
    return np.array(imageio.imread(path))


all_angles = []
all_h = []
all_w = []
all_z = []


class SUNRGBD3DBBox(data.Dataset):
    item_return_type = namedtuple("item_return_type",
                                  ['pts', 'rgb', 'boxes', 'cls_lbl'])

    def __init__(self,
                 root,
                 number_of_points=None,
                 download=True,
                 data_precent=1.0,
                 preload=True):
        super().__init__()
        self.preload = preload
        self.data_precent = data_precent
        self.npoints = number_of_points
        root = os.path.abspath(root)
        self.folder = "SUNRGBD"
        self.data_dir = os.path.join(root, self.folder)
        data_url = "http://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
        ground_truth_url = "http://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat"
        ground_truth_file = os.path.join(self.data_dir,
                                         os.path.basename(ground_truth_url))
        print(ground_truth_file)

        if not os.path.exists(self.data_dir):
            if download:
                zipfile = os.path.join(root, os.path.basename(data_url))
                subprocess.check_call(
                    shlex.split("curl {} -o {}".format(data_url, zipfile)))

                subprocess.check_call(
                    shlex.split("unzip {} -d {}".format(zipfile, root)))

                subprocess.check_call(shlex.split("rm {}".format(zipfile)))
            else:
                raise RuntimeError(
                    "Could not find '{}', please correct the path or set download to true".
                    format(self.data_dir))

        if not os.path.exists(ground_truth_file):
            if download:
                subprocess.check_call(
                    shlex.split("curl {} -o {}".format(ground_truth_url,
                                                       ground_truth_file)))
            else:
                raise RuntimeError(
                    "Could not find '{}', please correct the path or set download to true".
                    format(ground_truth_file))

        self.meta_data = sio.loadmat(ground_truth_file)[
            'SUNRGBDMeta'].squeeze()
        valid = []
        max_num_bb = 0
        for i in range(self.meta_data.shape[0]):
            data = self.meta_data[i]
            if data[10].shape[0] == 0: continue
            if data[10][0].shape[0] == 0: continue

            cls_lbl = np.array([
                classname_to_idx[data[10][0][idx][3][0]]
                for idx in range(data[10][0].shape[0])
            ])
            rank = cls_lbl.max()

            if rank >= 5: continue

            valid.append(i)

            max_num_bb = max(data[10][0].shape[0], max_num_bb)

        self.valid = valid
        self.cache = {}

        self.preloaded = False
        if self.preload:
            self.preload_run = True
            bar = progressbar.ProgressBar()
            for i in bar(range(len(self))):
                self[i]

            self.preloaded = True

        self.preload_run = False

    def __getitem__(self, idx):
        data = self.meta_data[self.valid[idx]]

        r_tilt = data[1]
        direction = r_tilt[:, 1]
        angle = direction[0:2] / np.sqrt(direction[0:2].dot(direction[0:2]))
        r_correction = np.eye(3)
        if angle[1] < 0.95:
            theta = np.arctan2(angle[0], angle[1])
            cosval = np.cos(theta)
            sinval = np.sin(theta)
            r_correction[0:2, 0:2] = np.array([[cosval, -sinval],
                                               [sinval, cosval]])

        data[1] = r_correction.dot(data[1])
        pts, rgb = self._read_3d_points(data)

        if self.preloaded:
            boxes, cls_lbl = self.cache[str(idx)]
        else:
            bbox = self._unpack_3dbbox(data)
            for i in range(bbox.shape[0]):
                bbox[i, 0:3] = bbox[i, 0:3].dot(r_correction.T)
                bbox[i, 4] = bbox[i, 4].dot(r_correction.T)

            boxes = self._bbox_to_boxes(bbox)  # [h, w, z, theta, cx, cy, cz]

            cls_lbl = np.array([
                classname_to_idx[data[10][0][idx][3][0]]
                for idx in range(data[10][0].shape[0])
            ])

            if cls_lbl.shape[0] > MAX_NUM_BBOX:
                cls_lbl = cls_lbl[:MAX_NUM_BBOX]
                boxes = boxes[:MAX_NUM_BBOX]
            elif cls_lbl.shape[0] < MAX_NUM_BBOX:
                boxes = np.concatenate(
                    [
                        boxes,
                        np.zeros((MAX_NUM_BBOX - boxes.shape[0],
                                  boxes.shape[1]))
                    ],
                    axis=0)
                cls_lbl = np.concatenate(
                    [cls_lbl,
                     np.zeros(MAX_NUM_BBOX - cls_lbl.shape[0])],
                    axis=0)

            if self.preload:
                self.cache.update({str(idx): (boxes, cls_lbl)})

        if self.preload_run:
            return None

        return self.item_return_type(
            torch.from_numpy(pts).type(torch.FloatTensor),
            torch.from_numpy(rgb).type(torch.FloatTensor),
            torch.from_numpy(boxes).type(torch.FloatTensor),
            torch.from_numpy(cls_lbl).type(torch.LongTensor))

    def __len__(self):
        return int(len(self.valid) * self.data_precent)

    def randomize(self):
        pass

    @staticmethod
    def flip_towards_viewer(normals, points):
        points = points / np.sqrt((points**2).sum(axis=1, keepdims=True))

        proj = np.sum(points * normals, axis=1)
        flip = proj > 0
        normals[flip] = -normals[flip]

        return normals

    @classmethod
    def get_corners_of_3dbbox(cls, bbox):
        corners_list = []
        for i in range(bbox.shape[0]):
            basis = bbox[i][0:3]
            inds = np.argsort(-np.abs(basis[:, 0]))
            basis = basis[inds]
            coeffs = bbox[i][3][inds]

            inds = np.argsort(-np.abs(basis[1:, 1]))
            if inds[0] == 2:
                basis[1:] = np.flip(basis[1:], axis=0)
                coeffs[1:] = np.flip(coeffs[1:])

            centroid = np.expand_dims(bbox[i, 4], 0)
            basis = cls.flip_towards_viewer(basis, centroid)

            corners = np.empty((8, 3))
            tmp = basis * np.expand_dims(coeffs, 0)

            corners[0] = -tmp[0] + tmp[1] + tmp[2]
            corners[1] = tmp[0] + tmp[1] + tmp[2]
            corners[2] = tmp[0] - tmp[1] + tmp[2]
            corners[3] = -tmp[0] - tmp[1] + tmp[2]

            corners[4] = -tmp[0] + tmp[1] - tmp[2]
            corners[5] = tmp[0] + tmp[1] - tmp[2]
            corners[6] = tmp[0] - tmp[1] - tmp[2]
            corners[7] = -tmp[0] - tmp[1] - tmp[2]

            corners = corners + centroid
            corners_list.append(corners)

        return np.stack(corners_list)

    def _unpack_3dbbox(self, data):
        bbox = np.stack([
            np.concatenate([data[10][0][idx][i] for i in range(3)], axis=0)
            for idx in range(data[10][0].shape[0])
        ])

        for i in range(bbox.shape[0]):
            basis = bbox[i, 0:3]
            inds = np.argsort(-np.abs(basis[:, 0]))
            basis = basis[inds]
            coeffs = bbox[i, 3][inds]

            inds = np.argsort(-np.abs(basis[1:, 1]))
            if inds[0] == 2:
                basis[1:] = np.flip(basis[1:], axis=0)
                coeffs[1:] = np.flip(coeffs[1:])

            centroid = np.expand_dims(bbox[i, 4], 0)

            bbox[i, 0:3] = basis
            bbox[i, 3] = coeffs
            bbox[i, 4] = centroid

        return bbox

    def _read_3d_points(self, data):
        depth_map_name = data[3][0].split("SUNRGBD")[1][1:]
        if self.preloaded:
            pts, rgb = self.cache[depth_map_name]
        else:
            depth_map = im_loader(os.path.join(self.data_dir, depth_map_name))

            depth_inpaint = np.bitwise_or(
                np.right_shift(depth_map, 3), np.left_shift(depth_map, 16 - 3))
            depth_inpaint = (depth_inpaint / 1000.0)
            depth_inpaint[depth_inpaint > 8] = 8

            valid = depth_inpaint != 0

            rgb_name = data[4][0].split("SUNRGBD")[1][1:]
            rgb = im_loader(os.path.join(self.data_dir, rgb_name))

            rgb = rgb.astype(np.float32)
            rgb = (rgb - rgb.min(axis=(0, 1), keepdims=True)) / (
                rgb.max(axis=(0, 1), keepdims=True) - rgb.min(
                    axis=(0, 1), keepdims=True))

            K = data[2]
            cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]

            x, y = np.meshgrid(
                np.array([x for x in range(1, depth_inpaint.shape[1] + 1)]),
                np.array([y for y in range(1, depth_inpaint.shape[0] + 1)]))

            x3 = (x - cx) * depth_inpaint / fx
            y3 = (y - cy) * depth_inpaint / fy
            z3 = depth_inpaint

            pts = np.stack(
                (x3[valid], z3[valid], -y3[valid]),
                axis=-1).astype(np.float32)
            rgb = rgb[valid]

            #  voxel_size = 5e-2
            #  voxel_grid = {}
            #  for i in range(pts.shape[0]):
            #  pt = pts[i]
            #  color = rgb[i]
            #  key = (pt / voxel_size).astype(np.int32)
            #  if key in voxel_grid:
            #  voxel_grid[key].append((pt, color))
            #  else:
            #  voxel_grid.update({key: [(pt, color)]})

            if self.preload:
                self.cache.update({depth_map_name: (pts, rgb)})

        if not self.preload_run:
            if self.npoints is not None:
                stride = int(pts.shape[0] / self.npoints)
                rand_idx = np.random.randint(0, stride, size=self.npoints)
                mask = np.concatenate([
                    np.concatenate([
                        np.array([0 for _ in range(rand_idx[i])] + [1] +
                                 [0 for _ in range(stride - rand_idx[i] - 1)])
                        for i in range(self.npoints)
                    ]),
                    np.array([
                        0 for _ in range(pts.shape[0] - stride * self.npoints)
                    ])
                ]).astype(bool)

                pts = pts[mask]
                rgb = rgb[mask]

            order = np.random.permutation(pts.shape[0])
            np.take(pts, order, axis=0, out=pts)
            np.take(rgb, order, axis=0, out=rgb)

            r_tilt = data[1].T
            pts = pts.dot(r_tilt).astype(np.float32)

        return pts, rgb

    def _get_label_truth_frcnn(self, data, pts):
        bbox = self._unpack_3dbbox(data)
        corners = torch.from_numpy(self.get_corners_of_3dbbox(bbox)).type(
            torch.FloatTensor)

        positive_boxes = []
        negative_boxes = []
        corner_highest = [(0.0, None) for _ in range(corners.size(0))]

        for j in range(corners.size(0)):
            c = corners[j]
            b = bbox[j]
            for i in range(pts.shape[0]):
                point = pts[i]
                for k, anchor in enumerate(anchor_set):
                    ba = get_bbox_of_anchor(anchor)
                    ca = self.get_corners_of_3dbbox(np.expand_dims(ba, axis=0))
                    ca += point[np.newaxis, :]
                    vol = cuboid_intersection_volume(c, ca).numpy()[0]

                    anchor_x = ba[0]
                    true_x = b[0]

                    cos_theta = max(
                        min(
                            anchor_x.dot(true_x) /
                            (np.sqrt(anchor_x.dot(anchor_x)) * np.sqrt(
                                true_x.dot(true_x))), 1.0), -1.0)
                    delta_theta = np.arccos(cos_theta)

                    anchor_coeffs = ba[3]
                    true_coeffs = b[3]

                    delta_coeffs = true_coeffs - anchor_coeffs

                    delta = np.empty((4, ), dtype=np.float32)
                    delta[:3] = delta_coeffs
                    delta[3] = delta_theta

                    idx = i * len(anchor_set) + k
                    if vol > 0.1:
                        positive_boxes.append((delta, idx))
                        corner_highest[j] = (None, None)
                    else:
                        negative_boxes.append((delta, idx))

                    best, _ = corner_highest[j]
                    if best is not None and vol > best:
                        corner_highest[j] = (vol, (delta, idx))

        for vol, item in corner_highest:
            if item is not None:
                positive_boxes.append(item)

        positive_sample = np.arange(len(positive_boxes))
        np.random.shuffle(positive_sample)
        negative_sample = np.arange(len(negative_boxes))
        np.random.shuffle(negative_sample)

        index, box_truth, conf_truth = [], [], []

        for i in range(min(len(positive_boxes), self.label_batch_size // 2)):
            delta, idx = positive_boxes[positive_sample[i]]

            box_truth.append(delta)
            conf_truth.append(1)
            index.append(idx)

        for i in range(self.label_batch_size - len(box_truth)):
            delta, idx = negative_boxes[negative_sample[i]]

            box_truth.append(delta)
            conf_truth.append(0)
            index.append(idx)

        return np.array(conf_truth), np.stack(box_truth), np.array(index)

    def _bbox_to_boxes(self, bbox):
        B = bbox.shape[0]
        boxes = np.empty((B, 7))

        for i in range(B):
            b = bbox[i]

            bx = b[0]
            by = b[1]
            wdh = b[3]
            centroid = b[4]

            if np.sign(bx[0]) != np.sign(by[1]):
                by = -by
            tx = (np.arccos(bx[0]))
            ty = np.arccos(by[1])
            theta = (tx + ty) / 2.0

            sinval = np.sin(theta)
            cosval = np.cos(theta)
            bx = np.array([cosval, -sinval, 0.0])
            by = np.array([sinval, cosval, 0.0])

            if np.dot(bx, centroid) > 0:
                bx = -bx
            if np.dot(by, centroid) > 0:
                by = -by

            boxes[i, :3] = wdh * np.array([1.0, 1.0, 1.0])
            boxes[i, 3] = tx
            boxes[i, 4:] = centroid

            #  global all_h
            #  global all_w
            #  global all_z
            #  all_h.append(whz[1])
            #  all_w.append(whz[0])
            #  all_z.append(whz[2])

        return boxes


if __name__ == "__main__":
    dset = SUNRGBD3DBBox("./", data_precent=1.0)
    all_boxes = []
    for k, v in dset.cache.items():
        try:
            tmp = int(k)
        except ValueError:
            continue

        boxes, _ = v
        valid = []
        for i in range(boxes.shape[0]):
            if (boxes[i] != 0).all():
                valid.append(i)

        for idx in valid:
            all_boxes.append(boxes[idx, 0:4])

    all_boxes = np.stack(all_boxes)

    from sklearn.cluster import KMeans
    kmeans_algo = KMeans(
        n_clusters=64,
        precompute_distances=True,
        n_init=int(1e3),
        max_iter=int(1e6),
        n_jobs=-1)
    kmeans = kmeans_algo.fit(all_boxes)
    centers = kmeans.cluster_centers_
    for j in range(centers.shape[0]):
        for i in range(centers.shape[1]):
            centers[j, i] = np.floor(centers[j, i] * 1e2) * 1e-2

    print(centers)
    np.save("bbox_clusters.npy", centers)

    if False:
        class_name_freq = {}
        for i in range(len(dset)):
            ret = dset[i]

            for name in ret:
                try:
                    class_name_freq[name] += 1
                except:
                    class_name_freq.update({name: 1})

        classname_to_idx = {}
        idx_to_classname = []

        from json import dumps
        import operator

        sorted_names = sorted(
            class_name_freq.items(), key=operator.itemgetter(1), reverse=True)

        for name, _ in sorted_names:
            classname_to_idx.update({name: len(idx_to_classname)})
            idx_to_classname.append(name)

        with open("classname_to_idx.txt", "w") as f:
            f.write(dumps(classname_to_idx))

        with open("idx_to_classname.txt", "w") as f:
            f.write(dumps(idx_to_classname))
