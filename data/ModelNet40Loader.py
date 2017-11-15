import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py, subprocess, shlex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def _get_data_files(list_filename):
    return [line.rstrip()[5:] for line in open(list_filename)]


def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


class ModelNet40Cls(data.Dataset):
    def __init__(self,
                 num_points,
                 root,
                 transforms=None,
                 train=True,
                 download=True):
        super().__init__()

        self.transforms = transforms

        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(root, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(roo, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile)))

            subprocess.check_call(shlex.split("unzip {} -d {}".format(zipfile, root)))

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train, self.num_points = train, num_points
        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(root, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

        self.randomize()

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.actual_number_of_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs, :]
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        self.actual_number_of_points = min(
            max(
                np.random.randint(self.num_points * 0.8,
                                  self.num_points * 1.2), 1),
            self.points.shape[1])


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(x_axis=True),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter()
    ])
    dset = ModelNet40Cls(16, "./", train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
