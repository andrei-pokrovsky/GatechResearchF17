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
    def __init__(self, num_points, root, train=True, download=True):
        super().__init__()
        root = os.path.abspath(root)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(root, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(self.url)

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.basename(self.url)
            subprocess.check_call(shlex.split("wget {}".format(self.url)))
            subprocess.check_call(shlex.split("unzip {}".format(zipfile)))
            if os.path.dirname(os.path.abspath(__file__)) != root:
                subprocess.check_call(
                    shlex.shlex("mv {} {}".format(zipfile[:-4], root)))
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
        print(np.mean(self.points.max(axis=1) - self.points.min(axis=1), axis=0))
        self.labels = np.concatenate(label_list, 0)

    def __getitem__(self, idx):
        actual_number_of_points = min(
            max(
                np.random.randint(self.num_points * 0.8,
                                  self.num_points * 1.2), 1), 2048)
        pt_idxs = np.arange(0, actual_number_of_points)
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs, :]).type(
            torch.FloatTensor)
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        current_points = rotate(current_points)
        current_points = translate(current_points)
        current_points = scale(current_points)
        current_points = jitter(current_points)

        if not (actual_number_of_points == 2048
                or actual_number_of_points == int(1.2 * self.num_points)):
            current_points = torch.cat(
                [
                    current_points,
                    torch.FloatTensor(
                        min(2048 - actual_number_of_points,
                            int(1.2 * self.num_points) -
                            actual_number_of_points), 3).zero_()
                ],
                dim=0)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = pts


def rotate(points):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = torch.FloatTensor([[cosval, 0, sinval], [0, 1, 0],
                                         [-sinval, 0, cosval]])
    return points @ rotation_matrix


def jitter(points, sigma=0.01, clip=0.05):
    jittered_data = torch.FloatTensor(*points.size()).normal_(
        mean=0.0, std=sigma).clamp_(-clip, clip)
    return points + jittered_data


def translate(points, sigma=1.0, clip=3.0):
    translation = (torch.FloatTensor(3).normal_(mean=0.0, std=sigma)).clamp_(
        -clip, clip)
    return points + translation


def scale(points, sigma=1.0, clip=1.8):
    mean = 2.0
    scaler = (torch.FloatTensor(1).normal_(mean=mean, std=sigma)).clamp_(
        max(mean - clip, 0.01), mean + clip)
    return scaler * points


if __name__ == "__main__":
    dset = ModelNet40Cls(16, "./", train=True)
    print(dset[0][0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
