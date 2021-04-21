import os
import torch
import numpy as np
from torch.utils.data import Dataset


class GiWRecord:
    def __init__(self, row):
        self._data = row

    @property
    def cur_path(self):
        return self._data[0]

    @property
    def fur_paths(self):
        return self._data[1]

    @property
    def yt(self):
        return self._data[2]


class GiWDataset(Dataset):
    def __init__(self, root_path=None, transforms=None, sampling=False, n_positive=4, positive_ratio=1. / 2.):
        self.data_root = root_path
        self.input_shape = None
        self.sampling = sampling
        self.n_positive = int(n_positive)
        self.n_negative = int(int(n_positive) * (1. / float(positive_ratio) - 1))
        self.positive_ratio = float(positive_ratio)
        self.record_list = None
        self.cur_transforms, self.fur_transforms, self.label_transforms = None, None, None
        self.is_transform = True if transforms is not None else False
        if self.is_transform:
            self.cur_transforms, self.fur_transforms, self.label_transforms = transforms
        self._prepare_list(self.data_root)

    def _prepare_list(self, path=None):
        self.record_list = list()
        scene_dirs = [os.path.join(path, s_dir) for s_dir in os.listdir(path) if os.path.isdir(s_dir)]
        for scene_dir in scene_dirs:
            video_dirs = [os.path.join(scene_dir, v_dir) for v_dir in os.listdir(scene_dir) if os.path.isdir(v_dir)]
            for vid_dir in video_dirs:
                cur_image_home = os.path.join(vid_dir, 'cur_images')
                for cur_image_name in os.listdir(cur_image_home):
                    cur_image_name = os.path.join(cur_image_home, cur_image_name)
                    fur_name_tmpl = cur_image_name.replace('cur', 'fur').replace('.npy', '_{:02d}.npy')
                    n_fur_state_names = [fur_name_tmpl.format(i) for i in range(self.n_fur_states)]
                    label_yt_name = cur_image_name.replace('cur_images', 'label_yt').replace('.npy', '_yt.npy')
                    record = GiWRecord([cur_image_name, n_fur_state_names, label_yt_name])
                    self.record_list.append(record)

    @staticmethod
    def load_np_images(np_paths):
        # preprocessed numpy image
        return [np.load(np_image_path) for np_image_path in np_paths]

    def __getitem__(self, index):
        record = self.record_list[index]
        cur_state = self.load_np_images([record.cur_path])

        selected_idx = record.fur_paths
        if self.sampling:
            # sampling positive and negative fur_states
            yt_values = np.load(record.yt)
            yt_sorted_idx = np.argsort(yt_values)[::-1]     # from large to small
            positive_idx = yt_sorted_idx[:self.n_positive]
            negative_idx = np.random.choice(yt_sorted_idx[self.n_positive:], self.n_negative)
            selected_idx = np.hstack(positive_idx, negative_idx)

        fur_states = self.load_np_images([record.fur_paths[idx] for idx in selected_idx])
        label_imgs = self.load_np_images([record.fur_paths[idx].replace('cur', 'label') for idx in selected_idx])

        if self.is_transform:
            cur_state = self.cur_transforms(cur_state)
            fur_states = self.fur_transforms(fur_states)
            label_imgs = self.label_transforms(label_imgs)

        return cur_state, fur_states, label_imgs

    def __len__(self):
        return len(self.record_list)

    @property
    def cur_shape(self):
        return 7, 360, 640

    @property
    def fur_shape(self):
        return 7, 45, 80

    @property
    def label_shape(self):
        return 1, 23, 40

    @property
    def n_fur_states(self):
        return 35
