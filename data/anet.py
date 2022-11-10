from PIL import Image
import os
import torch
import os.path
from torch.utils.data import Dataset
from data.anet_db import ANetDB
from tqdm import tqdm


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


def load_images(path, indices):
    images = []
    for i in indices:
        f = open(os.path.join(path, '%06d.jpg' % i), 'rb')
        img = Image.open(f)
        images.append(img.convert('RGB'))
        f.close()
    return images


class ANDataSet(Dataset):
    def __init__(self, root_path, list_file, duration, transform=None, mode='training', kd=False):
        assert mode in ['training', 'validation', 'testing'], 'mode must be one of train, val or test'
        self.dir_path = os.path.join(root_path, mode)
        self.duration = duration
        self.transform = transform
        self.mode = mode
        self.kd = kd

        self.db = ANetDB.get_db(list_file)
        self._make_sample_list(100)

    def _make_sample_list(self, sample_interval=10):
        instances_list = self.db.get_subset_instance(self.mode)
        self.sample_list = []
        for i in tqdm(instances_list):
            path = os.path.join(self.dir_path, '_'+i.name[:11])
            if self.mode == 'validation' and i.name[:11] in ['8Ny9NjNpQQA', 'cyznGwlE9hM', 'esTcWwmykKQ','iA2Q4t-o58w','tz3zHV1Z5po','yDH9iAn82Q8', '0dkIbKXXFzI']:
                continue

            f = open(os.path.join(path, 'n_frames'), 'r')
            n_frames = int(f.read().rstrip('\n\r'))
            f.close()
            b, e = i.covering_ratio
            begin = round((n_frames - 1) * b)
            end = round((n_frames - 1) * e)
            if end - begin + 1 < self.duration or self.mode in ['validation', 'testing']:
                mid = (begin + end) // 2
                first = max(mid - self.duration // 2 + 1, 0)
                if first + self.duration > n_frames :
                    first = n_frames - self.duration
                self.sample_list.append({'path':path, 'first': first, 'label': i.num_label})
            else:
                first = begin
                while True:
                    if first + self.duration > n_frames:
                        break
                    self.sample_list.append({'path': path, 'first': first, 'label': i.num_label})
                    first += sample_interval

    def __getitem__(self, index):
        instance = self.sample_list[index]
        path = instance['path']
        first = instance['first']
        label = instance['label']
        name = path.split('/')[-1]
        images = load_images(path, list(range(first, first+self.duration)))
        # print('get:{}'.format(time.time()-start))
        if self.transform is not None:
            self.transform.randomize_parameters()
            images = [self.transform(img) for img in images]
            if self.kd:
                s_images = images[(len(images) - 1) // 2].clone()
        images = torch.stack(images, 1)
        images = images.squeeze(dim=0)
        if self.kd:
            return images, s_images, label, name

        return images, label, name

    def __len__(self):
        return len(self.sample_list)
