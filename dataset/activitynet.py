"""
@Time   : 2021/06/16
@Author : Liu Zhe
@About  : ActivityNet Dataset
"""
from PIL import Image
import os
import torch
import os.path
import random
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


class Instance(object):
    """
    Representing an instance of activity in the videos
    """

    def __init__(self, idx, anno, vid_id, vid_info, name_num_mapping):
        self._starting, self._ending = anno['segment'][0], anno['segment'][1]
        self._str_label = anno['label']
        self._total_duration = vid_info['duration']
        self._idx = idx
        self._vid_id = vid_id
        self._file_path = None

        if name_num_mapping:
            self._num_label = name_num_mapping[self._str_label]

    @property
    def time_span(self):
        return self._starting, self._ending

    @property
    def covering_ratio(self):
        return self._starting / float(self._total_duration), self._ending / float(self._total_duration)

    @property
    def num_label(self):
        return self._num_label

    @property
    def label(self):
        return self._str_label

    @property
    def name(self):
        return '{}_{}'.format(self._vid_id, self._idx)

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This instance is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path


class Video(object):
    """
    This class represents one video in the activity-net db
    """
    def __init__(self, key, info, name_idx_mapping=None):
        self._id = key
        self._info_dict = info
        self._instances = [Instance(i, x, self._id, self._info_dict, name_idx_mapping)
                           for i, x in enumerate(self._info_dict['annotations'])]
        self._file_path = None

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._info_dict['url']

    @property
    def instances(self):
        return self._instances

    @property
    def duration(self):
        return self._info_dict['duration']

    @property
    def subset(self):
        return self._info_dict['subset']

    @property
    def instance(self):
        return self._instances

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This video is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path


class ANetDB(object):
    """
    This class is the abstraction of the activity-net db
    """

    _CONSTRUCTOR_LOCK = object()

    def __init__(self, token):
        """
        Disabled constructor
        :param token:
        :return:
        """
        if token is not self._CONSTRUCTOR_LOCK:
            raise ValueError("Use get_db to construct an instance, do not directly use the constructor")

    @classmethod
    def get_db(cls, file):
        """
        Build the internal representation of Activity Net databases
        We use the alphabetic order to transfer the label string to its numerical index in learning
        :param version:
        :return:
        """

        import os
        raw_db_file = file

        import json
        db_data = json.load(open(raw_db_file))

        me = cls(cls._CONSTRUCTOR_LOCK)
        me.version = raw_db_file.split('.')[1].replace('-', '.')
        me.prepare_data(db_data)

        return me

    def prepare_data(self, raw_db):
        self._version = raw_db['version']

        # deal with taxonomy
        self._taxonomy = raw_db['taxonomy']
        self._parse_taxonomy()

        self._database = raw_db['database']
        self._video_dict = {k: Video(k, v, self._name_idx_table) for k,v in self._database.items()}

        # split testing/training/validation set
        self._testing_dict = {k: v for k, v in self._video_dict.items() if v.subset == 'testing'}
        self._training_dict = {k: v for k, v in self._video_dict.items() if v.subset == 'training'}
        self._validation_dict = {k: v for k, v in self._video_dict.items() if v.subset == 'validation'}

        self._training_inst_dict = {i.name: i for v in self._training_dict.values() for i in v.instances}
        self._validation_inst_dict = {i.name: i for v in self._validation_dict.values() for i in v.instances}

    def get_subset_videos(self, subset_name):
        if subset_name == 'training':
            return self._training_dict.values()
        elif subset_name == 'validation':
            return self._validation_dict.values()
        elif subset_name == 'testing':
            return self._testing_dict.values()
        else:
            raise ValueError("Unknown subset {}".format(subset_name))

    def get_subset_instance(self, subset_name):
        if subset_name == 'training':
            return self._training_inst_dict.values()
        elif subset_name == 'validation':
            return self._validation_inst_dict.values()
        else:
            raise ValueError("Unknown subset {}".format(subset_name))

    def get_ordered_label_list(self):
        return [self._idx_name_table[x] for x in sorted(self._idx_name_table.keys())]

    def _parse_taxonomy(self):
        """
        This function just parse the taxonomy file
        It gives alphabetical ordered indices to the classes in competition
        :return:
        """
        name_dict = {x['nodeName']: x for x in self._taxonomy}
        parents = set()
        for x in self._taxonomy:
            parents.add(x['parentName'])

        # leaf nodes are those without any child
        leaf_nodes = [name_dict[x] for x
                      in list(set(name_dict.keys()).difference(parents))]
        sorted_lead_nodes = sorted(leaf_nodes, key=lambda l: l['nodeName'])
        self._idx_name_table = {i: e['nodeName'] for i, e in enumerate(sorted_lead_nodes)}
        self._name_idx_table = {e['nodeName']: i for i, e in enumerate(sorted_lead_nodes)}
        self._name_table = {x['nodeName']: x for x in sorted_lead_nodes}


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
        # with open(os.path.join(path, '%06d.jpg' % i), 'rb') as f:
        #     with Image.open(f) as img:
        #         images.append(img.convert('RGB'))
    return images


class ANDataSet(Dataset):
    def __init__(self, root_path, list_file, duration, transform=None, mode='training', type='kd', sample_interval=100, maximum_per_sample=10):
        assert mode in ['training', 'validation', 'testing'], 'mode must be one of train, val or test'
        assert type in ['3D', '2D+lstm', 'kd']
        self.dir_path = os.path.join(root_path, 'training' if mode == 'training' else 'validation')
        self.duration = duration
        self.transform = transform
        self.mode = mode
        self.type = type
        self.sample_interval = sample_interval if sample_interval > 0 else duration
        self.maximum_per_sample = maximum_per_sample
        self.db = ANetDB.get_db(list_file)
        self.list_name = str(mode) + '_' + str(duration) + '_' + str(sample_interval) + '_' + str(maximum_per_sample) + '.npy'
        if not os.path.isfile(self.list_name):
            self._make_sample_list()
        self.sample_list = np.load(self.list_name, allow_pickle=True)

    def _make_sample_list(self):
        instances_list = self.db.get_subset_instance('training' if self.mode == 'training' else 'validation')
        sample_list = []

        for i in tqdm(instances_list):
            path = os.path.join(self.dir_path, '_'+i.name[:11])
            if self.mode != 'training' and i.name[:11] in ['8Ny9NjNpQQA', 'cyznGwlE9hM', 'esTcWwmykKQ','iA2Q4t-o58w','tz3zHV1Z5po','yDH9iAn82Q8', '0dkIbKXXFzI']:
                continue
            f = open(os.path.join(path, 'n_frames'), 'r')
            n_frames = int(f.read().rstrip('\n\r'))
            f.close()
            b, e = i.covering_ratio
            begin = round((n_frames - 1) * b)
            start = begin
            end = round((n_frames - 1) * e)
            if end - begin > self.maximum_per_sample * self.sample_interval and self.maximum_per_sample != -1:
                interval = int((end - begin) / self.maximum_per_sample)
            else:
                interval = self.sample_interval

            if end - begin + 1 < self.duration or self.mode == 'validation':
                sample_list.append({'path':path, 'begin': begin, 'end': end, 'label': i.num_label, 'length': n_frames, 'start': start})
            else:
                while True:
                    if begin + interval > end:
                        break
                    sample_list.append({'path': path, 'begin': begin, 'end': begin + interval - 1, 'label': i.num_label, 'length': n_frames, 'start': start})
                    begin += interval
        np.save(self.list_name, sample_list)

    def __getitem__(self, index):
        instance = self.sample_list[index]
        path = instance['path']
        begin = instance['begin']
        end = instance['end']
        label = instance['label']
        n_frames = instance['length']
        start = instance['start']
        if self.mode == 'training':
            if end - begin + 1 <= self.duration:
                mid = begin + random.randint(0, end - begin)
                first = mid - self.duration // 2 + 1
            else:
                first = begin + random.randint(0, end - begin - self.duration + 1)
        else:
            first = (end + begin) // 2 - self.duration // 2 + 1
        name = path.split('/')[-1] + '_' + str(start)
        first = min(first, n_frames - self.duration)
        first = max(first, 0)

        images = load_images(path, list(range(first, first+self.duration)))

        if self.transform is not None :
            self.transform.randomize_parameters()
            images = [self.transform(img) for img in images]

        if self.type in ['3D', 'kd']:
            t_images = torch.stack(images, 1)
        if self.type in ['2D+lstm', 'kd']:
            s_images = torch.stack(images, 0)

        if self.type == 'kd':
            return t_images, s_images, label, name
        elif self.type == '3D':
            return t_images, label, name
        elif self.type == '2D+lstm':
            return s_images, label, name

    def __len__(self):
        return len(self.sample_list)


# class ANDataSet(Dataset):
#     def __init__(self, root_path, list_file, duration, transform_t=None, transform_s=None, mode='training', type='kd', sample_interval=100, maximum_per_sample=10):
#         assert mode in ['training', 'validation', 'testing'], 'mode must be one of train, val or test'
#         assert type in ['3D', '2D+lstm', 'kd']
#         self.dir_path = os.path.join(root_path, 'training' if mode == 'training' else 'validation')
#         self.duration = duration
#         self.transform_t = transform_t
#         self.transform_s = transform_s
#         self.mode = mode
#         self.type = type
#         self.sample_interval = sample_interval if sample_interval > 0 else duration
#         self.maximum_per_sample = maximum_per_sample
#         self.db = ANetDB.get_db(list_file)
#         self.list_name = str(mode) + '_' + str(duration) + '_' + str(sample_interval) + '_' + str(maximum_per_sample) + '.npy'
#         if not os.path.isfile(self.list_name):
#             self._make_sample_list()
#         self.sample_list = np.load(self.list_name, allow_pickle=True)
#
#     def _make_sample_list(self):
#         instances_list = self.db.get_subset_instance('training' if self.mode == 'training' else 'validation')
#         sample_list = []
#
#         for i in tqdm(instances_list):
#             path = os.path.join(self.dir_path, '_'+i.name[:11])
#             if self.mode != 'training' and i.name[:11] in ['8Ny9NjNpQQA', 'cyznGwlE9hM', 'esTcWwmykKQ','iA2Q4t-o58w','tz3zHV1Z5po','yDH9iAn82Q8', '0dkIbKXXFzI']:
#                 continue
#             f = open(os.path.join(path, 'n_frames'), 'r')
#             n_frames = int(f.read().rstrip('\n\r'))
#             f.close()
#             b, e = i.covering_ratio
#             begin = round((n_frames - 1) * b)
#             start = begin
#             end = round((n_frames - 1) * e)
#             if end - begin > self.maximum_per_sample * self.sample_interval and self.maximum_per_sample != -1:
#                 interval = int((end - begin) / self.maximum_per_sample)
#             else:
#                 interval = self.sample_interval
#
#             if end - begin + 1 < self.duration or self.mode == 'validation':
#                 sample_list.append({'path':path, 'begin': begin, 'end': end, 'label': i.num_label, 'length': n_frames, 'start': start})
#             else:
#                 while True:
#                     if begin + interval > end:
#                         break
#                     sample_list.append({'path': path, 'begin': begin, 'end': begin + interval - 1, 'label': i.num_label, 'length': n_frames, 'start': start})
#                     begin += interval
#         np.save(self.list_name, sample_list)
#
#     def visualize(self, path):
#         count_list = []
#         length_list = []
#         total_length_list = []
#         label_hist = []
#
#         instances_list = self.db.get_subset_instance(self.mode)
#         video_list = self.db.get_subset_videos(self.mode)
#
#         for i in instances_list:
#             path = os.path.join(self.dir_path, '_'+i.name[:11])
#             if self.mode != 'training' and i.name[:11] in ['8Ny9NjNpQQA', 'cyznGwlE9hM', 'esTcWwmykKQ','iA2Q4t-o58w','tz3zHV1Z5po','yDH9iAn82Q8', '0dkIbKXXFzI']:
#                 continue
#             f = open(os.path.join(path, 'n_frames'), 'r')
#             n_frames = int(f.read().rstrip('\n\r'))
#             f.close()
#             b, e = i.covering_ratio
#             begin = round((n_frames - 1) * b)
#             end = round((n_frames - 1) * e)
#             length_list.append(end - begin)
#             if end - begin > self.maximum_per_sample * self.sample_interval:
#                 interval = int((end - begin) / self.maximum_per_sample)
#             else:
#                 interval = self.sample_interval
#
#             if end - begin + 1 < self.duration or self.mode in ['validation', 'testing']:
#                 count = 1
#             else:
#                 count = int(end - begin) / interval
#
#             count_list.append(count)
#
#         for i in video_list:
#             path = os.path.join(self.dir_path, '_'+i.id)
#             if self.mode == 'validation' and i.id in ['8Ny9NjNpQQA', 'cyznGwlE9hM', 'esTcWwmykKQ','iA2Q4t-o58w','tz3zHV1Z5po','yDH9iAn82Q8', '0dkIbKXXFzI']:
#                 continue
#             f = open(os.path.join(path, 'n_frames'), 'r')
#             n_frames = int(f.read().rstrip('\n\r'))
#             f.close()
#             total_length_list.append(n_frames)
#
#         label_hist = [i['label'] for i in self.sample_list]
#
#         plt.figure(figsize=(15, 30))
#         a = plt.subplot(411)
#         a.set_title("num of samples per seg")
#         b = plt.subplot(412)
#         b.set_title("length of segs")
#         c = plt.subplot(413)
#         c.set_title("length of videos")
#         d = plt.subplot(414)
#         d.set_title("num of samples per category")
#         a.hist(count_list)
#         b.hist(length_list)
#         c.hist(total_length_list)
#         d.hist(label_hist, bins=np.arange(0, 101, 1))
#         plt.savefig(path)
#
#     def __getitem__(self, index):
#         instance = self.sample_list[index]
#         path = instance['path']
#         begin = instance['begin']
#         end = instance['end']
#         label = instance['label']
#         n_frames = instance['length']
#         start = instance['start']
#         if self.mode == 'training':
#             if end - begin + 1 <= self.duration:
#                 mid = begin + random.randint(0, end - begin)
#                 first = mid - self.duration // 2 + 1
#             else:
#                 first = begin + random.randint(0, end - begin - self.duration + 1)
#         else:
#             first = (end + begin) // 2 - self.duration // 2 + 1
#         name = path.split('/')[-1] + '_' + str(start)
#         first = min(first, n_frames - self.duration)
#         first = max(first, 0)
#
#         images = load_images(path, list(range(first, first+self.duration)))
#         if self.type in ['3D', 'kd']:
#             if self.transform_t is not None :
#                 self.transform_t.randomize_parameters()
#                 images_t = [self.transform_t(img) for img in images]
#             t_images = torch.stack(images_t, 1)
#         if self.type in ['2D+lstm', 'kd']:
#             if self.transform_s is not None :
#                 self.transform_s.randomize_parameters()
#                 if self.transform_t is not None:
#                     self.transform_s.transforms[2].p = self.transform_t.transforms[0].p
#                 images_s = [self.transform_s(img) for img in images]
#             s_images = torch.stack(images_s, 0)
#
#         if self.type == 'kd':
#             return t_images, s_images, label, name
#         elif self.type == '3D':
#             return t_images, label, name
#         elif self.type == '2D+lstm':
#             return s_images, label, name
#
#     def __len__(self):
#         return len(self.sample_list)

if __name__ == '__main__':
    try:
        from dataset.spatial_transforms import *
        from dataset.temporal_transforms import *
        from dataset.target_transforms import ClassLabel
    except:
        from spatial_transforms import *
        from temporal_transforms import *
        from target_transforms import ClassLabel
    scales = [1.0]
    crop_method = MultiScaleRandomCrop(scales, 112)
    spatial_transform = Compose([
        RandomHorizontalFlip(),
        ToTensor(255),
        # Normalize(mean=[114.7748, 107.7354, 99.475], std=[38.7568578, 37.88248729, 40.02898126])
    ])
    training_data = ANDataSet('/data5/liuzhe/activitynet/frames', '/home/liuzhe/anet/annotation/activity_net.v1-2.min.json', duration=16, transform=spatial_transform, mode='training', kd=True)
    print(len(training_data))
    # training_data.visualize("hist1.png")
    for i, label, S, N in tqdm(training_data) :

        pass

    # mean = torch.zeros(3)
    # std = torch.zeros(3)
    # for i, S, label, N in tqdm(training_data) :
    #     max = torch.max(i)
    #     min = torch.min(i)
    #     a, b = torch.std_mean(i, dim=(1,2,3))
    #     for d in range(3):
    #         mean[d] += i[d].mean()
    #         std[d] += i[d].std()
    #     mean.div_(len(training_data))
    #     std.div_(len(training_data))
    #     pass
    # print(mean.cpu().numpy())
    # print(std.cpu().numpy())
