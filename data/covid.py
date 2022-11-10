import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize, Normalize, RandomRotation
import matplotlib.pyplot as plt

DIRPATH = {
    'Normal': '/data/covid/curated_data/1NonCOVID',
    'COVID-19': '/data/covid/curated_data/2COVID',
    'CAP': '/data/covid/curated_data/3CAP'
}

CLASS_TO_LABEL = {
    'Normal': 0,
    'COVID-19': 1,
    'CAP': 2
}


def load_image(paths):
    images = []
    for p in paths:
        img = cv2.imread(p, flags=cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images


def read_info(path):
    d = pd.read_csv(
        filepath_or_buffer=path,
        encoding='ISO-8859-1'
    )
    data = {}
    if 'Slice' in d.columns:
        slice_key = 'Slice'
    else:
        slice_key = 'Slices_x'
    for _, i in d.iterrows():
        p_id = i['Patient ID']
        filename = i['File name']
        classname = i['Diagnosis']
        try:
            slice = int(i[slice_key])
        except:
            slice = -1
        if p_id not in data.keys():
            data[p_id] = [(slice, filename, classname)]
        else:
            data[p_id].append((slice, filename, classname))

    parted = []
    for k, v in data.items():
        v.sort(key=lambda x: x[0])
        i = None
        patch = []
        for s in v:
            if i is not None:
                if s[0] - i[0] > 9:
                    parted.append({
                        'Patient ID': k,
                        'Slices': patch
                    })
                    patch = [s]
                elif i[0] < 0:
                    parted.append({
                        'Patient ID': k,
                        'Slices': [i]
                    })
                else:
                    patch.append(s)
            else:
                patch.append(s)
            i = s
        parted.append({
            'Patient ID': k,
            'Slices': patch
        })
    return parted


class covid(Dataset):
    def __init__(self, mode='train', transform=None, min_length=8, duration=8, step=4, kd=False):
        self.mode = mode
        self.transform = transform
        self.kd = kd
        if mode == 'test' and kd:
            self.duration = 1
        else:
            self.duration = duration
        self.sample_list = make_dataset(
            mode=mode,
            min_length=min_length,
            sampling_length=self.duration,
            sampling_step=step
        )
        pass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        sample = self.sample_list[item]
        paths = [os.path.join(DIRPATH[sample['class']], f) for f in sample['filename']]
        images = np.stack(load_image(paths), axis=-1)
        label = CLASS_TO_LABEL[sample['class']]
        p_id = sample['patient']
        if self.transform:
            images = self.transform(images)
        images = torch.stack([images]*3, dim=0).squeeze(1)
        if self.kd and self.mode == 'train':
            s_image = images[:,(self.duration-1)//2,...]
            return images, s_image, label, p_id
        return images, label, p_id


def make_dataset(mode='train', min_length=8, sampling_length=8, sampling_step=4):
    assert min_length >= sampling_length
    covid = [i for i in read_info('/data/covid/meta_data_covid.csv') if len(i['Slices']) >= min_length]
    normal = [i for i in read_info('/data/covid/meta_data_normal.csv') if len(i['Slices']) >= min_length]
    cap = [i for i in read_info('/data/covid/meta_data_cap.csv') if len(i['Slices']) >= min_length]
    patch_list = []
    patch_list.extend(covid)
    patch_list.extend(normal)
    patch_list.extend(cap)
    test_list = patch_list[4:len(patch_list):5]
    train_list = [i for i in patch_list if i not in test_list]
    sample_list = []
    for p in train_list if mode == 'train' else test_list:
        first = 0
        while first + sampling_length <= len(p['Slices']):
            sample_list.append({
                'filename': [i[1] for i in p['Slices'][first: first + sampling_length]],
                'class': p['Slices'][0][2],
                'patient': p['Patient ID']
            })
            first += sampling_step
    return sample_list


transforms_covid_train = Compose([
    ToTensor(),
    Resize(size=(224, 224)),
    Normalize([0.5], [0.5]),
    # RandomRotation(degrees=10),
    RandomHorizontalFlip()
])

transforms_covid_test = Compose([
    ToTensor(),
    Resize(size=(224, 224)),
    Normalize([0.5], [0.5]),
])
