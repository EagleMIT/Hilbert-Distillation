from .spatial_transforms import *
from .target_transforms import ClassLabel, VideoID
from .target_transforms import Compose as TargetCompose
from .temporal_transforms import *
from .anet import ANDataSet
from .covid import covid, transforms_covid_train, transforms_covid_test


def generate_dataset(opt, mode='train'):
    if opt.DATASET == 'ActivityNet':
        if mode == 'train':
            spatial_transform = Compose([
                RandomHorizontalFlip(),
                # MultiScaleRandomCrop([0.7, 0.8, 0.9], opt.sample_size),
                ToTensor(opt.norm_value),
                Normalize(opt.mean, opt.std)
            ])
            dataset = ANDataSet(
                '/data/activitynet/frames',
                '/CDKD/annotation/activity_net.v1-2.min.json',
                duration=opt.sample_duration,
                transform=spatial_transform,
                mode='training',
                kd=opt.kd
            )
        elif mode == 'val':
            spatial_transform = Compose([
                # CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value),
                Normalize(opt.mean, opt.std)
            ])
            dataset = ANDataSet(
                '/data/activitynet/frames',
                '/CDKD/annotation/activity_net.v1-2.min.json',
                duration=opt.sample_duration,
                transform=spatial_transform,
                mode='validation',
                kd=opt.kd
            )
        elif mode == 'test':
            spatial_transform = Compose([
                # CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value),
                Normalize(opt.mean, opt.std)
            ])
            dataset = ANDataSet(
                '/data/activitynet/frames',
                '/CDKD/annotation/activity_net.v1-2.min.json',
                duration=opt.sample_duration,
                transform=spatial_transform,
                mode='validation',
                kd=opt.kd
            )
    elif opt.DATASET == 'covid':
        if mode == 'train':
            dataset = covid(mode='train', transform=transforms_covid_train, min_length=8, duration=opt.sample_duration, step=opt.step, kd=opt.kd)
        else:
            dataset = covid(mode='test',transform=transforms_covid_test, min_length=8, duration=opt.sample_duration, step=opt.test_step, kd=opt.kd)
    return dataset