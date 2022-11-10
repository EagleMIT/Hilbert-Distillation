import argparse
from torch.utils.data import DataLoader
from dataset import ANDataSet, spatial_transforms
import os
import torch
from model import resnet
from train import train

parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default:32)')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate (default:5e-4')
parser.add_argument('--num_workers', default=4, type=int,
                    help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
parser.add_argument('--duration', default=16, type=int,
                    help='The number of frames that would be sampled from each video (default:5)')
parser.add_argument('--seed', default=42, type=int,
                    help='initializes the pseudorandom number generator on the same number (default:42)')
parser.add_argument('--load_all_data_to_RAM', default=False, type=bool,
                    help='load dataset directly to the RAM, for faster computation. usually use when the num of class '
                         'is small (default:False')
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project '
                         'dir, if debug the info would be saved in debug folder(default:True)')
parser.add_argument('--load_checkpoint', default=False, type=bool,
                    help='Loading a checkpoint and continue training with it')
parser.add_argument('--checkpoint_path', default='', type=str, help='Optional path to checkpoint model')
parser.add_argument('--checkpoint_interval', default=20, type=int, help='Interval between saving model checkpoints')
parser.add_argument('--val_check_interval', default=1, type=int, help='Interval between running validation test')
parser.add_argument('--local_dir', default='/data5/liuzhe/ckpt_lrcn/3D', help='The local directory of the project, setting where to '
                                                             'save the results of the run')
parser.add_argument('--number_of_classes', default=100, type=int, help='The number of classes we would train on')

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def main():
    args = parser.parse_args()
    # ====== dataset ======
    spatial_transform_train = spatial_transforms.Compose([
        spatial_transforms.Resize(256),
        spatial_transforms.RandomCrop(224),
        spatial_transforms.RandomHorizontalFlip(),
        spatial_transforms.ToTensor(1),
        spatial_transforms.Normalize(mean=[114.7748, 107.7354, 99.475], std=[38.7568578, 37.88248729, 40.02898126])
    ])
    train_dataset = ANDataSet(
        '/data5/liuzhe/activitynet/frames',
        '/home/liuzhe/anet/annotation/activity_net.v1-2.min.json',
        duration=args.duration,
        transform=spatial_transform_train,
        mode='training',
        type='3D'
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    spatial_transform_test = spatial_transforms.Compose([
        spatial_transforms.Resize(224),
        spatial_transforms.ToTensor(1),
        spatial_transforms.Normalize(mean=[114.7748, 107.7354, 99.475], std=[38.7568578, 37.88248729, 40.02898126])
    ])
    test_dataset = ANDataSet(
        '/data5/liuzhe/activitynet/frames',
        '/home/liuzhe/anet/annotation/activity_net.v1-2.min.json',
        duration=args.duration,
        transform=spatial_transform_test,
        mode='validation',
        type='3D'
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # ======= if args.load_all_data_to_RAM True load dataset directly to the RAM (for faster computation) ======
    # if args.load_all_data_to_RAM:
    #     dataloaders = load_all_dataset_to_RAM(dataloaders, dataset_order, args.batch_size)

    print('Loading model...')
    model = resnet.resnet152(num_classes=args.number_of_classes, shortcut_type='B', sample_size=224, sample_duration=args.duration)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train(args, model, train_dataloader, test_dataloader, optimizer, scheduler)


if __name__ == '__main__':
    main()
