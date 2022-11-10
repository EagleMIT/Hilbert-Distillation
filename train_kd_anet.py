import torch
import torch.nn as nn
import argparse
import collections
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ANDataSet, spatial_transforms
from model.lrcn import ConvLstm
from utils.utils_action_recognition import save_setting_info, create_folder_dir_if_needed, save_loss_info_into_a_file, \
    set_project_folder_dir
from utils.training import test_model
from torch.utils.tensorboard import SummaryWriter
import os
from utils.teacher import ClassifierModel
from utils.kd_loss import BasicKD, hilbert_distillation

parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size (default:32)')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate (default:5e-4')
parser.add_argument('--num_workers', default=4, type=int,
                    help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
parser.add_argument('--duration', default=16, type=int,
                    help='The number of frames that would be sampled from each video (default:5)')
parser.add_argument('--seed', default=42, type=int,
                    help='initializes the pseudorandom number generator on the same number (default:42)')
parser.add_argument('--load_all_data_to_RAM', default=False, type=bool,
                    help='load dataset directly to the RAM, for faster computation. usually use when the num of class '
                         'is small (default:False')
parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
parser.add_argument('--hidden_size', default=256, type=int,
                    help="The number of features in the LSTM hidden state (default:256)")
parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
parser.add_argument('--bidirectional', default=True, type=bool, help='set the LSTM to be bidirectional (default:True)')
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project '
                         'dir, if debug the info would be saved in debug folder(default:True)')
parser.add_argument('--load_checkpoint', default=False, type=bool,
                    help='Loading a checkpoint and continue training with it')
parser.add_argument('--checkpoint_path', default='', type=str, help='Optional path to checkpoint model')
parser.add_argument('--checkpoint_interval', default=1, type=int, help='Interval between saving model checkpoints')
parser.add_argument('--val_check_interval', default=1, type=int, help='Interval between running validation test')
parser.add_argument('--local_dir', default='/data5/liuzhe/ckpt_lrcn/kd', help='The local directory of the project, setting where to '
                                                             'save the results of the run')
parser.add_argument('--number_of_classes', default=100, type=int, help='The number of classes we would train on')
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--beta', default=0.001, type=float)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main():
    # ====== set the run settings ======
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.local_dir)
    print('The setting of the run are:\n{}\n'.format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}'.format(folder_dir))
    save_setting_info(args, device, folder_dir)
    tensorboard_writer = SummaryWriter(folder_dir)

    print('Initializing Datasets and Dataloaders...')

    # ====== dataset ======
    spatial_transform_train_s = spatial_transforms.Compose([
        spatial_transforms.Resize(256),
        spatial_transforms.RandomCrop(224),
        spatial_transforms.RandomHorizontalFlip(),
        spatial_transforms.ToTensor(1),
        spatial_transforms.Normalize(mean=[114.7748, 107.7354, 99.475], std=[38.7568578, 37.88248729, 40.02898126])
    ])

    spatial_transform_train_t = spatial_transforms.Compose([
        spatial_transforms.RandomHorizontalFlip(),
        spatial_transforms.ToTensor(1),
        spatial_transforms.Normalize(mean=[114.7748, 107.7354, 99.475], std=[38.7568578, 37.88248729, 40.02898126])
    ])

    train_dataset = ANDataSet(
        '/data5/liuzhe/activitynet/frames',
        '/home/liuzhe/anet/annotation/activity_net.v1-2.min.json',
        duration=args.duration,
        transform=spatial_transform_train_s,
        mode='training',
        type='kd'
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    spatial_transform_test_s = spatial_transforms.Compose([
        spatial_transforms.Resize(224),
        spatial_transforms.ToTensor(1),
        spatial_transforms.Normalize(mean=[114.7748, 107.7354, 99.475], std=[38.7568578, 37.88248729, 40.02898126])
    ])

    test_dataset = ANDataSet(
        '/data5/liuzhe/activitynet/frames',
        '/home/liuzhe/anet/annotation/activity_net.v1-2.min.json',
        duration=args.duration,
        transform=spatial_transform_test_s,
        mode='validation',
        type='2D+lstm'
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # ======= if args.load_all_data_to_RAM True load dataset directly to the RAM (for faster computation) ======
    # if args.load_all_data_to_RAM:
    #     dataloaders = load_all_dataset_to_RAM(dataloaders, dataset_order, args.batch_size)

    print('Loading model...')
    num_class = args.number_of_classes
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    # ====== teacher model ===================================
    t_model = ClassifierModel.load_from_checkpoint('/data5/liuzhe/checkpoints/ckpt_ActivityNet_resnet_3/checkpoint_epoch=5-v2.ckpt')
    t_model = t_model.to(device)
    t_model.freeze()

    # ====== setting optimizer and criterion parameters ======
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    KDloss = BasicKD()
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    # ====== start training the model ======
    loss_hist = collections.deque(maxlen=500)
    cl_loss_hist = collections.deque(maxlen=500)
    kd_loss_hist = collections.deque(maxlen=500)
    for epoch in range(args.epochs):
        start_epoch = time.time()
        train_loss, train_acc = 0.0, 0.0
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar :
            for t_image, images, labels, ___ in train_dataloader :
                t_image, images, labels = t_image.to(device), images.to(device), labels.to(device)
                optimizer.zero_grad()  # zero the parameter gradients

                t_out, t_high = t_model.net(t_image)
                if hasattr(model, 'Lstm') :
                    model.Lstm.reset_hidden_state()

                output, s_high = model(images)

                # resize
                t_high = t_high.unsqueeze(1)
                t_high = t_high.repeat((1, s_high.shape[1], 1, 1, 1, 1))
                t_high = t_high.view(-1, t_high.shape[2],  t_high.shape[3],  t_high.shape[4], t_high.shape[5])
                s_high = s_high.view(-1, s_high.shape[2],  s_high.shape[3],  s_high.shape[4])

                cl_loss = criterion(output, labels)
                kd_loss = args.beta * hilbert_distillation(s_high, t_high)
                # loss = criterion(output, labels) + args.alpha * KDloss(output, t_out)
                loss = cl_loss + kd_loss
                # Accuracy calculation
                predicted_labels = output.detach().argmax(dim=1)
                acc = (predicted_labels == labels).cpu().numpy().sum()

                train_loss += loss.item()
                loss_hist.append(loss.item())
                cl_loss_hist.append(cl_loss.item())
                kd_loss_hist.append(kd_loss.item())
                train_acc += acc
                loss.backward()  # compute the gradients
                optimizer.step()  # update the parameters with the gradients
                pbar.set_postfix_str(
                    'Loss: {:1.3f}, CL Loss: {:1.3f}, KD Loss: {:1.3f}'.format(np.mean(loss_hist), np.mean(cl_loss_hist), np.mean(kd_loss_hist)))
                pbar.update(1)
        train_acc = 100 * (train_acc / train_dataloader.dataset.__len__())
        train_loss = train_loss / len(train_dataloader)

        if (epoch % args.val_check_interval) == 0:
            val_loss, val_acc, predicted_labels, images = test_model(model, test_dataloader, device, criterion)
            # plot_images_with_predicted_labels(images, label_decoder_dict, predicted_labels, folder_dir, epoch)
            end_epoch = time.time()
            # ====== print the status to the console and write it in tensorboard =======
            print('Epoch {} : Train loss {:.8f}, Train acc {:.3f}, Val loss {:.8f}, Val acc {:.3f}, epoch time {:.4f}'
                  .format(epoch, train_loss, train_acc, val_loss, val_acc, end_epoch - start_epoch))
            tensorboard_writer.add_scalars('train/val loss', {'train_loss': train_loss,
                                                              'val loss': val_loss}, epoch)
            tensorboard_writer.add_scalars('train/val accuracy', {'train_accuracy': train_acc,
                                                                  'val accuracy': val_acc}, epoch)
            # ====== save the loss and accuracy in txt file ======
            save_loss_info_into_a_file(train_loss, val_loss, train_acc, val_acc, folder_dir, epoch)
        if (epoch % args.checkpoint_interval) == 0:
            hp_dict = {'model_state_dict': model.state_dict()}
            save_model_dir = os.path.join(folder_dir, 'Saved_model_checkpoints')
            create_folder_dir_if_needed(save_model_dir)
            torch.save(hp_dict, os.path.join(save_model_dir, 'epoch_{}.pth.tar'.format(epoch)))


if __name__ == '__main__':
    main()
