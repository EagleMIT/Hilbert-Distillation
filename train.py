import torch
import torch.nn as nn
import argparse
import collections
import time
import numpy as np
from tqdm import tqdm
from utils.utils_action_recognition import save_setting_info, create_folder_dir_if_needed, save_loss_info_into_a_file, \
    set_project_folder_dir
from utils.training import train_model, test_model, foward_step
from torch.utils.tensorboard import SummaryWriter
import os


def train(args, model, train_dataloader, test_dataloader, optimizer, scheduler):
    # ====== set the run settings ======
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.local_dir)
    print('The setting of the run are:\n{}\n'.format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}'.format(folder_dir))
    save_setting_info(args, device, folder_dir)
    tensorboard_writer = SummaryWriter(folder_dir)

    # ======= if args.load_all_data_to_RAM True load dataset directly to the RAM (for faster computation) ======
    # if args.load_all_data_to_RAM:
    #     dataloaders = load_all_dataset_to_RAM(dataloaders, dataset_order, args.batch_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    # ====== start training the model ======
    loss_hist = collections.deque(maxlen=500)
    for epoch in range(args.epochs):
        start_epoch = time.time()
        train_loss, train_acc = 0.0, 0.0
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar :
            for images, labels, _ in train_dataloader :
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()  # zero the parameter gradients
                # Reset Lstm
                if hasattr(model, 'Lstm') :
                    model.Lstm.reset_hidden_state()
                output = model(images)
                if isinstance(output, tuple) :
                    output = output[0]
                # Loss calculation
                loss = criterion(output, labels)
                train_loss += loss.item()
                loss_hist.append(loss.item())
                # Accuracy calculation
                predicted_labels = output.detach().argmax(dim=1)
                acc = (predicted_labels == labels).cpu().numpy().sum()
                train_acc += acc
                # Backward
                loss.backward()  # compute the gradients
                optimizer.step()  # update the parameters with the gradients
                scheduler.step()

                pbar.set_postfix_str('Loss: {:1.3f}'.format(np.mean(loss_hist)))
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
