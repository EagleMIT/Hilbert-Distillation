"""
@Time   : 2022/04/29
@Author : Liu Zhe
@About  : 简要描述整个文件的用处
"""
import torch
from tqdm import tqdm


def foward_step(model, images, labels, criterion, mode=''):  # predections
    # Must be done before you run a new batch. Otherwise the LSTM will treat a new batch as a continuation of a sequence
    if hasattr(model, 'Lstm'):
        model.Lstm.reset_hidden_state()
    if mode == 'test':
        with torch.no_grad():
            output = model(images)
    else:
        output = model(images)
    if isinstance(output, tuple) :
        output = output[0]
    loss = criterion(output, labels)
    # Accuracy calculation
    predicted_labels = output.detach().argmax(dim=1)
    acc = (predicted_labels == labels).cpu().numpy().sum()
    return loss, acc, predicted_labels.cpu()


def train_model(model, dataloader, device, optimizer, criterion):
    train_loss, train_acc = 0.0, 0.0
    model.train()
    with tqdm(total=len(dataloader)) as pbar:
        for local_images, local_labels, ___ in dataloader:
            local_images = local_images.transpose(1, 2).contiguous()
            local_images, local_labels = local_images.to(device), local_labels.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            loss, acc, ___ = foward_step(model, local_images, local_labels, criterion, mode='train')
            train_loss += loss.item()
            train_acc += acc
            loss.backward()  # compute the gradients
            optimizer.step()  # update the parameters with the gradients
            pbar.update(1)
    train_acc = 100 * (train_acc / dataloader.dataset.__len__())
    train_loss = train_loss / len(dataloader)
    return train_loss, train_acc

def test_model(model, dataloader, device, criterion, mode='test'):
    val_loss, val_acc = 0.0, 0.0
    model.eval()
    if mode == 'save_prediction_label_list':
        prediction_labels_list = []
        true_labels_list = []
    with tqdm(total=len(dataloader)) as pbar:
        for local_images, local_labels, indexs in dataloader:
                local_images, local_labels = local_images.to(device), local_labels.to(device)
                loss, acc, predicted_labels = foward_step(model, local_images, local_labels, criterion, mode='test')
                if mode == 'save_prediction_label_list':
                    prediction_labels_list += [predicted_labels.detach().cpu()]
                    true_labels_list += [local_labels.detach().cpu()]
                val_loss += loss.item()
                val_acc += acc
                pbar.update(1)
    val_acc = 100 * (val_acc / dataloader.dataset.__len__())
    val_loss = val_loss / len(dataloader)
    if mode == 'save_prediction_label_list':
        return val_loss, val_acc, prediction_labels_list, local_images.cpu(), true_labels_list, indexs
    else:
        return val_loss, val_acc, predicted_labels, local_images.cpu()