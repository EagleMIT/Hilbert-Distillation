"""
@Time   : 2021/04/03
@Author : Liu Zhe
@About  : 简要描述整个文件的用处
"""
import os
import torch
import argparse
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from torch.nn import CrossEntropyLoss
from model import resnet
from torch import nn

pl.seed_everything(123)
parser = argparse.ArgumentParser('3DCNN')
parser.add_argument('--gpu', type=list, default=['0', '1'])
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--config', type=str, default='./config/Mobilenet3D.yaml')

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

class BaseCLModel(LightningModule):
    def __init__(self):
        super(BaseCLModel, self).__init__()
        self.result = []
        self.metric = {}
        self.gt = {}

    def training_epoch_end(self, outputs):
        train_loss_mean = 0
        for output in outputs:
            train_loss_mean += output['loss']

        train_loss_mean /= len(outputs)

        # log training accuracy at the end of an epoch
        self.log('train_loss', train_loss_mean)

    # def validation_epoch_end(self, outputs):
    #     self.test_epoch_end(outputs)

    # def test_epoch_end(self, outputs):
    #     # calculate accuracy
    #     acc = torch.sum(self.metric['CORRECT']) / torch.sum(self.metric['COUNT'])
    #
    #     self.log('acc', acc)
    #     print('acc: {}'.format(acc.item()))
    #
    #     self.metric.clear()
    #
    # def measure(self, outputs, targets, names):
    #     n = outputs.shape[1]
    #     outputs = torch.argmax(outputs, dim=1)
    #     # rank = outputs.topk(k=3, dim=1)
    #     outputs = F.one_hot(outputs, num_classes=n)
    #     targets = F.one_hot(targets, num_classes=n)
    #     correct = torch.mul(outputs, targets)
    #     if bool(self.metric):
    #         self.metric['CORRECT'] += torch.sum(correct, dim=0).cpu()
    #         self.metric['COUNT'] += torch.sum(targets, dim=0).cpu()
    #     else:
    #         self.metric['CORRECT'] = torch.sum(correct, dim=0).cpu()
    #         self.metric['COUNT'] = torch.sum(targets, dim=0).cpu()

    def measure(self, outputs, targets, names):
        n = outputs.shape[1]
        outputs = torch.softmax(outputs, dim=1)
        # rank = outputs.topk(k=3, dim=1)
        for i in range(len(outputs)):
            if names[i] not in self.metric.keys():
                self.metric[names[i]] = outputs[i].cpu()
            else:
                self.metric[names[i]] += outputs[i].cpu()
            self.gt[names[i]] = targets[i].cpu()

    def test_epoch_end(self, outputs):
        correct = 0
        count = 0
        for k, v in self.metric.items():
            rank = v.topk(k=1).indices
            if (self.gt[k] == rank).any():
                correct += 1
            count += 1
        # calculate accuracy
        acc = correct / count

        self.log('acc', acc)
        print('acc: {}'.format(acc))

        self.metric.clear()
        self.gt.clear()


class ClassifierModel(BaseCLModel):
    def __init__(self, cfg):
        super(ClassifierModel, self).__init__()
        self.save_hyperparameters(cfg)
        self.net, self.parameter = generate_model(self.hparams)
        self.metric = torchmetrics.Accuracy()

    def forward(self, x):
        output, _ = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self.forward(inputs)
        loss = CrossEntropyLoss()(outputs, targets)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx) :
        if len(batch) == 4:
            _, inputs, targets, names = batch
        else:
            inputs, targets, names = batch
        outputs = self.forward(inputs)
        outputs = torch.softmax(outputs, dim=1)
        # pred = outputs.topk(k=1, dim=1).indices.squeeze(1)
        # return {'pred': pred, 'target': targets}
        self.metric(outputs, targets)
        # self.log('acc', self.metric, on_epoch=True)

    def validation_epoch_end(self, validation_step_outputs):
        # count = 0
        # correct = 0
        # for out in validation_step_outputs:
        #     pred = out['pred']
        #     targets = out['target']
        #     count += len(targets)
        #     correct += torch.sum((pred == targets).int())
        # acc = correct / count
        # self.log('acc', acc)
        # print('acc: {}'.format(acc))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        acc = self.metric.compute()
        print('acc: {}'.format(acc))
        self.log('acc', acc)

    def test_step(self, batch, batch_idx):
        inputs, targets, name = batch
        outputs = self.forward(inputs)
        self.measure(outputs, targets, name)

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameter, lr=self.hparams.LR)
        # scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.EPOCHS, eta_min=1e-6),
        #              'interval': 'epoch',
        #              'frequency': 1}
        opt = torch.optim.SGD(self.parameter, self.hparams.LR)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min'),
                     'interval': 'epoch',
                     'frequency': 1,
                     'monitor': 'train_loss'
                     }
        return [opt], [scheduler]

def generate_model(opt):
    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from model.resnet import get_fine_tuning_parameters
        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    if opt.pretrain_path:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
        opt.arch = '{}'.format(opt.model)
        assert opt.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'])

        if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
            model.module.classifier = nn.Sequential(
                            nn.Dropout(0.9),
                            nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
            model.module.classifier = model.module.classifier.cuda()
        elif opt.model == 'squeezenet':
            model.module.classifier = nn.Sequential(
                            nn.Dropout(p=0.5),
                            nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                            nn.ReLU(inplace=True),
                            nn.AvgPool3d((1,4,4), stride=1))
            model.module.classifier = model.module.classifier.cuda()
        else:
            model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_portion)
        return model.module, parameters

    return model, model.parameters()