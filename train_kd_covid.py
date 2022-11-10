import os
import torch
import argparse
import torchmetrics
from models_2d import resnet50
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from train_covid import ClassifierModel
from torch.utils.data import DataLoader
from utils.base_pl_model import BaseCLModel
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from config.config import load_cfg
from utils.kd_loss import basic_distillation, hilbert_distillation, variable_length_hilbert_distillation
from data import generate_dataset

pl.seed_everything(123)
parser = argparse.ArgumentParser('KD')
parser.add_argument('--gpu', type=list, default=['0'])
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--config', type=str, default='./config/covid_kd.yaml')

# get config
args = parser.parse_args()
cfg = load_cfg(args.config)


class ClassifierKDModel(BaseCLModel) :
    def __init__(self, cfg):
        super(ClassifierKDModel, self).__init__()
        self.save_hyperparameters(cfg)
        self.num_class = self.hparams.n_classes
        self.t_net = ClassifierModel.load_from_checkpoint(checkpoint_path='/checkpoint/last.ckpt')
        self.t_net.freeze()
        self.net = resnet50(num_classes=self.hparams.n_classes)
        self.metric = torchmetrics.Accuracy()

    def forward(self, x):
        output = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs, inputs_2d, targets, _ = batch
        self.t_net.eval()
        t_out, t_high = self.t_net.net(inputs)
        outputs, high, = self.forward(inputs_2d)
        bce_loss = CrossEntropyLoss()(outputs, targets)

        kd_loss = variable_length_hilbert_distillation(high, t_high)
        # basic_kd_loss = basic_distillation(outputs, t_out)
        loss = bce_loss + self.hparams.alpha * kd_loss
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            _, inputs, targets, names = batch
        else:
            inputs, targets, names = batch
        outputs, high = self.forward(inputs)
        outputs = torch.softmax(outputs, dim=1)
        self.metric(outputs, targets)

    def validation_epoch_end(self, validation_step_outputs):
        acc = self.metric.compute()
        print('acc: {}'.format(acc))
        self.log('acc', acc)

    def test_step(self, batch, batch_idx) :
        if len(batch) == 4:
            _, inputs, targets, names = batch
        else:
            inputs, targets, names = batch
        outputs, high = self.forward(inputs)

        self.measure(outputs, targets, names)

    def train_dataloader(self):
        dataset = generate_dataset(self.hparams, 'train')
        return DataLoader(dataset, batch_size=self.hparams.Batch_Size, num_workers=16, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        dataset = generate_dataset(self.hparams, 'test')
        return DataLoader(dataset, batch_size=self.hparams.Batch_Size, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        dataset = generate_dataset(self.hparams, 'val')
        return DataLoader(dataset, batch_size=self.hparams.Batch_Size, num_workers=8, pin_memory=True)

    def configure_optimizers(self) :
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.LR)
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.EPOCHS, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]


def main() :
    args = parser.parse_args()
    model = ClassifierKDModel(cfg)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.MODEL_SAVE_DIR, 'ckpt_{}_{}'.format(cfg.DATASET, cfg.model)),
        filename='checkpoint_{epoch}',
        save_top_k=-1,
        every_n_val_epochs=1
    )
    logger = TensorBoardLogger(cfg.LOG_SAVE_DIR, name='{}_{}'.format(cfg.DATASET, cfg.model))
    trainer = Trainer(max_epochs=cfg.EPOCHS, gpus=[int(i) for i in args.gpu],
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      accelerator='ddp',
                      check_val_every_n_epoch=1,
                      plugins=DDPPlugin(find_unused_parameters=False)
    )
    trainer.fit(model)


def test() :
    args = parser.parse_args()
    dirpath = os.path.join(cfg.MODEL_SAVE_DIR, 'ckpt_{}'.format(cfg.model))
    model = ClassifierKDModel.load_from_checkpoint(checkpoint_path=os.path.join(dirpath, 'last.ckpt'))
    trainer = Trainer(gpus=[int(i) for i in args.gpu])
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()
