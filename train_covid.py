import os
import torch
import argparse
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models import generate_model
from torch.utils.data import DataLoader
from utils.base_pl_model import BaseCLModel
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss
from config.config import load_cfg
from data import generate_dataset
from pytorch_lightning.plugins import DDPPlugin

pl.seed_everything(123)
parser = argparse.ArgumentParser('3DCNN')
parser.add_argument('--gpu', type=list, default=['0'])
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--config', type=str, default='./config/covid3D.yaml')


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

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            _, inputs, targets, names = batch
        else:
            inputs, targets, names = batch
        outputs = self.forward(inputs)
        outputs = torch.softmax(outputs, dim=1)
        self.metric(outputs, targets)

    def validation_epoch_end(self, validation_step_outputs):
        acc = self.metric.compute()
        print('acc: {}'.format(acc))
        self.log('acc', acc)

    def test_step(self, batch, batch_idx):
        inputs, targets, name = batch
        outputs = self.forward(inputs)
        self.measure(outputs, targets, name)

    def train_dataloader(self):
        dataset = generate_dataset(self.hparams, 'train')
        return DataLoader(dataset, batch_size=self.hparams.Batch_Size, num_workers=32, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        dataset = generate_dataset(self.hparams, 'test')
        return DataLoader(dataset, batch_size=self.hparams.Batch_Size, num_workers=24, pin_memory=True)

    def val_dataloader(self):
        dataset = generate_dataset(self.hparams, 'val')
        return DataLoader(dataset, batch_size=self.hparams.Batch_Size, num_workers=24, pin_memory=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameter, lr=self.hparams.LR)
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.EPOCHS, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]


def main():
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    model = ClassifierModel(cfg)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.MODEL_SAVE_DIR, 'ckpt_{}_{}_{}'.format(cfg.DATASET, cfg.model, cfg.Dimension)),
        filename='checkpoint_{epoch}',
        save_top_k=-1,
        save_last=True,
        monitor='acc',
        mode='max',
        every_n_val_epochs=1
    )

    trainer = Trainer(
        max_epochs=cfg.EPOCHS,
        gpus=[int(i) for i in args.gpu],
        callbacks=[checkpoint_callback],
        logger = TensorBoardLogger(cfg.LOG_SAVE_DIR, name='{}_{}_{}'.format(cfg.DATASET, cfg.model, cfg.Dimension)),
        check_val_every_n_epoch=1,
    )
    trainer.fit(model)


def test():
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    dirpath = os.path.join(cfg.MODEL_SAVE_DIR, 'ckpt_{}_{}_{}'.format(cfg.DATASET, cfg.model, cfg.Dimension))
    model = ClassifierModel.load_from_checkpoint(checkpoint_path=os.path.join(dirpath, 'last.ckpt'))
    trainer = Trainer(gpus=[int(i) for i in args.gpu])
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()
