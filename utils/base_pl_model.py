import torch
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule


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

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def measure(self, outputs, targets, names):
        n = outputs.shape[1]
        outputs = torch.softmax(outputs, dim=1)
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

