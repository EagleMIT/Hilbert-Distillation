from models_2d.resnet import resnet18, resnet50
from models_2d.mobilenet import MobileNetV2
from models_2d.vgg import vgg16_bn, vgg19_bn, vgg16
import torch

def generate_model(opt):
    assert opt.model in ['resnet18', 'resnet50']
    if opt.model == 'resnet18':
        if opt.pretrain:
            model = resnet18(pretrained=opt.pretrain)
            num_fc_ftr = model.fc.in_features
            model.fc = torch.nn.Linear(num_fc_ftr, opt.n_classes)
        else:
            model = resnet18(num_classes=opt.n_classes, pretrained=opt.pretrain)
    if opt.model == 'resnet50':
        if opt.pretrain:
            model = resnet50(pretrained=opt.pretrain)
            num_fc_ftr = model.fc.in_features
            model.fc = torch.nn.Linear(num_fc_ftr, opt.n_classes)
        else:
            model = resnet50(num_classes=opt.n_classes, pretrained=opt.pretrain)

    return model