# Hilbert Distillation for Cross-Dimensionality Networks
The code repository of our NeurIPS 2022 paper "Hilbert Distillation for Cross-Dimensionality Networks":

Openreview: https://openreview.net/forum?id=kZnGYt-3f_X

arXiv: https://arxiv.org/abs/2211.04031

# Too long; didn't read
If you are only interested in the implementation of Hilbert Distillation method, please refer to the method 
hilbert_distillation in ```tils\kd_loss.py``` (line 55-77). Feel free to transplant them to your own projects.

If you are also interested in the experiment environments of this paper, please check the following information.
# Run Book
## Requirements
This code is built with Pytorch (for ActivityNet) and Pytorch-lightning (for Large-COVID-19). The key dependencies are as follows:  
```
pandas==1.1.3
tqdm==4.50.2
six==1.15.0
matplotlib==3.3.2
numpy==1.19.2
torch==1.8.0+cu111
nibabel==3.2.1
torchvision==0.9.0+cu111
torchmetrics==0.4.0
opencv_python==4.5.2.54
pytorch_lightning==1.3.7
Pillow==8.4.0
PyYAML==6.0
hilbertcurve==2.0.5
```
Please refer to requirement.txt for entire environment.

## Dataset

### Large-COVID-19
Download data [here](https://www.kaggle.com/maedemaftouni/large-covid19-ct-slice-dataset)

### ActivityNet
Download data [here](http://activity-net.org/)

## Running

### on Large-COVID-19
step 1: run ```train_covid.py``` to train teacher networks

step 2: run```train_kd_covid.py``` for Cross-Dimensionality distillation

☆☆☆ To keep the readability, we experimentally adopt the Pytorch-lightning architecture. It is recommended to reconstruct part of the Pytorch-lightning architecture according to [official document](https://pytorch-lightning.readthedocs.io/en/stable/)
if you use a higher version, as a large number of incremental updates between versions can lead to inconsistent model performance.
Before knowledge distillation, a well-trained teacher model is required. 

### on ActivityNet
step 1: run ```script\build_of.py``` to preprocess the data

step 2: run ```train_3d_anet.py``` to train teacher networks

step 3: run```train_kd_anet.py``` for Cross-Dimensionality distillation

Please refer to the paper for more details.
