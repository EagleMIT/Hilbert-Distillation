import os
import torch
import argparse
from tqdm import tqdm
from model.lrcn import ConvLstm
from torch.utils.data import DataLoader
from dataset import ANDataSet, spatial_transforms

parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size (default:32)')
parser.add_argument('--num_workers', default=16, type=int,
                    help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
parser.add_argument('--duration', default=16, type=int,
                    help='The number of frames that would be sampled from each video (default:5)')
parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
parser.add_argument('--hidden_size', default=256, type=int,
                    help="The number of features in the LSTM hidden state (default:256)")
parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
parser.add_argument('--bidirectional', default=True, type=bool, help='set the LSTM to be bidirectional (default:True)')
parser.add_argument('--checkpoint_path', default='/data5/liuzhe/ckpt_lrcn/kd/20220518-022433-nokd/Saved_model_checkpoints/epoch_39.pth.tar', type=str, help='Optional path to checkpoint model')
parser.add_argument('--number_of_classes', default=100, type=int, help='The number of classes we would train on')

# /home/liuzhe/Projects/LRCN/20220501-101914/Saved_model_checkpoints/epoch_60.pth.tar 2D

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, args.number_of_classes)

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
        mode='testing',
        type='2D+lstm',
        # sample_interval=-1,
        # maximum_per_sample=10
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)
    print('loading model...')
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_acc = 0.0
    predicted_dict = {}
    label_dict = {}
    # vote_dict = {}

    with tqdm(total=len(test_dataloader)) as pbar:
        for image, labels, indexs in test_dataloader:
            image, labels = image.to(device), labels.to(device)
            model.Lstm.reset_hidden_state()
            with torch.no_grad():
                output = model(image) # shape: (batch size, num classes)
                # Accuracy calculation
            if isinstance(output, tuple):
                output = output[0]
            predicted_labels = output.detach().argmax(dim=1) # shape: (num classes)
            acc = (predicted_labels == labels).cpu().numpy().sum()
            test_acc += acc
            # onehot = torch.nn.functional.one_hot(predicted_labels, output.shape[1])
            for i in range(output.shape[0]):
                if indexs[i] in predicted_dict.keys():
                    predicted_dict[indexs[i]] += output[i]
                    # vote_dict[indexs[i]] += onehot[i]
                else:
                    predicted_dict[indexs[i]] = output[i]
                    label_dict[indexs[i]] = labels[i]
                    # vote_dict[indexs[i]] = onehot[i]
            pbar.update(1)

    test_acc = 100 * (test_acc / test_dataset.__len__())

    print('Mean Accuracy : %f' % test_acc)
    # vote
    test_acc_2 = 0.0
    for k, v in predicted_dict.items():
        predict = v.argmax()
        acc = (predict == label_dict[k]).cpu().item()
        test_acc_2 += acc

    test_acc_2 = 100 * (test_acc_2 / len(predicted_dict))
    print('Accuracy per fragment (voted by logits) : %f' % test_acc_2)
    #
    # test_acc_3 = 0.0
    # for k, v in vote_dict.items():
    #     predict = v.argmax()
    #     acc = (predict == label_dict[k]).cpu().item()
    #     test_acc_3 += acc
    #
    # test_acc_3 = 100 * (test_acc_3 / len(vote_dict))
    # print('Accuracy per fragment (voted by one-hot): %f' % test_acc_3)


if __name__ == '__main__':
    main()

    # Accuracy per fragment (voted by logits) : 55.586420

