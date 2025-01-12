import argparse
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
import tqdm

from torch.utils import data
from torchvision import datasets, transforms

import datasets as mydatasets
import init
import model
import resnet_models
import vgg_models
from utils import average_meter

from models import mlpnet


def get_optimizer(args, model):
    logging.info('Optimizer is {}'.format(args.optimizer))
    if args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                               momentum=args.momentum)
    else:
        raise NotImplementedError


def get_lr_scheduler(args, optimizer):
    logging.info('LR Scheduler is {}'.format(args.lr_scheduler))
    if args.lr_scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                               gamma=args.lr_gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones,
                                                    gamma=args.lr_gamma)
    else:
        raise NotImplementedError


def get_dataloaders(args):
    if args.dataset_name == 'MNIST':
        transform = transforms.ToTensor()
        trainset = datasets.MNIST(root=args.data_path, download=True, train=True, transform=transform)
        valset = datasets.MNIST(root=args.data_path, download=True, train=True, transform=transform)
        testset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    elif args.dataset_name == 'MNISTNorm':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST(root=args.data_path, download=True, train=True, transform=transform)
        valset = datasets.MNIST(root=args.data_path, download=True, train=True, transform=transform)
        testset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    elif args.dataset_name == 'CIFAR10':
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(cifar_mean, cifar_std)])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(cifar_mean, cifar_std)])
        trainset = datasets.CIFAR10(root=args.data_path, download=True, train=True, transform=train_transform)
        valset = datasets.CIFAR10(root=args.data_path, download=True, train=True, transform=test_transform)
        testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
    elif args.dataset_name == 'HeteroMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = mydatasets.HeteroMNIST(root=args.data_path, download=True, train=True, transform=transform,
                                          special_digit=args.hetero_special_digit,
                                          special_train_split=args.hetero_special_train,
                                          other_digits_train_split=args.hetero_other_digits_train_split)
        valset = mydatasets.HeteroMNIST(root=args.data_path, download=True, train=True, transform=transform,
                                        special_digit=args.hetero_special_digit,
                                        special_train_split=args.hetero_special_train,
                                        other_digits_train_split=args.hetero_other_digits_train_split)
        testset = mydatasets.HeteroMNIST(root=args.data_path, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError
    # Setting the seed for a consistent split
    random_seed = 543
    np.random.seed(random_seed)
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    valid_size = 0.1
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler)
    valloader = data.DataLoader(valset, batch_size=args.batch_size, sampler=valid_sampler)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    return trainloader, valloader, testloader


class Trainer:
    def __init__(self, train_init, args):
        self.args = args
        self.train_init = train_init
        self.trainloader, self.valloader, self.testloader = get_dataloaders(args)
        if args.model_name == 'FC':
            #self.model = model.FCModel(input_dim=args.input_dim, hidden_dims=args.hidden_dims,
            #                           output_dim=args.output_dim)
            self.model = mlpnet.MlpNetBase(input_dim=args.input_dim, num_classes=args.output_dim)
        elif args.model_name == 'Conv':
            input_channels = 1 if args.dataset_name == 'MNIST' else 3
            self.model = model.ConvModel(input_channels=input_channels,
                                         output_dim=10)
        elif args.model_name == 'vgg11':
            self.model = vgg_models.vgg11(num_classes=10)
        elif args.model_name == 'resnet18':
            self.model = resnet_models.resnet18(num_classes=10,
                                                use_max_pool=args.resnet_use_max_pool)
        elif args.model_name == 'ImageRNN':
            self.model = model.ImageRNN(n_outputs = args.output_dim, n_inputs = args.input_dim,
                                        n_steps = args.rnn_steps, n_neurons = args.hidden_dims,
                                        act_type = args.rnn_act_type, step_start = args.rnn_step_start)
        elif args.model_name == 'ImageLSTM':
            self.model = model.ImageLSTM(n_outputs=args.output_dim, n_inputs=args.input_dim,
                                         n_steps=args.rnn_steps, n_neurons=args.hidden_dims,
                                         step_start=args.rnn_step_start)
        else:
            raise NotImplementedError

        if args.evaluate or args.load_checkpoint:
            self.load_model()
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = get_optimizer(args, self.model)
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

    def train_epoch(self, epoch, tag='train'):
        if tag == 'train':
            dataloader = self.trainloader
            self.model.train()
        elif tag == 'val':
            dataloader = self.valloader
            self.model.eval()
        elif tag == 'test':
            dataloader = self.testloader
            self.model.eval()
        else:
            raise NotImplementedError

        tbar = tqdm.tqdm(dataloader)
        total = 0
        correct = 0
        loss_logger = average_meter.AverageMeter()
        for batch_idx, (images, labels) in enumerate(tbar):
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.model_name == 'FC':
                logits = self.model(images.view(images.size(0), -1))
            elif self.args.model_name in ['ImageRNN', 'ImageLSTM']:
              logits = self.model(images.squeeze())
            else:
                logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            prediction = torch.argmax(logits, dim=1)
            total += images.size(0)
            correct += torch.sum(labels == prediction)
            loss_logger.update(loss.item())

            if tag == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        accuracy = 100.0 * correct / total
        self.train_init.writer.add_scalar('Loss/' + tag, loss_logger.avg, epoch)
        self.train_init.writer.add_scalar('Accuracy/' + tag, accuracy, epoch)
        return accuracy

    def train(self):
        logging.info('Starting train.')
        best_val_acc = 0
        val_acc = self.train_epoch(0, tag='val')
        test_acc = 0

        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.args.num_epochs))
            _ = self.train_epoch(epoch, tag='train')
            val_acc = self.train_epoch(epoch, tag='val')
            test_acc = self.train_epoch(epoch, tag='test')
            self.lr_scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(self.train_init.model_path, 'best_val_acc_model.pth'.format(epoch))
                torch.save({'epoch': epoch,
                            'val_acc': val_acc,
                            'test_acc': test_acc,
                            'model_state_dict': self.model.state_dict(),
                            'config': self.model.get_model_config()},
                           save_path)

        save_path = os.path.join(self.train_init.model_path, 'final_model.pth')
        torch.save({'epoch': self.args.num_epochs,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'model_state_dict': self.model.state_dict(),
                    'config': self.model.get_model_config()},
                   save_path)

        logging.info('Training finished.')
        logging.info('Model saved at {}'.format(save_path))
        logging.info('Best acc val:{}, test:{}'.format(val_acc, test_acc))

    def evaluate(self):
        logging.info('Starting evaluation')
        val_acc = self.train_epoch(epoch=0, tag='val')
        test_acc = self.train_epoch(epoch=0, tag='test')
        logging.info('Validation acc:{}, Test acc:{}'.format(val_acc, test_acc))

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.checkpoint_path)['model_state_dict'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='FC')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--result_path', type=str, default='result')

    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'MultiStepLR'])
    parser.add_argument('--lr_step_size', type=int, default=10000)
    parser.add_argument('--lr_gamma', type=float, default=1.0)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
    parser.add_argument('--momentum', type=float, default=0)

    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[])
    parser.add_argument('--output_dim', type=int, default=10)

    parser.add_argument('--log_step', type=int, default=100,
                        help='The steps after which models would be logged.')

    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)

    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument("--seed", default=24601, type=int)

    parser.add_argument('--hetero_special_digit', type=int, default=4,
                        help='Special digit for heterogeneous MNIST')
    parser.add_argument('--hetero_special_train', default=False, action='store_true',
                        help='If HeteroMNIST with special digit split is used for training.')
    parser.add_argument('--hetero_other_digits_train_split', default=0.9, type=float)

    parser.add_argument('--resnet_use_max_pool', default=False, action='store_true')

    parser.add_argument('--rnn_steps', type=int, default=1,
                        help='Num of steps for which RNN model runs')
    parser.add_argument('--rnn_step_start', type=int, default=0,
                        help='Step number to start with for RNN experiments, helper flag')
    parser.add_argument('--rnn_act_type', type=str, default='tanh',
                        choices=['tanh', 'relu'])

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.gpu_id_list = [int(s) for s in args.gpu_ids.split(',')]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.basicConfig(level=logging.INFO)
    logging.info("Running training for {}".format(args.experiment_name))

    run_str = 'debug_seed_{}'.format(args.seed)
    if args.dataset_name == 'HeteroMNIST':
        run_params = ['special_dig', args.hetero_special_digit]
        if args.hetero_special_train:
            run_params.append('special_train')
        run_str += '_'.join([str(x) for x in run_params])
    if args.evaluate:
        args.resume = True
    train_init = init.Init(args=args, run_str=run_str)
    trainer = Trainer(train_init, args)
    if args.evaluate:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
