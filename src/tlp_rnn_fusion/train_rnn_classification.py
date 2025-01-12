import torch
from tqdm import tqdm
import numpy as np
import os
import argparse
from datetime import datetime
import sys
sys.path.append('./')
sys.path.append('../')
import logging
from rnn_dataset_torch import HomogeneousMNIST,SSTDataset
from tlp_rnn_fusion.rnn_models import RNNWithDecoder,RNNWithEncoderDecoder, LSTMWithDecoder,LSTMWithEncoderDecoder
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import datasets, transforms
from permute_model import PermuteNN

RANDOM_SEED = 543

def get_dataloaders(exp_args):
    if exp_args.dataset_name == "MyMNIST":
        train_dataset = HomogeneousMNIST(exp_args.train_data_path)
        test_dataset = HomogeneousMNIST(exp_args.test_data_path)
    elif exp_args.dataset_name == "MNISTNorm":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root=exp_args.train_data_path, download=True, train=True, transform=transform)
        test_dataset = datasets.MNIST(root=exp_args.test_data_path, train=False, download=True, transform=transform)
    elif exp_args.dataset_name == "SST":
        train_dataset = SSTDataset(exp_args.train_data_path,exp_args.glove_path)
        test_dataset = SSTDataset(exp_args.train_data_path,exp_args.glove_path)
    else:
        raise NotImplementedError

    np.random.seed(RANDOM_SEED)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    valid_size = 0.1
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = DataLoader(train_dataset, batch_size=exp_args.batch_size, sampler=train_sampler)
    valloader = DataLoader(train_dataset, batch_size=exp_args.batch_size, sampler=valid_sampler)

    testloader = DataLoader(test_dataset, batch_size=exp_args.batch_size, shuffle=False)

    return trainloader,valloader,testloader

def get_optimizer(exp_args, model):
    print('Optimizer is {}'.format(exp_args.optimizer))
    if exp_args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=exp_args.learning_rate, weight_decay=exp_args.weight_decay)
    elif exp_args.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=exp_args.learning_rate, weight_decay=exp_args.weight_decay,
                               momentum=exp_args.momentum)
    else:
        raise NotImplementedError

class Trainer:

    def __init__(self,exp_args,model,trainloader,valloader,testloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.exp_args = exp_args
        # if exp_args.loss_fn == "cross_entropy":
        m = torch.nn.LogSoftmax(dim=1)
        loss =torch.nn.NLLLoss(reduction='mean') 
        def loss_func(last_output,y): 
            return(loss(m(last_output),y))
        self.loss_fn = loss_func
        # if accuracy_fn == "acc_classification":
        #     def accuracy_func(last_output,y):
        #         max_idx = torch.max(m(last_output),1)[1].int()
        #         return sum(max_idx==y) / y.size()[0]
        #     self.acc_fn = accuracy_func         
       
        self.optimizer = get_optimizer(exp_args,self.model)


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

        correct = 0
        loss = 0
        num_samples = 0

        for i_batch, samples_batched in enumerate(iter(dataloader)): # samples_batched - word sentences in a batch of sentences (batch_size,100)
            x_batched, y_batched = samples_batched 
            if self.exp_args.dataset_name == "MNISTNorm":
                x_batched = torch.squeeze(x_batched, 1)
            x_batched = x_batched.to(self.exp_args.device)
            y_batched = y_batched.to(self.exp_args.device)
            
            batch_size = x_batched.size(0)
            
            # init_hiddens = [torch.zeros(1,x_batched.size(0),hidden_dim).to(self.exp_args.device) for hidden_dim in self.model.channels[1:-1] ]
            # logits = self.model(x_batched,init_hiddens) 
            if tag == 'train':
                logits = self.model(x_batched) # logits size(batch_size,seq_len,vocab_size)
            else:
                with torch.no_grad():
                    logits = self.model(x_batched)
                    
            last_logits = logits[:,-1,:] # size(batch_size,vocab_size)
            
            batch_loss = self.loss_fn(last_logits,y_batched)
            loss += batch_loss
            num_samples += batch_size  

            prediction = torch.argmax(last_logits.detach(), dim=1) # size(batch_size)
            correct += torch.sum(y_batched == prediction)

            if tag == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

        accuracy = 100.0 * correct / num_samples
        print("Epoch:",epoch,",",tag,", accuracy:",accuracy)
        return accuracy


    def train(self):
        print('Starting train.')
        best_val_acc = 0
        val_acc = self.train_epoch(0, tag='val')
        test_acc = 0

        all_epoch_val_test = []

        for epoch in range(1,self.exp_args.num_epochs+1):
            print('Epoch {}/{}'.format(epoch, self.exp_args.num_epochs))
            _ = self.train_epoch(epoch, tag='train')
            val_acc = self.train_epoch(epoch, tag='val')
            test_acc = self.train_epoch(epoch, tag='test')

            # all_epoch_val_test.append([epoch,val_acc.item(),0])
            all_epoch_val_test.append([epoch,val_acc.item(),test_acc.item()])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(self.exp_args.model_save_path, 'best_val_acc_model.pth'.format(epoch))
                torch.save({'epoch': epoch,
                            'val_acc': val_acc,
                            'test_acc': test_acc,
                            'model_state_dict': self.model.state_dict(),
                            'config': self.model.get_model_config()},
                           save_path)

        save_path = os.path.join(self.exp_args.model_save_path, 'final_model.pth')
        torch.save({'epoch': self.exp_args.num_epochs,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'model_state_dict': self.model.state_dict(),
                    'config': self.model.get_model_config()},
                   save_path)

        print('Training finished.')
        print('Model saved at {}'.format(save_path))
        print('Best acc val:{}, test:{}'.format(val_acc, test_acc))
        try:
            np.savetxt(save_path+"all_epoch_val_test.csv",all_epoch_val_test,delimiter=',')
            print("all_epoch_val_test was saved successfully")
        except:
            print("all_epoch_val_test not saved successfully")


    def evaluate(self):
        print('Starting evaluation')
        val_acc = self.train_epoch(epoch=0, tag='val')
        test_acc = self.train_epoch(epoch=0, tag='test')
        print('Validation acc:{}, Test acc:{}'.format(val_acc, test_acc))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='rnn',choices=['rnn','lstm'],
                        help='name of models to fuse')
    parser.add_argument('--dataset_name', type=str, default='MNIST', choices=['SST','MyMNIST','MNISTNorm'],
                        help='name of the dataset to use')  
    parser.add_argument('--train_data_path', type=str, default='train.csv',
                        help='dataset path')       
    parser.add_argument('--test_data_path', type=str, default='train.csv',
                        help='dataset path')   
    parser.add_argument('--glove_path', type=str, default=None)   
                        



    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of iterations for trainer')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size for training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device for Pytorch to use')
    parser.add_argument('--optimizer', type=str, default='SGD',)
                        

    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='check whether to load a previously trained model or to initialize a new model')
    parser.add_argument('--model_load_path', type=str, default='',
                        help='path to load a model')
    parser.add_argument('--model_save_path', type=str, default='',
                        help='path to save a model')
    parser.add_argument('--permute_model', default=False, action='store_true')
    

    # arguments for simple RNN models
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='vocabulary size')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='embedding dimension of an RNN model')
    parser.add_argument('--hidden_dims', type=str, default=None,
                        help='list of hidden dimensions')
    parser.add_argument('--input_dim', type=int, default=None)

    parser.add_argument('--hidden_activations', type=str, default=None,
                        help='list of hidden activations')
    parser.add_argument('--encoder', default=False, action='store_true')

    parser.add_argument('--bias', action='store_true',
                        help='bias')
                        
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0)

    exp_args = parser.parse_args(sys.argv[1:])
    
    hidden_dims = [int(i) for i in exp_args.hidden_dims.strip('[]').split(',')]
    hidden_activations = None if exp_args.hidden_activations is None else exp_args.hidden_activations.strip('[]').split(',')

    """
    Set up model
    """
    print("exp_args.encoder",exp_args.encoder)
    if exp_args.model_name == "rnn" and exp_args.encoder:
        # dimensions of input, hidden, output must be specified
        model = RNNWithEncoderDecoder(output_dim = exp_args.vocab_size, input_dim=exp_args.input_dim,embed_dim=exp_args.embed_dim,hidden_dims=hidden_dims, hidden_activations=hidden_activations,bias= exp_args.bias)
    elif exp_args.model_name == "rnn" and not exp_args.encoder :
        # dimensions of input, hidden, output must be specified
        model = RNNWithDecoder(output_dim = exp_args.vocab_size, embed_dim=exp_args.embed_dim,hidden_dims=hidden_dims, hidden_activations=hidden_activations,bias= exp_args.bias)
    elif exp_args.model_name == "lstm" and exp_args.encoder:
        # dimensions of input, hidden, output must be specified
        model = LSTMWithEncoderDecoder(output_dim = exp_args.vocab_size, input_dim = exp_args.input_dim, embed_dim=exp_args.embed_dim,hidden_dims=hidden_dims, hidden_activations=hidden_activations,bias= exp_args.bias)
    elif exp_args.model_name == "lstm" and not exp_args.encoder:
        # dimensions of input, hidden, output must be specified
        model = LSTMWithDecoder(output_dim = exp_args.vocab_size, embed_dim=exp_args.embed_dim,hidden_dims=hidden_dims, hidden_activations=hidden_activations,bias= exp_args.bias)
    
    
    print(model.get_model_config)
    if exp_args.pretrained is True and os.path.isfile(exp_args.model_load_path):
        try:
            checkpoint = torch.load(exp_args.model_load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'----------finish loading model from path: {exp_args.model_load_path}----------')
        except: 
            print("Failed to load model, Unexpected error:", sys.exc_info()[0])
            pass

    if hasattr(exp_args, 'permute_model') and exp_args.permute_model:
            # Useful for testing fusion algorithms!
            permute_nn = PermuteNN(model)
            model = permute_nn.permute()

    model.to(exp_args.device)

    """
    Set up dataset loaders
    """
    trainloader,valloader,testloader = get_dataloaders(exp_args)

    # print("testing train loader")
    # number = 0
    # for i_batch, samples_batched in enumerate(iter(testloader)):
    #     x,y = samples_batched
    #     if i_batch < 4:
    #         print(x.size(),y.size())
    #     number += x.size(0)
    # print(number,"number")




    """
    Set up trainer
    """
    trainer = Trainer(
        exp_args = exp_args,
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
    )


    """
    start training 
    """
    if exp_args.evaluate:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()