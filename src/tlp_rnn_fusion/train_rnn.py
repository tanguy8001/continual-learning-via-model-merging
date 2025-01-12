import time
from numpy.core.numeric import NaN
from numpy.ma.extras import isin
import torch
from tqdm import tqdm
import numpy as np
import os
import argparse
from datetime import datetime
import random
import sys
sys.path.append('./')
sys.path.append('../')
import pickle
import csv

from tlp_rnn_fusion.rnn_models import RNNModel
from tlp_rnn_fusion.rnn_dataset import RNNDataset

class Trainer:

    def __init__(
            self,
            model,
            dataset,
            loss_fn,
            accuracy_fn,
            steps_per_epoch=100,
            learning_rate=1e-3,
            batch_size=2,
            eval_batch_size=8,
            device='cuda'
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.dataset = dataset
        if loss_fn == "cross_entropy":
            m = torch.nn.LogSoftmax(dim=1)
            loss =torch.nn.NLLLoss() 
            def loss_func(last_output,y): 
                return(loss(m(last_output),y))
            self.loss_fn = loss_func
        if accuracy_fn == "acc_classification":
            def accuracy_func(last_output,y):
                max_idx = torch.max(m(last_output),1)[1].int()
                return sum(max_idx==y) / y.size()[0]
            self.acc_fn = accuracy_func         
       
        self.steps_per_epoch = steps_per_epoch
        self.eval_batch_size = eval_batch_size

        self.optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.diagnostics = {'Num Batches': 0}

    def get_loss(self, x, y,return_acc=True):
        init_hiddens = [torch.zeros(1,x.size(1),hidden_dim).to(self.device) for hidden_dim in self.model.channels[1:-1] ]
        hiddens,outputs = self.model(x,init_hiddens)
        # print("outputs",outputs.size())
        loss = self.loss_fn(outputs[-1,:,:],y)
        if return_acc:
            if self.acc_fn is None:
                raise NotImplementedError('accuracy function not specified')
            acc_sum = self.acc_fn(
                outputs[-1,:,:].detach(),
                y.detach(),
            )
            # print("loss acc",loss,acc_sum)
            return loss, acc_sum
        return loss

    def train_epoch(self, train=True):
        

        self.diagnostics['Average Train Loss'] = NaN
        self.diagnostics['Start Train Loss'] = NaN
        self.diagnostics['Final Train Loss'] = NaN
        self.diagnostics['Start Train Accuracy'] = NaN
        self.diagnostics['Final Train Accuracy'] = NaN
        self.diagnostics['Time Training'] = NaN
        self.diagnostics['Test Loss'] = NaN
        self.diagnostics['Test Accuracy'] = NaN    
        self.diagnostics['Time Testing'] = NaN
        self.diagnostics['Num Batches'] = 0

        if train:
            self.dataset.start_train_epoch()
            train_losses, tr_accuracy = [], []
            self.model.train()
            start_train_time = time.time()    
            step_loss = 0
            step_num = 0
            while not self.dataset.train_epoch_ends:
            # for _ in tqdm(range(self.steps_per_epoch)):
                self.optim.zero_grad()
                x,y = self.dataset.get_batch(train=True)
                loss,acc = self.get_loss(x, y,return_acc=True)
                loss.backward()
                train_losses.append(loss.detach().cpu().item())
                tr_accuracy.append(acc.detach().cpu().item())
                self.optim.step()
                self.diagnostics['Num Batches'] += 1
                

            end_train_time = time.time()

            avg_tr_loss = sum(train_losses) / self.diagnostics['Num Batches']
            self.diagnostics['Average Train Loss'] = avg_tr_loss
            self.diagnostics['Start Train Loss'] = train_losses[0]
            self.diagnostics['Final Train Loss'] = train_losses[-1]
            self.diagnostics['Start Train Accuracy'] = tr_accuracy[0]
            self.diagnostics['Final Train Accuracy'] = tr_accuracy[-1]
            self.diagnostics['Time Training'] = end_train_time - start_train_time


        test_loss, test_accuracy = 0., 0.
        self.dataset.start_test_epoch()
        self.model.eval()
        start_test_time = time.time() 
        test_steps = 0
        with torch.no_grad():
            # for _ in range(test_steps):
             while not self.dataset.test_epoch_ends:
                x,y= self.dataset.get_batch(train=False)
                loss,acc = self.get_loss(x,y,return_acc=True)
                test_loss += loss.detach().cpu().item() 
                test_accuracy += acc.detach().cpu().item() 
                test_steps += 1
        end_test_time = time.time()
        self.diagnostics['Test Loss'] = test_loss / test_steps
        self.diagnostics['Test Accuracy'] = test_accuracy / test_steps
        self.diagnostics['Time Testing'] = end_test_time - start_test_time

        if train:
            return  self.diagnostics['Average Train Loss'],self.diagnostics['Final Train Accuracy'],self.diagnostics['Test Loss'],self.diagnostics['Test Accuracy']
        else:
            return None,None,test_loss,test_accuracy



def main(loss_fn,accuracy_fn):
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, default='test_task',
                        help='task to train or evalute models')
    parser.add_argument('--model_name', type=str, default='rnn',
                        help='name of models to fuse')
    parser.add_argument('--dataset_name', type=str, default='dataset',
                        help='name of the dataset to use')  
    parser.add_argument('--train_data_path', type=str, default='train.csv',
                        help='dataset path')               


    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of iterations for trainer')
    parser.add_argument('--steps_per_iter', type=int, default=100,
                        help='Number of gradient steps per iteration')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size for training')
    parser.add_argument('--gpu_batch_size', type=int, default=16,
                        help='Max batch size to put on GPU (used for gradient accumulation)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device for Pytorch to use')
    
    parser.add_argument('--note', '-n', default='',
                        help='An optional note to be logged to W&B')
    parser.add_argument('--include_date', action='store_true',
                        help='Whether to include date in run name')

    parser.add_argument('--save_models', action='store_true',
                        help='Whether or not to save the model files locally')
    parser.add_argument('--save_models_every', type=int, default=10,
                        help='How often to save models locally')
    parser.add_argument('--pretrained', action='store_true',
                        help='check whether to load a previously trained model or to initialize a new model')
    parser.add_argument('--model_load_path', type=str, default='',
                        help='path to load a model')
    parser.add_argument('--model_save_path', type=str, default='',
                        help='path to save a model')

    # arguments for simple RNN models
    parser.add_argument('--input_dim', type=int, default=None,
                        help='input dimension of an RNN model')
    parser.add_argument('--hidden_dims', type=str, default=None,
                        help='list of hidden dimensions')
    parser.add_argument('--output_dim', type=int, default=None,
                        help='path to save a model')
    parser.add_argument('--hidden_activations', type=str, default=None,
                        help='list of hidden activations')
    parser.add_argument('--bias', action='store_true',
                        help='bias')

    exp_args = parser.parse_args(sys.argv[1:])
    
    hidden_dims = [int(i) for i in exp_args.hidden_dims.strip('[]').split(',')]
    hidden_activations = None if exp_args.hidden_activations is None else exp_args.hidden_activations.strip('[]').split(',')

    if exp_args.include_date:
        timestamp = datetime.now().strftime('%m-%d')
        exp_name = f'{timestamp}-{exp_args.model_name}-{exp_args.task_name}'
    

    # TODO: change dataset class, make it able to create differnt dataset given differnt dataset name
    exp_args.batch_size = exp_args.gpu_batch_size if exp_args.device == 'cuda' else exp_args.batch_size 
    dataset = RNNDataset(batch_size=exp_args.batch_size,eval_batch_size=exp_args.batch_size, device=exp_args.device,dataset_name=exp_args.dataset_name,train_data_path=exp_args.train_data_path)



    """
    Set up model
    """
    if exp_args.model_name == "rnn":
        # dimensions of input, hidden, output must be specified
        assert exp_args.input_dim is not None and hidden_dims is not None and exp_args.output_dim is not None
        model = RNNModel(exp_args.input_dim, hidden_dims, exp_args.output_dim, hidden_activations,exp_args.bias)
    
    print("os.path.isfile(exp_args.model_load_path)",os.path.isfile(exp_args.model_load_path))
    print(exp_args.model_load_path)
    if exp_args.pretrained is True and os.path.isfile(exp_args.model_load_path):
        try:
            checkpoint = torch.load(exp_args.model_load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'----------finish loading model from path: {exp_args.model_load_path}----------')
        except: 
            print("Failed to load model, Unexpected error:", sys.exc_info()[0])
            pass

    model.to(exp_args.device)

    """
    Set up trainer
    """

    trainer = Trainer(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        steps_per_epoch=exp_args.steps_per_iter,
        learning_rate=exp_args.learning_rate,
        batch_size=exp_args.batch_size,
        eval_batch_size=exp_args.batch_size,
        device = exp_args.device
    )

    """
    Set up logging
    """

    short_name = str(random.randint(int(1e5), int(1e6) - 1))
    run_name = f'{exp_name}-{short_name}'
    end_traing = False
    pre_tr_loss = None

    record_epoches = []

    # Evalutation
    avg_tr_loss,tr_accuracy,test_loss,test_accuracy = trainer.train_epoch(train=False)
    print(f'| Iteration {" " * 15} | {0:25} |')
    for k, v in trainer.diagnostics.items():
        print(f'| {k:25} | {v:25} |')

    for epoch in range(1, exp_args.num_epochs+1):
        avg_tr_loss,tr_accuracy,test_loss,test_accuracy = trainer.train_epoch(train=True)

        record_epoches.append([avg_tr_loss,tr_accuracy,test_loss,test_accuracy])

        with open( "records_" + str(exp_args.batch_size) + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(record_epoches)

        if epoch % 10 == 0:
            print('=' * 57)
            print(f'| Iteration {" " * 15} | {epoch:25} |')
            for k, v in trainer.diagnostics.items():
                print(f'| {k:25} | {v:25} |')

        # diff = 1 if pre_tr_loss is None else avg_tr_loss - pre_tr_loss
        # pre_tr_loss = avg_tr_loss
        # if diff < 0.0001:
        #     end_traing = True

        if exp_args.save_models and ((epoch) % exp_args.save_models_every == 0 or
                            (epoch) == exp_args.num_epochs or end_traing):
            with open(exp_args.model_save_path, 'wb') as f:
                state_dict = dict(epoch=epoch,tr_acc=tr_accuracy,test_acc=test_accuracy,model_state_dict=model.state_dict(),config=model.get_model_config(),optimizer_state_dict=trainer.optim.state_dict())
                torch.save(state_dict, f)
                # torch.save(model.state_dict(),f)
            print(f'Saved model at {epoch} iters: {run_name}')


if __name__ == "__main__":
    main(loss_fn="cross_entropy",accuracy_fn="acc_classification")

"""
for testing purpose, test on datset of names
"""
# if __name__ == "__main__":
#     batch_size = 2
#     dataset = RNNDataset("cpu",None,batch_size,batch_size,'names',None)
#     x,y = dataset.get_batch(True)
#     print("type of x:",type(x))
#     print(x.size())
#     print("type of y:",type(y))
#     print(y.size())
#     print(dataset.params['max_len']-1)
#     print(len(dataset.params['all_categories']))
#     model = RNNModel(dataset.params['n_letters'],[15],len(dataset.params['all_categories']),None)
#     # h0 = torch.randn(1, batch_size, 2)

#     # last_output shape = (batch_size,hidden_size)
#     # y shape = (batch_size)
#     #torch.nn.NLLLoss() -- Input: (N,C), target(N)
#     m = torch.nn.LogSoftmax(dim=1)
#     loss =torch.nn.NLLLoss() 
#     def loss_fn(last_output,y):
#         return(loss(m(last_output),y))
#     def accuracy_fn(last_output,y):
#         return 0
#     trainer = Trainer(model=model,dataset=dataset,loss_fn=torch.nn.NLLLoss(),accuracy_fn=accuracy_fn,batch_size=batch_size,eval_batch_size=batch_size)
#     trainer.train_epoch()
    