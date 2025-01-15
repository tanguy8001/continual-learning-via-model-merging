import logging
import os
import shutil
import torch

from torch.utils import tensorboard


def make_dirs(dirname, rm=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif rm:
        logging.info('Rm and mkdir {}'.format(dirname))
        shutil.rmtree(dirname)
        os.makedirs(dirname)


class Init:
    def __init__(self, args, run_str=None):
        torch.manual_seed(args.seed)
        model_name = args.model_name + '_' + args.dataset_name
        self.output_path = os.path.join(args.result_path, args.experiment_name, model_name)
        self.sample_path = os.path.join(self.output_path, 'samples')
        self.run_path = os.path.join(self.output_path, 'runs')
        if run_str is None:
            run_str = 'no_run_str'
        self.writer_path = os.path.join(self.run_path, run_str)
        self.model_path = os.path.join(self.writer_path, 'snapshots')

        if not args.resume:
            make_dirs(args.result_path)
            make_dirs(self.output_path)
            make_dirs(self.sample_path)
            make_dirs(self.run_path)
            make_dirs(self.writer_path, rm=True)
            make_dirs(self.model_path)
            args_state = {k: v for k, v in args._get_kwargs()}
            with open(os.path.join(self.run_path, 'args.txt'), 'w') as f:
                print(args_state, file=f)

        if args.evaluate:
            self.writer = tensorboard.SummaryWriter(comment=model_name, log_dir='/tmp')
        else:
            self.writer = tensorboard.SummaryWriter(comment=model_name, log_dir=self.writer_path)