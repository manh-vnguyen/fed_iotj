from torch import nn
import torch
from utils import BatchNorm, BNTT
from spikingjelly.activation_based import functional, neuron, layer

from utils import ARGS
from runner import ExperimentRunnerProcess
from trainer import train
    
class VGG9(nn.Module):
    def __init__(self, num_classes, C, H, W):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            BatchNorm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            BatchNorm(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * (H // 8) * (W // 8), 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes, bias=True),
        )
        
        self.bn_modules = nn.ModuleList([m for m in self.modules() if isinstance(m, (BatchNorm))])

    def forward(self, x):
        return self.net(x)
    
class Runner(ExperimentRunnerProcess):
    def run_experiment(self, args, gpu_id):
        device = f'cuda:{gpu_id}'
        model = VGG9(args.num_classes, 3, 32, 32).to(device)

        train(args, model, device)
if __name__ == '__main__':
    from itertools import product
    db_table = 'vgg9'
    db_column_keys = ['num_clients', 'num_byz_clients', 'clients_dropout', 'attack', 'defense', 
                    'non_iid_alpha', 'client_batch_size', 'seed', 'test_batch_size', 'momentum',
                    'dataset', 'num_classes', 'bn_stats_rollback', 'total_epochs']
    argss = []
    momentum_confs = ['global', 'local']
    defense_confs = [
        'Mean',
        {
            'name': 'TopK',
            'params': {
                'kappa': 0.02
            }
        },
        {
            'name': 'TopK',
            'params': {
                'kappa': 0.01
            }
        }
    ]

    non_iid_confs = [0.1]
    
    for momentum_conf, defense_conf, alpha in product(momentum_confs, defense_confs, non_iid_confs):
        argss.append(ARGS({
            'db_table': db_table,
            'db_column_keys': db_column_keys,
            'seed': 32,
            'num_clients': 20,
            'num_byz_clients': 4,
            'momentum': momentum_conf,
            'defense': defense_conf,
            'attack': 'MinMax',
            'clients_dropout': None,
            'client_batch_size': 8,
            'non_iid_alpha': alpha,
            'test_batch_size': 8,
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'bn_stats_rollback': True,
            'total_epochs': 10,
            'monitor': False,
        }))

    for momentum, alpha in product(momentum_confs, non_iid_confs):
        argss.append(ARGS({
            'db_table': db_table,
            'db_column_keys': db_column_keys,
            'seed': 32,
            'num_clients': 20,
            'momentum': momentum,
            'defense': 'Mean',
            'clients_dropout': None,
            'client_batch_size': 8,
            'non_iid_alpha': alpha,
            'test_batch_size': 8,
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'bn_stats_rollback': True,
            'total_epochs': 10,
            'monitor': False,
        }))

    runner = Runner({0: 2})
    # runner.run_parallel_experiment(argss)
    for args in argss:
        runner.run_experiment(args, 0)