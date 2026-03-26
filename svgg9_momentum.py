from torch import nn
import torch
from utils import BatchNorm, BNTT
from spikingjelly.activation_based import functional, neuron, layer

from utils import ARGS
from runner import ExperimentRunnerProcess
from trainer import train

class SVGG9(nn.Module):
    conv_bias = False
    def __init__(self, num_classes, C, H, W, T, norm='bntt', spike_gen=False):
        super().__init__()
        assert norm in ('bntt', 'bn')
        self.spike_gen = spike_gen
        self.T = T
        self.act = neuron.IFNode(step_mode='m', v_reset=None)

        def make_norm(num_features):
            if norm == 'bntt':
                return BNTT(T, num_features)
            else:
                return BatchNorm(num_features)

        blocks = []

        # block 1: C -> 64 -> 64, pool
        blocks.append(layer.MultiStepContainer(
            nn.Conv2d(C, 64, kernel_size=3, padding=1, bias=False),
            make_norm(64),
        ))
        blocks.append(layer.MultiStepContainer(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            make_norm(64),
        ))
        blocks.append(layer.MultiStepContainer(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            make_norm(128),
        ))
        blocks.append(layer.MultiStepContainer(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            make_norm(128),
        ))
        blocks.append(layer.MultiStepContainer(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            make_norm(256),
        ))
        blocks.append(layer.MultiStepContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            make_norm(256),
        ))
        blocks.append(layer.MultiStepContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            make_norm(256),
        ))

        # head: flatten -> fc -> norm -> fc
        feat_dim = 256 * (H // 8) * (W // 8)
        blocks.append(layer.MultiStepContainer(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(feat_dim, 1024, bias=True),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.last_layer = layer.MultiStepContainer(
            nn.Linear(1024, num_classes, bias=True)
        )

        self.bn_modules = nn.ModuleList([m for m in self.modules() if isinstance(m, (BatchNorm))])

    def reset_bntt(self):
        for m in self.modules():
            if isinstance(m, BNTT):
                m.reset()

    def poisson_gen(self, x, rescale_fac=2.0):
        return ((torch.rand((self.T, *x.shape), device=x.device) * rescale_fac).le(torch.abs(x)) * torch.sign(x)).float()

    def forward(self, x):
        self.reset_bntt()
        if self.spike_gen: x = self.poisson_gen(x)
        else: x = x.repeat(self.T, 1, 1, 1, 1)
        for block in self.blocks:
            x = block(x)
            functional.reset_net(self.act)
            x = self.act(x)
        x = self.last_layer(x)
        return torch.mean(x, dim=0)

class Runner(ExperimentRunnerProcess):
    def run_experiment(self, args, gpu_id):
        device = f'cuda:{gpu_id}'
        model = SVGG9(args.num_classes, 3, 32, 32, args.timesteps, norm=args.norm, spike_gen=args.spike_gen).to(device)
        train(args, model, device)

if __name__ == '__main__':
    from itertools import product
    db_table = 'svgg9'
    db_column_keys = ['num_clients', 'num_byz_clients', 'clients_dropout', 'attack', 'defense', 
                    'non_iid_alpha', 'client_batch_size', 'seed', 'test_batch_size', 'momentum',
                    'dataset', 'num_classes', 'bn_stats_rollback', 'total_epochs',
                    'norm', 'timesteps', 'spike_gen']
    argss = []
    momentum_confs = ['global']
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
    
    for momentum_conf,  defense_conf in product(momentum_confs, defense_confs):
        argss.append(ARGS({
            'db_table': db_table,
            'db_column_keys': db_column_keys,
            'seed': 32,
            'num_clients': 20,
            'num_byz_clients': 8,
            'momentum': momentum_conf,
            'attack': 'MinMax',
            'defense': defense_conf,
            'clients_dropout': None,
            'client_batch_size': 8,
            'timesteps': 4,
            'spike_gen': False,
            'non_iid_alpha': None,
            'test_batch_size': 32,
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'norm': 'bntt',
            'bn_stats_rollback': True,
            'total_epochs': 10,
        }))

    runner = Runner({3: 2, 4: 2, 5: 2, 6: 2, 7: 3})
    # runner.run_parallel_experiment(argss)
    for args in argss:
        runner.run_experiment(args, 0)