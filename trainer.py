from torch import nn
import torch
from utils import BatchNorm, BNTT, get_best_gpu, ARGS, write_run_to_db, RunningAverage, TrackSampling
import torch.nn.functional as F
from fedlib import Client, test_model, eval_model, extract_bn_stats, extract_grads, apply_bn_stats, apply_grads, extract_params, apply_params
from ds import get_ds, iid, non_iid
from torch.utils.data import Subset
import numpy as np
import attacks
import defenses

import random

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Python 3.9+



def write_result(args, max_accuracy):
    json_if_dict = lambda value: value if not isinstance(value, dict) else json.dumps(value)
    data = {
        'created_at': datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        'max_accuracy': max_accuracy,
    }
    data.update({
        key: json_if_dict(getattr(args, key, None)) for key in args.db_column_keys
    })
    write_run_to_db(args.db_table, data)

def train(args, model, device):
    args.monitor = getattr(args, 'monitor', False)
    if args.monitor: print(args)
    
    if (getattr(args, 'attack', None) == None and getattr(args, 'num_byz_clients', 0) == 0):
        attack = None; args.attack = None
        args.num_byz_clients = 0
    elif (getattr(args, 'attack', None) != None and getattr(args, 'num_byz_clients', 0) > 0):
        if isinstance(args.attack, dict): attack = getattr(attacks, args.attack['name'])(**args.attack['params'])
        else: attack = getattr(attacks, args.attack)()
    else:
        raise ValueError("Error attack config")

    train_ds, test_ds = get_ds(args.dataset)
    test_loader = torch.utils.data.DataLoader(test_ds, args.test_batch_size)

    if getattr(args, 'seed', None) is not None: g = torch.Generator(); g.manual_seed(args.seed)
    else: g = None; args.seed = None

    if args.non_iid_alpha is None: idxes = iid(np.arange(len(train_ds)), num_users=args.num_clients, seed=args.seed)
    else: idxes = non_iid(train_ds, list(range(len(train_ds.classes))), args.num_clients, alpha=args.non_iid_alpha, seed=args.seed)

    if args.monitor: print([len(idxes[i]) for i in range(args.num_classes)])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    clients = []

    for i in range(args.num_clients - args.num_byz_clients):
        ds = Subset(train_ds, idxes[i])
        loader = torch.utils.data.DataLoader(ds, batch_size=args.client_batch_size, shuffle=True, drop_last=True, generator=g)
        clients.append(Client('benign', model, loader, nn.CrossEntropyLoss(), device))

    if not isinstance(args.defense, dict): defense = getattr(defenses, args.defense)()
    else: defense = getattr(defenses, args.defense['name'])(**args.defense['params'])

    max_accuracy = 0
    epoch_target_samples = len(train_ds)
    assert args.momentum in ['local', 'global'], "Momentum configuration not correct"

    if args.bn_stats_rollback: bn_backup = extract_bn_stats(model.bn_modules)
    if args.momentum == 'local': params_backup = extract_params(model)

    if args.monitor: tracker = TrackSampling()
    
    for epoch in range(args.total_epochs):
        total_trained_samples = 0
        while total_trained_samples <  epoch_target_samples:
            updates = []
            bn_stats = []

            for i in range(args.num_clients - args.num_byz_clients):
                if i > 0:
                    if args.bn_stats_rollback: apply_bn_stats(bn_backup, model.bn_modules)
                    if args.momentum == 'local': apply_params(params_backup, model)
                    
                clients[i].train()
                if args.momentum == 'local': 
                    opt.step()
                    updates.append(extract_params(model) - params_backup)
                else: 
                    updates.append(extract_grads(model))
                if args.bn_stats_rollback: bn_stats.append(extract_bn_stats(model.bn_modules))

                total_trained_samples += clients[i].num_samples
                if args.monitor: tracker.add(clients[i].num_samples)

            if attack != None: 
                mal_update = attack(updates)
                for _ in range(args.num_byz_clients):
                    updates.append(mal_update)

            if getattr(args, 'clients_dropout', None) is not None:
                n_dropout = int(args.clients_dropout * args.num_clients)
                updates = random.sample(updates, len(updates) - n_dropout)
                
            aggr_update = defense(updates)
            if args.momentum == 'local': 
                apply_params(params_backup + aggr_update, model)
                params_backup = extract_params(model)
            else:
                apply_grads(aggr_update, model)
                opt.step()

            if args.bn_stats_rollback:
                bn_backup = torch.stack(bn_stats).mean(dim=0)
                apply_bn_stats(bn_backup, model.bn_modules)

            if args.monitor and tracker.count > 1000:
                print(f"Training speed: {tracker.sps} samples/s")

        test_loss, test_acc = eval_model(model, test_loader, device)
        print(f"{epoch=}, {test_loss=}, {test_acc=}")
        max_accuracy = max(max_accuracy, test_acc)

    write_result(args, max_accuracy)