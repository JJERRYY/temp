import sys
from collections import Counter

sys.path.append("../")

import wandb
import argparse
import yaml
import traceback

import torch
import torchvision
import numpy as np
import random

from fl_utils.helper import Helper
from fl_utils.fler import FLer

import os
os.environ["WANDB_MODE"] = "dryrun"
def setup_wandb(config_path, sweep):
    with open(config_path, 'r', encoding='utf-8') as stream:
        sweep_configuration = yaml.safe_load(stream)
    if sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='FanL-clean')
        return sweep_id
    else:
        config = sweep_configuration['parameters']
        d = dict()
        for k in config.keys():
            v = config[k][list(config[k].keys())[0]]
            if type(v) is list:
                if v:
                    d[k] = {'value': v[0]}
                else:
                    d[k] = {'value': []}
            else:
                d[k] = {'value': v}
        yaml.dump(d, open('./yamls/tmp.yaml', 'w'))
        wandb.init(config='./yamls/tmp.yaml')
        return None

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def main():
    run = wandb.init(project='main')
    set_seed(wandb.config.seed)

    config = wandb.config
    helper = Helper(wandb.config)
    fler = FLer(helper)
    experiment_name = run.name
    if helper.config.is_poison:
        run.name = experiment_name + '_' + helper.config.dataset + '_' + helper.config.model + '_' + helper.config.defense_technique + '_' + helper.config.attacker + '_' + helper.config.note
    else:
        run.name = experiment_name + '_' + helper.config.dataset + '_' + helper.config.model + '_' + 'clean' + '_' + helper.config.note
    fler.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='configs/config_1.yaml')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))
    sweep_id = setup_wandb(args.params, args.sweep)
    if args.sweep:
        print('sweep_id:', sweep_id)
        wandb.agent(sweep_id, function=main)
    else:
        main()