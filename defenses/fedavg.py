import math
from copy import deepcopy
from typing import List, Any, Dict
import torch
import logging
import os
from torch.nn import Module

from fl_utils.helper import Helper

# from utils.parameters import Params

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FedAvg:
    # params: Params
    helper: Helper
    ignored_weights = ['num_batches_tracked']  # ['tracked', 'running']

    def __init__(self, helper: Helper) -> None:
        # self.params = params
        self.helper = helper

    # FedAvg aggregation
    def aggr(self, weight_accumulator, _: Module, models_updates: List[Dict[str, Any]]):
        for i in range(self.helper.config.num_sampled_participants):
            # updates_name = '{0}/saved_updates/update_{1}.pth'\
            #     .format(self.params.folder_path, i)
            # loaded_params = torch.load(updates_name)

            self.accumulate_weights(weight_accumulator, models_updates[i])

    def accumulate_weights(self, weight_accumulator, local_update):
        scale = 1 / self.helper.config.num_sampled_participants
        for name, value in local_update.items():
            value = value * scale
            weight_accumulator[name].add_(value)

    def update_global_model(self, weight_accumulator, global_model: Module):
        # self.last_global_model = deepcopy(self.model)
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            # scale = 1 / self.helper.config.num_sampled_participants
            average_update = sum_update
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)




    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if 'tracked' in name or 'running' in name:
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def add_noise(self, sum_update_tensor: torch.Tensor, sigma):
        noised_layer = torch.FloatTensor(sum_update_tensor.shape)
        noised_layer = noised_layer.to(self.params.device)
        noised_layer.normal_(mean=0, std=sigma)
        sum_update_tensor.add_(noised_layer)

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False
