import sys

from torch import nn

from models.FashionMNISTGenerator import GeneratorResnetFashionMNIST

sys.path.append("../")
import time
import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18
from models.cifar10generator import GeneratorResnetCIFAR
import copy
import os
import math


class Attacker_max_feature_map_generator:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        if self.helper.config.dataset == 'cifar10' or self.helper.config.dataset == 'tinyimagenet':
            self.generator = GeneratorResnetCIFAR().cuda()
        elif self.helper.config.dataset == 'fashion-mnist':
            self.generator = GeneratorResnetFashionMNIST().cuda()
        self.eps = 10.0 / 255

    def calculate_feature_map_distribution(self, model, dl):
        model.eval()
        sum_feature_maps = None
        count_samples = 0

        for inputs, labels in dl:
            inputs, labels = inputs.cuda(), labels.cuda()
            feature_map = model(inputs)

            if sum_feature_maps is None:
                sum_feature_maps = torch.sum(feature_map, dim=0)
            else:
                sum_feature_maps += torch.sum(feature_map, dim=0)
            count_samples += inputs.size(0)

        average_feature_maps = sum_feature_maps / count_samples
        average_feature_maps = average_feature_maps / torch.max(average_feature_maps)
        return average_feature_maps

    def calculate_kl_divergence(self, feature_maps_normal, feature_maps_adv):
        kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        log_softmax = torch.nn.LogSoftmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)
        kl_loss = kl_div(log_softmax(feature_maps_adv), softmax(feature_maps_normal))
        return kl_loss

    def update_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        target_featuremap = self.calculate_feature_map_distribution(model, self.helper.train_data_target[adversary_id])

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t * m + (1 - m) * inputs
                    labels[:] = self.helper.config.target_class
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct / num_data
            return asr, total_loss

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config.trigger_lr

        K = self.helper.config.trigger_outter_epochs
        self.generator.train()
        for iter in range(K):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                self.gen_optimizer.zero_grad()
                adv = self.generator(inputs)
                adv = torch.min(torch.max(adv, inputs - self.eps), inputs + self.eps)
                adv = torch.clamp(adv, 0.0, 1.0)

                outputs = model(self.normalize(adv))

                labels[:] = self.helper.config.target_class
                class_loss = ce_loss(outputs, labels)

                target_featuremaps = target_featuremap.expand(outputs.shape[0], -1)
                kl_loss = self.calculate_kl_divergence(outputs, target_featuremaps) + self.calculate_kl_divergence(
                    target_featuremaps, outputs)
                sim_loss = self.calculate_neighbourhood_similarity(outputs, target_featuremaps, inputs.size(0))
                sim_loss = sim_loss * 10000
                loss = self.helper.config.attack_alpha * kl_loss + (1 - self.helper.config.attack_alpha) * sim_loss
                wandb.log({'local_kl_loss': kl_loss.item(), 'lcoal_class_loss': class_loss.item(),
                           'local_total_loss': loss.item()}, step=epoch)

                if loss is not None:
                    loss.backward(retain_graph=True)
                    self.gen_optimizer.step()

        trigger_optim_time_end = time.time()
        self.generator.eval()

    def calculate_neighbourhood_similarity(self, adv_out, img_match_out, batch_size):
        criterion_kl = nn.KLDivLoss(size_average=False)

        St = torch.matmul(img_match_out, img_match_out.t())
        norm = torch.matmul(torch.linalg.norm(img_match_out, dim=1, ord=2),
                            torch.linalg.norm(img_match_out, dim=1, ord=2).t())
        St = St / norm

        Ss = torch.matmul(adv_out, adv_out.t())
        norm = torch.matmul(torch.linalg.norm(adv_out, dim=1, ord=2), torch.linalg.norm(adv_out, dim=1, ord=2).t())
        Ss = Ss / norm

        loss_sim = (1.0 / batch_size) * criterion_kl(F.log_softmax(Ss, dim=1), F.softmax(St, dim=1))
        loss_sim += (1.0 / batch_size) * criterion_kl(F.log_softmax(St, dim=1), F.softmax(Ss, dim=1))

        return loss_sim

    def normalize(self, tensor):
        if self.helper.config.dataset == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(tensor.device)
        elif self.helper.config.dataset == 'tinyimagenet':
            mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1).to(tensor.device)
        elif self.helper.config.dataset == 'fashion-mnist':
            mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(tensor.device)
            std = torch.tensor([0.5]).view(1, 1, 1, 1).to(tensor.device)
        return (tensor - mean) / std

    def get_gaussian_kernel(kernel_size=3, pad=2, sigma=2, channels=3):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, padding=kernel_size - pad, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])

        adv = self.generator(inputs[:bkd_num]).detach()
        adv = torch.min(torch.max(adv, inputs[:bkd_num] - self.eps), inputs[:bkd_num] + self.eps)
        adv = self.normalize(adv.detach())
        inputs[:bkd_num] = adv
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels