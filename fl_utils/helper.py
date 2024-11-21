import copy
import sys
import time
from typing import Dict

from prefetch_generator import BackgroundGenerator
from torch import nn
from torchvision.models import ResNet18_Weights

from models.vgg13_fashion_mnist import VGG13_fashion

sys.path.append("../")
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from PIL import Image

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18
from models.vgg import get_vgg13
from models.CNN import FiveLayerCNN
import torchvision.models as models
# from models.resnet_timagenet import ResNet18_tinyimagenet
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
class Helper:
    def __init__(self, config):
        start = time.time()
        self.config = config
        self.config.data_folder = './datasets'
        self.local_model = None
        self.global_model = None
        self.client_models = []
        self.all_user_updates = {}  # 用于保存每个客户端的上一次更新参数

        self.setup_all()
        print('Time taken to setup helper:', time.time()-start)

    def setup_all(self):
        self.load_data()
        self.load_model()
        self.config_adversaries()


    def load_model(self):
        num_channels = 3 if self.config.dataset != 'fashion-mnist' else 1
        if self.config.model == 'resnet18':
            if self.config.dataset in ['cifar10', 'fashion-mnist', 'tinyimagenet']:
                self.local_model = ResNet18(num_classes = self.num_classes, num_channels=num_channels)
                self.local_model.cuda()
                self.global_model = ResNet18(num_classes = self.num_classes, num_channels=num_channels)
                self.global_model.cuda()
                for i in range(self.config.num_total_participants):
                    t_model = ResNet18(num_classes = self.num_classes)
                    t_model.cuda()
                    self.client_models.append(t_model)
            elif self.config.dataset == 'TinyImageNet':
                model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
                model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),)
                model.maxpool =nn.Sequential()
                # load_pretrained_weights(model, r'C:\Users\iamje\Downloads\A3FL-3\A3FL-main\models\pretrained_weights\resnet18-5c106cde.pth')
                self.local_model = model
                self.local_model.cuda()
                self.global_model = copy.deepcopy(model)
                self.global_model.cuda()
                for i in range(self.config.num_total_participants):
                    t_model = copy.deepcopy(model)
                    t_model.cuda()
                    self.client_models.append(t_model)



                # self.local_model = ResNet18_tinyimagenet(pretrain=True,)
                # self.local_model.cuda()
                # self.global_model = ResNet18_tinyimagenet(pretrain=True)
                # self.global_model.cuda()
                # for i in range(self.config.num_total_participants):
                #     t_model = ResNet18_tinyimagenet(pretrain=True)
                #     t_model.cuda()
                #     self.client_models.append(t_model)

        elif self.config.model == 'vgg13':
            if self.config.dataset =='fashion-mnist': # 弃用
                self.local_model = VGG13_fashion(num_classes = self.num_classes)
                self.local_model.cuda()
                self.global_model = VGG13_fashion(num_classes = self.num_classes)
                self.global_model.cuda()
                for i in range(self.config.num_total_participants):
                    t_model = VGG13_fashion(num_classes = self.num_classes)
                    t_model.cuda()
                    self.client_models.append(t_model)
            else:
                in_channels = 3
                self.local_model = get_vgg13(num_classes = self.num_classes,in_channels=in_channels)
                self.local_model.cuda()
                self.global_model = get_vgg13(num_classes = self.num_classes,in_channels=in_channels)
                self.global_model.cuda()
                for i in range(self.config.num_total_participants):
                    t_model = get_vgg13(num_classes = self.num_classes,in_channels=in_channels)
                    t_model.cuda()
                    self.client_models.append(t_model)
        elif self.config.model =='CNN':
            if self.config.dataset == 'fashion-mnist':
                self.local_model = FiveLayerCNN()
                self.local_model.cuda()
                self.global_model = FiveLayerCNN()
                self.global_model.cuda()
                for i in range(self.config.num_total_participants):
                    t_model = FiveLayerCNN()
                    t_model.cuda()
                    self.client_models.append(t_model)




    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def get_train(self, indices):
        # train_loader = torch.utils.data.DataLoader(
        #     self.train_dataset,
        #     batch_size=self.config.batch_size,
        #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        #     num_workers=self.config.num_worker)
        train_loader = DataLoaderX(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config.num_worker, pin_memory=self.config.pin_memory)
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)

        return test_loader

    def load_data(self):
        start = time.time()
        self.num_classes = 10 if self.config.dataset == 'cifar10' or self.config.dataset=='fashion-mnist' else 200

        transform_train = None
        transform_test = None

        if self.config.dataset == 'cifar10':
            self.num_classes = 10
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.config.dataset == 'fashion-mnist':
            self.num_classes = 10
            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # 对于灰度图像，只需使用单个通道的均值和标准差
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # 同上
            ])
        elif self.config.dataset == 'TinyImageNet':
            # transform_train = transforms.Compose([
            #     transforms.RandomResizedCrop(64),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 根据ImageNet的统计数据
            # ])
            # transform_test = transforms.Compose([
            #     transforms.Resize(64),
            #     transforms.CenterCrop(64),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # ])
            # Tiny ImageNet 的数据变换
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 根据ImageNet的统计数据
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif self.config.dataset == 'tinyimagenet':
            self.num_classes = 100
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

        if self.config.dataset == 'cifar10':
            self.train_dataset = datasets.CIFAR10(
                self.config.data_folder, train=True,
                download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(
                self.config.data_folder, train=False, transform=transform_test)
        elif self.config.dataset == 'fashion-mnist':
            self.train_dataset = datasets.FashionMNIST(
                self.config.data_folder, train=True,
                download=True, transform=transform_train)
            self.test_dataset = datasets.FashionMNIST(
                self.config.data_folder, train=False, transform=transform_test)
        elif self.config.dataset == 'TinyImageNet':
            # Tiny ImageNet 的数据加载逻辑
            _data_dir = os.path.join(self.config.data_folder, 'tiny-imagenet-200')
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'), transform_train)
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'), transform_test)
        elif self.config.dataset == 'tinyimagenet':
            self.train_dataset = datasets.tinyimagenet(
                self.config.data_folder, train=True,
                download=True, transform=transform_train)
            self.test_dataset = datasets.tinyimagenet(
                self.config.data_folder, train=False, transform=transform_test)

        indices_per_participant = self.sample_dirichlet_train_data(
            self.config.num_total_participants,
            alpha=self.config.dirichlet_alpha)

        train_loaders = [self.get_train(indices)
                         for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()

        train_loaders_target = []
        for participant_id, indices in indices_per_participant.items():
            target_indices = [idx for idx in indices if self.train_dataset[idx][1] == self.config.target_class]
            target_train_loader = DataLoaderX(
                self.train_dataset,
                batch_size=self.config.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(target_indices),
                num_workers=self.config.num_worker, pin_memory=self.config.pin_memory)
            train_loaders_target.append(target_train_loader)
        self.train_data_target = train_loaders_target
        print('Time taken to load data:', time.time()-start)

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if name =='num_batches_tracked':
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update

    def config_adversaries(self):
        if self.config.is_poison:
            self.adversary_list = list(range(self.config.num_adversaries))
        else:
            self.adversary_list = list()


def load_pretrained_weights(model, pretrained_weights_path):
    pretrained_weights = torch.load(pretrained_weights_path)
    model_dict = model.state_dict()

    # 初始化一个列表来记录不匹配的层的名字
    mismatched_layers = []
    matched_layers = []

    # 过滤并加载权重
    for k, v in pretrained_weights.items():
        if k in model_dict:
            # 如果形状一样，就加载权重
            if v.shape == model_dict[k].shape:
                model_dict[k] = v
                matched_layers.append(k)
            # 如果形状不一样，就记录网络层的名字
            else:
                mismatched_layers.append(k)

    # 加载新的state dict
    model.load_state_dict(model_dict)
    # print("Matched layers: ", matched_layers)
    # 返回不匹配的层的名字
    return mismatched_layers