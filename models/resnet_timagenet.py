import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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

def ResNet18_tinyimagenet(pretrain, pretrained_weights_path=r'C:\Users\iamje\Downloads\A3FL-3\A3FL-main\models\pretrained_weights\resnet18-5c106cde.pth'):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=200)
    if pretrain:
        mismatched_layers = load_pretrained_weights(model, pretrained_weights_path)
        # print("Mismatched layers: ", mismatched_layers)
    return model

# def ResNet18_tinyimagenet(pretrain, pretrained_weights_path='A3FL-main/models/pretrained_weights/resnet18-5c106cde.pth'):
#     model = ResNet(BasicBlock, [2,2,2,2], num_classes=200)
#     if pretrain:
#         load_pretrained_weights(model, pretrained_weights_path)
#     return model

if __name__ == '__main__':
    #测试
    # net = ResNet18_tinyimagenet(True)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 200)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), )
    model.maxpool = nn.Sequential()
    # load_pretrained_weights(model,
    #                         r'C:\Users\iamje\Downloads\A3FL-3\A3FL-main\models\pretrained_weights\resnet18-5c106cde.pth')
    #
    weights = model.state_dict()

    y = model(torch.randn(1,3,64,64))
    print(y.size())

