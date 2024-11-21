import torch
import torch.nn as nn
import torch.nn.functional as F

class FiveLayerCNN(nn.Module):
    def __init__(self):
        super(FiveLayerCNN, self).__init__()
        # 第一层：卷积层 + ReLU
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层：卷积层 + ReLU
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 第三层：最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Dropout层
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层：卷积 + ReLU
        x = F.relu(self.conv1(x))
        # 第二层：卷积 + ReLU
        x = F.relu(self.conv2(x))
        # 最大池化层
        x = self.pool(x)
        # 扁平化
        x = x.view(-1, 64 * 14 * 14)
        # Dropout层 + 全连接层 + ReLU
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        # Dropout层 + 全连接层
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':

    # 检查模型结构
    model = FiveLayerCNN()
    for name, param in model.named_parameters():
        print(name, param.size())
    print(model)
