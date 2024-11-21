import torch
import torch.nn as nn
import torch.nn.functional as F

# 控制生成器中的特征图数量
ngf = 64

class GeneratorResnetCIFAR(nn.Module):
    def __init__(self):
        super(GeneratorResnetCIFAR, self).__init__()

        # 初始卷积块
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, ngf, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # 下采样块
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # 残差块
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)

        # 上采样块
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # 最终卷积块
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, 3, kernel_size=3, padding=0)
        )

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        x = torch.tanh(x)  # 输出范围 [-1, 1]
        # 将范围从 [-1, 1] 转换到 [0, 255]
        x = (x + 1) /2  # 输出范围 [0, 255]
        return x  # 应用标准化

# def normalize(tensor):
#     mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(tensor.device)
#     std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(tensor.device)
#     return (tensor / 255.0 - mean) / std

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


if __name__ == '__main__':
    netG = GeneratorResnetCIFAR()
    test_sample = torch.rand(10, 3, 32, 32)
    print('Generator output:', netG(test_sample).size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad) / 1000000)
    print(test_sample)