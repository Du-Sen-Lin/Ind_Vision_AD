import torch
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

N_PARAMS = {'affine': 6,
            'translation': 2,
            'rotation': 1,
            'scale': 2,
            'shear': 2,
            'rotation_scale': 3,
            'translation_scale': 4,
            'rotation_translation': 3,
            'rotation_translation_scale': 5}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class STNModule(nn.Module):
    """实现空间变换网络（STN）的 PyTorch 模块"""
    def __init__(self, in_num, block_index, args):
        """init

        Args:
            in_num (_type_): 输入通道数
            block_index (_type_): 阶段的索引
            args (_type_): 命令行参数
        """
        super(STNModule, self).__init__()
        # 初始化 STN 操作的参数，包括特征图的大小、STN 模式和 STN 操作的参数数量。
        self.feat_size = 56 // (4 * block_index)
        self.stn_mode = args.stn_mode
        self.stn_n_params = N_PARAMS[self.stn_mode]
        # 卷积层序列:包含两个卷积层和池化层，用于提取输入图像的特征。
        self.conv = nn.Sequential(
            conv3x3(in_planes=in_num, out_planes=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv3x3(in_planes=64, out_planes=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # 全连接层序列:包含两个线性层，用于生成 STN 操作的参数。
        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * self.feat_size * self.feat_size, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=self.stn_n_params),
        )
        self.fc[2].weight.data.fill_(0)
        self.fc[2].weight.data.zero_()
        # 针对不同的 STN 模式，初始化了线性层的权重和偏置
        if self.stn_mode == 'affine':
            self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode in ['translation', 'shear']:
            self.fc[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.stn_mode == 'scale':
            self.fc[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'rotation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale':
            self.fc[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))

    def forward(self, x):
        mode = self.stn_mode
        batch_size = x.size(0)
        conv_x = self.conv(x) # 用卷积层序列 self.conv 提取输入图像的特征
        theta = self.fc(conv_x.view(batch_size, -1)) # 将特征展平并通过全连接层序列 self.fc 得到 STN 操作的参数 theta。
        # 根据不同的 STN 模式生成仿射变换矩阵 theta1。
        if mode == 'affine':
            theta1 = theta.view(batch_size, 2, 3)
        else:
            theta1 = Variable(torch.zeros([batch_size, 2, 3], dtype=torch.float32, device=x.get_device()),
                              requires_grad=True)
            theta1 = theta1 + 0
            theta1[:, 0, 0] = 1.0
            theta1[:, 1, 1] = 1.0
            if mode == 'translation':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
            elif mode == 'rotation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
            elif mode == 'scale':
                theta1[:, 0, 0] = theta[:, 0]
                theta1[:, 1, 1] = theta[:, 1]
            elif mode == 'shear':
                theta1[:, 0, 1] = theta[:, 0]
                theta1[:, 1, 0] = theta[:, 1]
            elif mode == 'rotation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 1]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 2]
            elif mode == 'translation_scale':
                theta1[:, 0, 2] = theta[:, 0]
                theta1[:, 1, 2] = theta[:, 1]
                theta1[:, 0, 0] = theta[:, 2]
                theta1[:, 1, 1] = theta[:, 3]
            elif mode == 'rotation_translation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle)
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle)
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]
            elif mode == 'rotation_translation_scale':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * theta[:, 3]
                theta1[:, 0, 1] = -torch.sin(angle)
                theta1[:, 1, 0] = torch.sin(angle)
                theta1[:, 1, 1] = torch.cos(angle) * theta[:, 4]
                theta1[:, 0, 2] = theta[:, 1]
                theta1[:, 1, 2] = theta[:, 2]
        # 使用 PyTorch 提供的 F.affine_grid 和 F.grid_sample 函数对输入图像进行仿射变换，得到输出图像 img_transform 和 STN 操作的输出参数 theta1
        grid = F.affine_grid(theta1, torch.Size(x.shape))
        img_transform = F.grid_sample(x, grid, padding_mode="reflection")

        return img_transform, theta1


class BasicBlock(nn.Module):
    """ ResNet 中基本块（Basic Block）的 PyTorch 模块"""
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """定义了一个基于 ResNet 的神经网络模型，其中嵌入了一些空间变换网络（STN）的模块。
    """
    def __init__(self, args, block, layers):
        """定义了 ResNet 模型的基本结构，包括卷积层、批归一化层、ReLU 激活函数以及四个阶段的残差块（Residual Block）。
        每个阶段都嵌入了一个对应的 STN 模块。
        Args:
            args (_type_): _description_
            block (_type_): _description_
            layers (_type_): _description_
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.stn1 = STNModule(64, 1, args)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.stn2 = STNModule(128, 2, args)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.stn3 = STNModule(256, 3, args)

    def _make_layer(self, block, planes, blocks, stride=1):
        """创建 ResNet 模型中的一个阶段，该阶段由多个残差块（block）组成

        Args:
            block (_type_): 指定的残差块类型，例如 BasicBlock。
            planes (_type_): 该阶段的输出通道数。
            blocks (_type_): 该阶段包含的残差块数量。
            stride (int, optional): 残差块的步幅，默认为 1。. Defaults to 1.

        Returns:
            _type_: nn.Sequential(*layers)：将创建的多个残差块按顺序组合成一个 Sequential 模块
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _fixstn(self, x, theta):
        """执行固定的空间变换（Spatial Transformation）的操作

        Args:
            x (_type_): 输入图像，是一个张量
            theta (_type_): 仿射变换的参数，包含旋转、缩放、平移等信息

        Returns:
            _type_: img_transform：通过仿射变换得到的图像
        """
        grid = F.affine_grid(theta, torch.Size(x.shape)) # 创建一个仿射变换的网格。这个网格定义了输入图像中每个像素在输出图像中的位置。
        img_transform = F.grid_sample(x, grid, padding_mode="reflection") # 对输入图像进行仿射变换

        return img_transform

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x) # 初始卷积层（conv1）和池化层（maxpool）
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # torch.Size([32, 64, 56, 56])
        # 在每个阶段的输出上都进行了 STN 操作;
        # 在每个 STN 操作后，还进行了一个固定的空间变换 _fixstn，将输出进行修正。修正的目的可能是为了避免过度的空间变换引入的误差。
        x = self.layer1(x) # 将处理后的图像 x 通过第一个阶段的残差块（layer1）。
        x, theta1 = self.stn1(x) # 经过 self.stn1 模块，得到 STN 操作的输出 x 和相应的仿射变换参数 theta1。
        # 这个数组的目的是添加仿射矩阵的最后一行，由于仿射矩阵是一个3x3的矩阵，而 theta1 是2x3的矩阵，需要通过添加一行 [0, 0, 1] 来变成3x3。
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        # 将 STN 输出的变换参数 theta1 和上一步创建的数组在第一个轴上拼接，形成仿射变换矩阵。这里的 theta1 是2x3的矩阵，而 tmp 是3x1的矩阵，拼接后形成3x3的矩阵。
        fixthea1 = torch.from_numpy(np.linalg.inv(np.concatenate((theta1.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cuda()
        # 计算修正的仿射变换参数 fixthea1，并将 self.stn1_output 设置为经过固定空间变换得到的输出。
        self.stn1_output = self._fixstn(x.detach(), fixthea1)
        # after layer1 shape:  torch.Size([32, 64, 56, 56])

        x = self.layer2(x)
        x, theta2 = self.stn2(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32) # 将 STN 输出的变换参数 theta2 转换为仿射变换矩阵的格式。
        # np.concatenate((theta2.detach().cpu().numpy(), tmp), axis=1)：将 STN 输出的变换参数 theta2 和上一步创建的数组在第一个轴上拼接，形成仿射变换矩阵。
        # np.linalg.inv(...)：计算拼接后的仿射变换矩阵的逆矩阵。
        # torch.from_numpy(...).cuda()：将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU 上。
        fixthea2 = torch.from_numpy(np.linalg.inv(np.concatenate((theta2.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cuda()
        # 使用 _fixstn 方法对 STN 输出的图像 x 进行两次反向仿射变换，使用逆矩阵 fixthea2 和 fixthea1。
        self.stn2_output = self._fixstn(self._fixstn(x.detach(), fixthea2), fixthea1)
        # after layer2 shape:  torch.Size([32, 128, 28, 28])

        x = self.layer3(x)
        out, theta3 = self.stn3(x)
        tmp = np.tile(np.array([0, 0, 1]), (x.shape[0], 1, 1)).astype(np.float32)
        fixthea3 = torch.from_numpy(np.linalg.inv(np.concatenate((theta3.detach().cpu().numpy(), tmp), axis=1))[:,:-1,:]).cuda()
        self.stn3_output = self._fixstn(self._fixstn(self._fixstn(out.detach(), fixthea3), fixthea2), fixthea1)
        # after layer3 shape:  torch.Size([32, 256, 14, 14])

        return out


def stn_net(args, pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(args, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model