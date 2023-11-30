#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder

def get_autoencoder(out_channels=384):
    """创建一个自动编码器模型，该模型包含编码器和解码器部分

    Args:
        out_channels (int, optional): 输出通道的数量，影响模型的最后一层卷积层的通道数. Defaults to 384.

    Returns:
        _type_: _description_
    """
    return nn.Sequential(
        # encoder 编码器部分包含了一系列卷积层
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder 解码器部分包含了一系列上采样层和卷积层，用于将编码后的特征图重建为原始输入。
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    """创建小尺寸 PDN（ Patch description network）模型
    Args:
        out_channels (int, optional): 输出通道的数量，影响模型的最后一层卷积层的通道数. Defaults to 384.
        padding (bool, optional): 一个布尔值，表示是否应该在卷积层中使用零填充. Defaults to False.

    Returns:
        _type_: _description_
    """
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult), # 平均池化层 nn.AvgPool2d，用于下采样。
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False):
    """创建中等尺寸的 PDN（Patch Description Network）模型
    中等尺寸的 PDN 模型相比于小尺寸的模型，引入了更多的卷积层，可能更适合于学习更复杂的特征。
    Args:
        out_channels (int, optional): 输出通道的数量. Defaults to 384.
        padding (bool, optional): 一个布尔值，表示是否应该在卷积层中使用零填充. Defaults to False.

    Returns:
        _type_: _description_
    """
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

class ImageFolderWithoutTarget(ImageFolder):
    """只保留了super().__getitem__(index)返回的样本部分，而目标标签部分被丢弃。
    对于一些场景，比如无监督学习或者某些特定任务中，可能不需要目标标签，只需要图像数据。
    """
    def __getitem__(self, index):
        """super().__getitem__(index)调用父类ImageFolder的__getitem__方法，
        该方法默认返回元组(sample, target)，其中sample是图像数据，target是目标标签。
        return sample只返回图像数据，而不包括目标标签。
        """
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        """返回图像数据、目标标签和文件路径的元组。
        """
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    """它接受一个数据加载器（例如PyTorch中的DataLoader对象），并返回一个无限生成器，该生成器能够无限次迭代加载器中的数据。
    得到一个能够无限次遍历数据的生成器，这在一些训练循环中可能会有用。生成器会不断产生数据，直到手动中断或者结束程序。
    """
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
