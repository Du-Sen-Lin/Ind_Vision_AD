#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    # parser.add_argument('-a', '--mvtec_ad_path',
    #                     default='./mvtec_anomaly_detection',
    #                     help='Downloaded Mvtec AD dataset')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='/root/dataset/public/mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 用于标准化的均值和标准差值是在ImageNet数据集上训练的图像的典型值
])
# 对图像应用一种随机选择的颜色转换
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2), # 随机调整图像的亮度。
    transforms.ColorJitter(contrast=0.2), # 随机调整图像的对比度。
    transforms.ColorJitter(saturation=0.2) # 随机调整图像的饱和度。
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')
    # 检查是否使用预训练惩罚：设置为"none"以禁用ImageNet预训练惩罚
    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    # 如果数据集类型是 mvtec_ad，则按照建议，将训练集划分为训练集和验证集。划分的比例是 90% 的数据作为训练集，10% 的数据作为验证集。
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set 
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    # 如果数据集类型是 mvtec_loco，则只使用完整的训练集作为训练集，没有显式的验证集。
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')

    # 用于训练集的数据加载器: 
    # batch_size=1? 
    # shuffle=True表示在每个epoch开始时随机打乱数据，有助于模型学习更好的特征。
    # num_workers=4表示使用4个进程来加载数据，以提高数据加载的效率。
    # pin_memory=True表示将数据加载到固定内存区域，对于GPU加速可以提高性能。
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    # 根据是否启用了预训练惩罚（pretrain_penalty）来创建一个数据加载器，用于进行预训练。
    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else: # 如果未启用预训练惩罚， penalty_loader_infinite 被设置为 itertools.repeat(None)，即一个无限重复 None 的迭代器。这样的设计可能是为了在后续代码中进行条件检查而不引入额外的逻辑。
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels) # 创建一个自动编码器模型

    # teacher frozen
    teacher.eval() # 将教师模型设置为评估（推理）模式
    student.train() # 将学生模型和自动编码器模型设置为训练模式，通常用于启用梯度计算。
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader) # 通过调用 teacher_normalization 函数计算教师模型在训练集上的均值和标准差。这可能是为了进行输入数据的归一化，以确保训练和评估中的一致性。
    # 创建一个Adam优化器，用于更新学生模型和自动编码器模型的参数。 weight_decay (权重衰减)是L2正则化的项，有助于防止过拟合。
    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1) # 创建一个学习率调度器，它将在训练的0.95倍步数时将学习率乘以0.1。这是一种动态调整学习率的策略，可以在训练中逐渐减小学习率。
    tqdm_obj = tqdm(range(config.train_steps)) # 迭代指定数量的训练步骤
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st) # 使用 teacher 模型对 image_st（训练样本）进行前向传播
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std  # 并进行归一化
        student_output_st = student(image_st)[:, :out_channels] # 使用 student 模型对 image_st 进行前向传播
        distance_st = (teacher_output_st - student_output_st) ** 2 # 计算与教师模型输出的距离。教师模型输出和学生模型输出之间的欧氏距离的平方，用于度量它们之间的差异
        """distance_st 是包含了教师模型输出与学生模型输出之间距离的张量。通过计算这个张量的 99.9% 分位数，可以得到一个阈值，
        超过这个阈值的距离将被认为是异常值或异常情况.
        在训练过程中，这个硬阈值 d_hard 被用于定义损失，将超过这个阈值的距离纳入损失计算，以促使模型学习更好的表示。
        使用索引操作 distance_st[distance_st >= d_hard]，选取了距离大于或等于硬阈值的部分，计算了这些距离值的平均值，得到了硬损失 loss_hard。
        这个损失表示在距离大于硬阈值的部分，距离的平均值。        
        硬阈值的计算和硬损失的使用可能是为了过滤掉一些异常值或特别大的距离，以确保模型对正常样本有较好的拟合效果.
        """
        d_hard = torch.quantile(distance_st, q=0.999) #  torch.quantile 计算距离的硬阈值，并计算硬损失: 目的是计算张量 distance_st 中距离的硬阈值，该硬阈值对应于距离的 99.9% 分位数。
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None: # 如果有惩罚样本，则使用 student 模型对 image_penalty 进行前向传播，并计算惩罚损失。将困难特征损失（loss_hard）和惩罚损失相加得到总体损失 loss_st。
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard
        # 对 image_ae（自动编码器样本）进行自动编码器前向传播，计算与教师模型输出的距离 loss_ae 和与学生模型输出的距离 loss_stae。
        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output)**2 # 教师模型输出与自编码器输出之间的欧氏距离的平方
        distance_stae = (ae_output - student_output_ae)**2 # 自编码器输出与学生模型输出之间的欧氏距离的平方
        loss_ae = torch.mean(distance_ae) # 教师模型输出与自编码器输出之间距离的平均值，作为自编码器相关的损失。
        loss_stae = torch.mean(distance_stae) # 自编码器输出与学生模型输出之间距离的平均值，作为学生模型与自编码器的一致性损失。
        loss_total = loss_st + loss_ae + loss_stae # 最终的总体损失为 loss_total，包括硬损失、惩罚损失、自动编码器损失和自动编码器与学生模型输出之间的损失。

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0: # 定期打印当前损失。
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_total.item()))

        if iteration % 1000 == 0: # 每隔一定步数，保存当前的教师模型、学生模型和自动编码器模型。
            torch.save(teacher, os.path.join(train_output_dir,
                                             'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                             'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                 'autoencoder_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation 每隔一定步数，进行中间评估。在评估之前，将模型设置为评估模式（eval()）。
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
            # 调用 test 函数进行中间推理，并计算AUC。
            auc = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=None, desc='Intermediate inference')
            print('Intermediate image auc: {:.4f}'.format(auc))

            # teacher frozen 将模型重新设置为训练模式。
            teacher.eval()
            student.train()
            autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    """运行推理并计算模型在测试集上的 AUC

    Args:
        test_set (_type_): 包含测试集的数据加载器
        teacher (_type_): _description_
        student (_type_): _description_
        autoencoder (_type_): _description_
        teacher_mean (_type_): _description_
        teacher_std (_type_): _description_
        q_st_start (_type_): _description_
        q_st_end (_type_): _description_
        q_ae_start (_type_): _description_
        q_ae_end (_type_): _description_
        test_output_dir (_type_, optional): ）保存推理结果的目录. Defaults to None.
        desc (str, optional): _description_. Defaults to 'Running inference'.

    Returns:
        _type_: _description_
    """
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc): # 用 tqdm 来迭代测试集的每个样本。
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        # 获取模型输出的地图。
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        # 地图进行必要的处理，如填充和插值，以还原原始图像的尺寸
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)
        # 根据样本的标签（defect_class）确定真实标签 y_true_image，其中 "good" 类别对应 0，其他类别对应 1。
        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image) 
        y_score.append(y_score_image) # 记录模型输出的最大值作为 y_score_image。
    auc = roc_auc_score(y_true=y_true, y_score=y_score) # 使用 roc_auc_score 函数计算整个测试集上的AUC分数。
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    """预测模型输出的地图

    Args:
        image (_type_): 输入图像
        teacher (_type_): _description_
        student (_type_): _description_
        autoencoder (_type_): _description_
        teacher_mean (_type_): 教师模型的均值，用于归一化。
        teacher_std (_type_): 教师模型的标准差
        q_st_start (_type_, optional): 通道的起始和结束值，用于归一化地图。. Defaults to None.
        q_st_end (_type_, optional): _description_. Defaults to None.
        q_ae_start (_type_, optional): _description_. Defaults to None.
        q_ae_end (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    # 计算学生模型地图 map_st，表示教师模型输出与学生模型输出之间的距离。
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    # 计算自动编码器地图 map_ae，表示自动编码器模型输出与学生模型输出之间的距离。
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start) # 做归一化处理
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start) # 做归一化处理
    map_combined = 0.5 * map_st + 0.5 * map_ae # 综合的地图 map_combined
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    """对模型输出的地图进行归一化

    Args:
        validation_loader (_type_): 包含验证集的数据加载器
        teacher (_type_): _description_
        student (_type_): _description_
        autoencoder (_type_): _description_
        teacher_mean (_type_): 教师模型的均值，用于归一化
        teacher_std (_type_): 教师模型的标准差
        desc (str, optional): _description_. Defaults to 'Map normalization'.

    Returns:
        _type_: 返回这些分位数值，它们将在后续中间评估中使用
    """
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc): # 使用 tqdm 来迭代验证集的每个样本。
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st) # 学生模型和自动编码器模型的地图分别存储在 maps_st 和 maps_ae 列表中。
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae) # 将 maps_st 和 maps_ae 列表连接成张量。
    q_st_start = torch.quantile(maps_st, q=0.9) # 使用 torch.quantile 计算学生模型和自动编码器模型地图的起始（90%分位数）和结束（99.5%分位数）值。
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    """用于计算教师模型在训练集上的均值和标准差。
    这里使用了 @torch.no_grad() 装饰器，表示在计算过程中不需要梯度信息。

    Args:
        teacher (_type_): 教师模型
        train_loader (_type_): 用于加载训练数据的数据加载器。

    Returns:
        _type_: 函数返回计算得到的通道均值和通道标准差。
    """

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None] # 计算所有平均值的均值，形成通道均值 channel_mean。

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None] # 通道方差 channel_var
    channel_std = torch.sqrt(channel_var) # 计算其平方根得到通道标准差 channel_std。

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
