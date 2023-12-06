import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='/root/dataset/public/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    return parser.parse_args()


def main():

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d)) # idx 包含了 d 个从 0 到 t_d - 1 范围内随机选择的索引的 PyTorch 张量。这

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output): # 该函数将被注册到模型的某些层，以捕获它们的中间输出。
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0] # 图像 ROC-AUC 曲线
    fig_pixel_rocauc = ax[1] # 像素级 ROC-AUC 曲线

    total_roc_auc = [] # 总体 ROC-AUC 分数的列表。
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES: # 分类别
        # 加载训练和测试数据集
        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
        # 创建字典，存储模型在训练和测试集上的中间输出
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name) # 训练集特征文件的保存路径
        if not os.path.exists(train_feature_filepath): # 训练集特征文件不存在，提取特征
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction 模型预测
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs 获取中间层输出
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs 初始化 hook 输出
                outputs = []
            for k, v in train_outputs.items(): # 对每个键值对应的输出进行拼接，形成完整的特征张量
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat 完成了对不同层中间输出的拼接，得到了一个包含了多个层特征的张量 embedding_vectors。
            """将预训练模型中三个不同层的feature map对应位置进行拼接得到embedding vector，
            这里分别取resnet18中layer1、layer2、layer3的最后一层，模型输入大小为224x224，
            这三层对应输出维度分别为(209,64,56,56)、(209,128,28,28)、(209,256,14,14)，
            这里实现1中是通过将小特征图每个位置复制多份得到与大特征图同样的spatial size，然后再进行拼接。
            比如28x28的特征图中一个1x1的patch与56x56的特征图中对应位置的2x2的patch相对应，
            将该1x1的patch复制4份得到2x2大小patch后再与56x56对应位置的2x2 patch进行拼接，下面的代码是通过F.unfold实现的
            """
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension 随机选择维度, 为了在特征维度上进行降维或选择特定的通道。
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution 从 embedding_vectors 中选择的通道进行多元高斯分布的计算
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy() # 计算均值向量 mean：
            """C 是特征的数量（通道数）。
                H * W 是空间维度的大小。
                对于每个类别，cov 是一个大小为 (C, C, H * W) 的三维数组，其中每个元素 (i, j, k) 表示第 i 个特征和第 j 个特征在空间位置 k 上的协方差。
            """
            cov = torch.zeros(C, C, H * W).numpy() # 初始化一个全零的协方差矩阵 cov，形状为 (C, C, H * W)。
            I = np.identity(C) # 使用 np.identity(C) 创建一个 C 维单位矩阵。
            """协方差矩阵的可逆性
            协方差矩阵的可逆性是一个重要的性质。协方差矩阵通常是对称正定的，这意味着它是一个实对称矩阵，并且对于任意非零向量，其与协方差矩阵的乘积都是正的。
            为了确保协方差矩阵的可逆性，添加了一个小的正则化项 0.01 * I 到协方差矩阵。这个小的正则化项通常被称为"平滑项"，用于防止协方差矩阵的奇异性（不可逆性），尤其是在估计样本协方差矩阵时，当样本数量较小时可能会出现。
            奇异性是指矩阵不是满秩的，即存在线性相关的列或行。对于协方差矩阵而言，奇异性可能导致矩阵不可逆。
            当样本数量相对较小时，样本协方差矩阵可能会变得不稳定，特别是当特征的数量大于样本数时。这种情况下，样本协方差矩阵可能是奇异的，即存在线性相关的特征，导致矩阵不可逆。
            添加正则化项的目的是通过在对角线上增加一个小的常数，来防止协方差矩阵的奇异性。这个小的正则化项有时被称为"平滑项"，它确保协方差矩阵在数值上是稳定的，并且保持了一定的数值非奇异性。
            mean: (100, 3136) cov: (100, 100, 3136)
            100个维度: 56x56, 1x1视野为4x4,输入分辨率为(224x224). 每个维度一个均值。
            
            """
            for i in range(H * W): # 计算协方差矩阵 cov：
                # 使用 LedoitWolf 方法估计协方差矩阵
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                # 直接使用 np.cov 函数计算协方差矩阵
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I # 对每个位置计算对应通道的协方差矩阵。添加一个小的正则化项 0.01 * I，以确保协方差矩阵的可逆性。
            # save learned distribution 计算得到了均值向量 mean 和协方差矩阵 cov，然后将这些参数保存到文件中。
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features 每个批次包含图像 x、真实标签 y 和标签对应的掩码 mask。
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction 模型预测, 但在这里并没有使用预测结果 
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs 通过注册的钩子（hook）获取中间层的输出，并将这些输出存储在 test_outputs 字典中。
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items(): # 对于字典 test_outputs 中的每个键值对，将对应值（张量列表）按照维度0进行拼接
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat 特征拼接
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension 随机选择维度
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix 计算距离矩阵
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy() # 将 embedding_vectors 重新调整为形状 (B, C, H * W)。
        dist_list = []
        for i in range(H * W): # 计算每个样本到训练集中对应位置的均值的马氏距离
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        # 转置距离矩阵并调整形状为 (B, H, W)
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample 上采样
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map , 对分数图应用高斯平滑
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization 分数图的归一化操作
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score 计算了图像级别的 ROC AUC
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1) # 计算每个图像的最大分数
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores) # 计算 ROC 曲线
        img_roc_auc = roc_auc_score(gt_list, img_scores) # 计算图像级别的 ROC AUC 分数
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc)) # 保存和打印结果：
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold 获取最优阈值（Optimal Threshold）以用于二值化分数图
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten()) # 使用 precision_recall_curve 函数计算 Precision-Recall 曲线的精确度（Precision）、召回率（Recall）和阈值。
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)] # 获取最优阈值：使用 thresholds[np.argmax(f1)] 获取使 F1 值最大化的阈值

        # calculate per-pixel level ROCAUC 像素级别的 ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc)) # 像素级别的 ROCAUC
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name) # 保存图像和绘图
    # 计算了平均的图像级别 ROCAUC 和像素级别 ROCAUC
    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    """对输入的图像数据进行反标准化（denormalization）。
    反标准化的目的是将经过标准化（normalization）的图像数据还原为原始的像素值范围。
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    """在空间维度上对两个输入张量进行拼接，实现了一种局部的特征融合操作。
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
