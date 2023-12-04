# RegAD

# 															--Few-Shot Anomaly Detection 

**Registration based Few-Shot Anomaly Detection: ⽆需微调即可推 ⼴，上交⼤、上海⼈⼯智能实验室等提出基于配准的少样本异常检测框架**

RegAD的模型架构：

![image-20231202122915126](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202122915126.png)

The model architecture of the proposed RegAD. Given paired images from the same category, features are extracted by three convolutional residual blocks each followed by a spatial transformer network. A Siamese network acts as the feature encoder, supervised by a registration loss for feature similarity maximization.（给定来⾃同⼀类别的图像对，通过三个卷积 残差块提取特征，每个残差块后跟⼀个空间变换⽹络。孪⽣⽹络充当特征编码器， 由配准损失来监督 特征相似度最⼤化）.

# 一、Paper Reading

## First: 

### 1、Abstract: 

```markdown
本文考虑了少样本异常检测（FSAD），这是一种实用但尚未充分研究的异常检测（AD）设置，其中在训练时仅为每个类别提供有限数量的正常图像。到目前为止，现有的 FSAD 研究遵循标准 AD 所使用的每类别一个模型的学习范式，并且尚未探索类别间的共性。受到人类如何检测异常的启发，即将有问题的图像与正常图像进行比较，我们在这里利用配准（一种本质上可跨类别推广的图像对齐任务）作为代理任务来训练与类别无关的异常检测模型。在测试过程中，通过比较测试图像及其相应的支持（正常）图像的注册特征来识别异常。据我们所知，这是第一个训练单个可泛化模型并且不需要针对新类别进行重新训练或参数微调的 FSAD 方法。实验结果表明，该方法在 MVTec 和 MPDD 基准上的 AUC 优于最先进的 FSAD 方法 3%-8%。源代码位于：https://github.com/MediaBrain-SJTU/RegAD
```

### 2、论文中的图片、表格

![image-20231202134033972](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202134033972.png)

**图 1. 与 (a) vanilla AD 和 (b) 每个类别一个模型学习范式下的现有 FSAD 方法不同，所提出的方法 (c) 利用特征注册作为 FSAD 的类别不可知方法，在-模型全类别学习范式。该模型使用多个类别的聚合数据进行训练，无需任何参数微调即可直接适用于新类别，只需要在给定相应支持集的情况下估计正态特征分布。**

![image-20231202143343766](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202143343766.png)

**图 2. 所提出的 RegAD 的模型架构。给定来自同一类别的配对图像，通过三个卷积残差块提取特征，每个残差块后面跟着一个空间变换器网络。孪生网络充当特征编码器，通过配准损失进行监督，以实现特征相似性最大化。**

![image-20231202144102593](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202144102593.png)



**表 1. MVTec 数据集上的 k-shot 异常检测结果，与最先进的方法进行比较。结果以 10 次运行的平均 AUC 百分比形式列出，并针对每个类别单独标记。最后一行还报告了所有类别的宏观平均分数。效果最好的方法以粗体显示。**



![image-20231202144512494](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202144512494.png)

**表 4. MVTec 和 MPDD 数据集上的异常检测和异常定位结果，与最先进的普通 AD 方法进行比较。结果以 AUC（以百分比表示）列出，作为每个数据集中所有类别的宏观平均得分。**

- 公式

![image-20231202143525140](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202143525140.png)



![image-20231202143932632](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202143932632.png)



![image-20231202143955646](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202143955646.png)



![image-20231202144018482](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202144018482.png)



![image-20231202144039654](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231202144039654.png)

### 3、结论

```markdown
本文提出了一种利用注册（一种本质上可跨类别推广的任务）作为代理任务的 FSAD 方法。只给每个类别一些正常样本，我们用聚合数据训练了一个与类别无关的特征注册网络。该模型被证明可以直接推广到新类别，不需要重新训练或参数微调。通过比较测试图像及其相应的支持（正常）图像的注册特征来识别异常。对于异常检测和异常定位，即使与使用大量数据进行训练的普通 AD 方法相比，该方法也具有竞争力。令人印象深刻的结果表明所提出的方法在现实世界的异常检测环境中具有很高的应用潜力。
```

## Second:

- 流程:

基于配准的少样本异常检测的框架。与常规的异常检测方法（one-model-per-category）不同，这项工作（one-model-all-category）首先使用多类别数据联合训练一个基于配准的异常检测通用模型。来自不同类别的正常图像一起用于联合训练模型，随机选择来自同一类别的两个图像作为训练对。在测试时，为目标类别以及每个测试样本提供了由几个正常样本组成的支撑集。给定支撑集，使用基于统计的分布估计器估计目标类别注册特征的正态分布。超出统计正态分布的测试样本被视为异常。

这项工作采用了一个简单的配准网络，同时参考了 Siamese [1], STN [2] 和 FYD [3]。具体地说，以孪生神经网络（Siamese Network）为框架，插入空间变换网络（STN）实现特征配准。为了更好的鲁棒性，本文作者利用**特征级的配准损失**，而不是像典型的配准方法那样逐像素配准，这可以被视为像素级配准的松弛版本。（https://www.linkresearcher.com/theses/3d5dafdb-cee1-4a79-ac76-f61dc2608b10）

- 本文的贡献主要如下：


```markdown
1、引入特征配准作为少样本异常检测（FSAD）的一种类别无关方法。据我们所知，这是第一种FSAD方法，可以训练单一的可推广模型，并且不需要对新类别进行再训练或参数微调。
2、在最近的基准数据集上进行的大量实验表明，所提出的RegAD在异常检测和异常定位任务上都优于最先进的FSAD方法。
```









# 二、Code