# EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies（毫秒级延迟的准确视觉异常检测）

# 一、Paper Reading

## First: 

### 1、Abstract: 

```markdown
检测图像中的异常是一项重要任务，尤其是在实时计算机视觉应用中。在这项工作中，我们专注于计算效率，并提出了一种轻量级特征提取器，可以在现代 GPU 上在不到一毫秒的时间内处理图像。然后，我们使用学生-教师的方法来检测异常特征。我们训练学生网络来预测正常（即无异常训练图像）的提取特征。由于学生无法预测自己的特征，因此可以在测试时检测到异常情况。我们提出了一种训练损失，阻止学生模仿教师特征提取器超出正常图像的范围。它使我们能够大幅降低学生-教师模型的计算成本，同时改进异常特征的检测。我们还解决了具有挑战性的逻辑异常的检测，这些异常涉及正常局部特征的无效组合，例如对象的错误排序。我们通过有效地结合全局分析图像的自动编码器来检测这些异常。我们在来自三个工业异常检测数据集集合的 32 个数据集上评估我们的方法（称为 EfficientAD）。 EfficientAD 为异常检测和定位设定了新标准。它的延迟时间为 2 毫秒，吞吐量为每秒 600 张图像，可以快速处理异常情况。再加上其低错误率，这使其成为现实应用的经济解决方案，并为未来研究奠定了富有成果的基础.
```

### 2、论文中的图片、表格

![image-20231129102157056](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129102157056.png)

**图 1. NVIDIA RTX A6000 GPU 上的异常检测性能与每张图像的延迟。每个 AU-ROC 值是 MVTec AD [7, 9]、VisA [74] 和 MVTec LOCO [8] 数据集集合上图像级检测 AU-ROC 值的平均值**

![image-20231129102309888](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129102309888.png)

**图 2. EfficientAD-S 的补丁描述网络 (PDN) 架构。以完全卷积的方式将其应用于图像可以在一次前向传递中产生所有特征。**

![image-20231129102531166](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129102531166.png)

**图 3. 上行：位于输出中心的单个特征向量相对于每个输入像素的绝对梯度，在输入和输出通道上取平均值。下排：从 ImageNet [55] 中随机选择的 1000 张图像中第一个输出通道的平均特征图。这些图像的平均值显示在左侧。 DenseNet [25] 和 WideResNet 的特征图表现出很强的伪影。**

![image-20231129102727322](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129102727322.png)

**图 4. 训练期间由硬特征损失生成的随机选取的损失掩码。掩模像素的亮度指示相应特征向量的多少维被选择用于反向传播。学生网络已经在背景上很好地模仿了老师，因此专注于学习不同旋转螺钉的特征**

![image-20231129103009628](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129103009628.png)

**图 5. EfficientAD 应用于 MVTec LOCO 的两个测试图像。正常输入图像包含连接任意高度的两个拼接连接器的水平电缆。左侧的异常现象是电缆末端有一个小金属垫圈形式的异物。它在局部异常图中可见，因为学生和教师的输出不同。右侧的逻辑异常是存在第二根电缆。自动编码器无法在教师的特征空间中重建右侧的两条电缆。除了老师的输出之外，学生还预测自动编码器的输出。由于其感受野仅限于图像的小块，因此它不受附加红色电缆存在的影响。这会导致自动编码器和学生的输出不同。 “Diff”是指计算两个输出特征图集合之间的逐元素平方差，并计算其跨特征图的平均值。为了获得像素异常分数，使用双线性插值调整异常图的大小以匹配输入图像。**

![image-20231129103310196](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129103310196.png)

**表 1. 异常检测和异常定位性能与延迟和吞吐量的比较。每个 AU-ROC 和 AU-PRO 百分比分别是 MVTec AD、VisA 和 MVTec LOCO 上平均 AU-ROC 和平均 AU-PRO 的平均值。对于 EfficientAD，我们报告五次运行的平均值和标准差。**

![image-20231129103529213](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129103529213.png)

**表 2. 每个数据集集合的平均异常检测 AU-ROC 百分比（左）以及 MVTec LOCO 的逻辑和结构异常（右）。对于 EfficientAD，我们报告五次运行的平均值。仅在 MVTec AD (MAD) 上执行方法开发很容易导致设计选择过度拟合剩余的少数错误分类测试图像。**

![image-20231129104139754](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129104139754.png)

**表 3. 改变分位数位置时，MVTec AD、VisA 和 MVTec LOCO 上 EfficientAD-M 的平均异常检测 AU-ROC。这是基于分位数的图归一化和挖掘因子 phard 的两个采样点 a 和 b。将 phard 设置为零会禁用建议的硬特征丢失。我们实验中使用的默认值以粗体突出显示**。

![image-20231129110552361](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129110552361.png)

**图6.每GPU等待时间。在每个GPU上，方法的排名相同，除了DSR比FastFlow略快的情况下。**



![image-20231129112029812](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129112029812.png)

**图 7. EfficientAD 在 VisA 上的非精挑细选的定性结果。对于 12 个场景中的每一个，我们都展示了随机采样的缺陷图像、地面实况分割掩模以及 EfficientAD-M 生成的异常图。**

![image-20231129112122097](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129112122097.png)

**表 4. 累积消融研究，其中技术逐渐组合以形成 EfficientAD。每个 AU-ROC 百分比是 MVTec AD、VisA 和 MVTec LOCO 上平均 AU-ROC 的平均值。**



![image-20231129112231687](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20231129112231687.png)

**表 5. 孤立的消融研究，其中的技术是从 EfficientAD-S 中单独删除的.**

### 3、结论

```markdown
在本文中，我们介绍了EfficientAD，一种具有强大的异常检测性能和高计算效率的方法。它为结构异常和逻辑异常的检测设定了新标准。 EfficientAD-S 和 EfficientAD-M 在异常检测和定位方面都远远优于其他方法。与第二好的方法 AST 相比，EfficientAD-S 延迟降低了 24 倍，吞吐量提高了 15 倍。其低延迟、高吞吐量和高检测率使其适合实际应用。对于未来的异常检测研究，EfficientAD 是重要的基线和富有成果的基础。例如，其高效的补丁描述网络也可以用作其他异常检测方法中的特征提取器，以减少其延迟。

局限性。学生-教师模型和自动编码器旨在检测不同类型的异常。自动编码器检测逻辑异常，而学生-教师模型检测粗粒度和细粒度的结构异常。然而，细粒度的逻辑异常仍然是一个挑战——例如，螺丝太长了两毫米。为了检测这些，从业者必须使用传统的计量方法[62]。至于与其他最近的异常检测方法相比的局限性：与基于 kNN 的方法相比，我们的方法需要训练，特别是让自动编码器学习正常图像的逻辑约束。在我们的实验设置中，这需要二十分钟
```



## Sencond:







# 二、Code

```
https://github.com/nelson1425/EfficientAD
```





