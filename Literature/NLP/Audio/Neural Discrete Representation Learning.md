## Neural Discrete Representation Learning

### Note

#### PixelCNN

是DeepMind于2016年提出的图像生成模型，是一个自回归模型，相比于从前的自回归生成模型，区别只是采用了更先进的CNN作为基本模块

自回归生成模型用于图像领域时，思路就是**逐像素预测**

若是使用RNN，则模型的训练和推理都会非常慢，使用CNN时，训练时可以全部并行化处理，而推理时依然需要逐像素预测

PixelCNN需要解决的一个问题是，在训练时要确保模型有类似推理时的场景，也即不能看到未来的像素，因此需要**对卷积进行mask**。通过多层卷积的叠加，可以实现对左上角图像的完全感受视野（但是其实会存在右侧的感受盲区，需要进一步优化，**GatedPixelCNN**）

PixelCNN的损失是交叉熵损失，将像素预测视作256分类问题，但这其实也是有问题的，会**割裂类别之间的联系**。连续像素之间的差别是非常小的，因此这种预测错误不应当有很大的损失，应当有某种方法将这个因素考虑进去





#### Straight-through estimator(STE)

直通估计器，用于对离散化（量化quantization）过程中无法求导的问题进行导数估计，其思路非常简单，就是直接将输出的导数作为了输入的导数（其中，输入输出针对的是无法求导的这个计算），图示如下

![image-20231015194230331](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231015194230331.png)

一个可能的优化思路是，若是不可导操作对应多个不同的level（多个离散值），可以针对不同level采用不同STE，会提高估计精度，详见https://zhuanlan.zhihu.com/p/570322025

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231015195038644.png" alt="image-20231015195038644" style="zoom:50%;" />







### Introduction

离散向量指的是每一个特征都是离散值（int）而非连续值（float）

VQ-VAE（Vector Quantised VAE）其实是VAE（变分Auto Encoder，将隐层分布限制为一个正态分布）的一个变体，但是其更像一个AE而非VAE（对于隐层分布没有限制），也即其更多的是注重于数据压缩和重构，而非图像生成

Vector Quantised指的是Encoder编码得到的隐层状态应该是一个离散向量，此时不利于Decoder对图像进行重构，因此可以用类似NLP中embedding的思路，为每个离散的特征值映射到一个对应的嵌入向量中，这个embedding我们管它叫**codebook**

由于采用离散向量，因此会存在无法求导的问题，VQ-VAE中采用了STE的方法进行导数估计



### Method

#### embedding/codebook

关于Encoder编码得到离散向量，一个直观的想法是输出离散值，然后look-up到embedding space取对应的embedding向量，如下图

![image-20231015203711286](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231015203711286.png)

但是实际中可以采用一个更高效的方式来进行，跳过离散值这个中间变量，直接encode得到一个向量$z_e$，使其和embedding向量$z_q$相关联，如采用**最近邻方式**获取对应$z_q$，如下图

![image-20231015204428881](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231015204428881.png)

所以离散向量编码，并不一定需要出现离散值，更显著的特征应该是**有一个codebook**

在VQ-VAE的实现中，一个图像会使用多层卷积编码为m*m个长度为d的向量z（d就是codebook向量的长度），每个$z_k$都会被映射到codebook中的一个向量，整体称为$z_q$，可以理解为一个图像被编码为了m\*m个离散整数

（为什么要编码为m*m个：因为只编码为1个会降低表现力，且失去结构位置信息）



#### 梯度设计

对Encoder进行优化：

由于codebook最近邻映射这一步的argmin操作没有梯度，梯度无法传递到Encoder，因此采用了直通估计器的方式来设计求导：

$||x-decoder(z+sg[z_q-z])||^2$

在前向计算loss时，计算的是$x-decoder(z_q)$的误差，而反向传播时计算的是$x-decoder(z)$的loss，这样就允许我们对Encoder进行优化了（因为对z进行求导了，z有导数可以继续反向传播）



对codebook的优化维护：

最近邻映射机制，要求z和z_q是比较接近的，但是上一个loss中并没有对此加以约束，因此还需要增加一个$||z-z_q||^2$的项，目前loss为：

$||x-decoder(z+sg[z_q-z])||^2+\beta||z-z_q||^2$

实际上还可以做的更细节一些，考虑到z是由Encoder产生的，为了尽可能保证编码质量，不应该被这个loss项过分约束产生太大的变化，而codebook的z_q是相对比较自由的，因此可以考虑让z_q尽可能地靠近z，而不是反过来，因此修改loss如下：

$||x-decoder(z+sg[z_q-z])||^2+\beta||sg[z]-z_q||^2+\gamma||z-sg[z_q]||^2$，其中$\gamma<\beta$



#### 作为生成器

将VQ-VAE作为生成器，需要使用额外的模型对Encoder生成的隐变量z进行分布建模，比如PixelCNN，然后随机采样生成一个新的编码矩阵，codebook映射后送入decoder进行生成



