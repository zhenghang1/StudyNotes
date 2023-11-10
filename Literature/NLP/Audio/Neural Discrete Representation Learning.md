## Neural Discrete Representation Learning

### Note

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

但是实际中可以采用一个更高效的方式来进行，跳过离散值这个中间变量，直接encode得到一个向量$z_e$，使其和embedding向量$z_q$相关联，如采用最近邻方式获取对应$z_q$，如下图

![image-20231015204428881](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231015204428881.png)