## HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

有一篇讲得很好的文章：https://zhuanlan.zhihu.com/p/569958749



### Note

#### WER(Word error rate)

词错误率，是衡量语音识别准确率的评价指标之一，计算方式：

为了使识别出来的词序列和标准的词序列之间保持一致而需要进行替换，删除，或者插入某些词的数目，除以标准词序中的总词数得到的百分比

![image-20231011170032231](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231011170032231.png)



#### wav2vec

一个将音频信息encode转化为一个类似词向量的speech representation的模型，其思路类似word2vec，也采用了负采样的方法

具体模型方法如下

![image-20231011220351694](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231011220351694.png)

其中，x是输入音频信号，z[i]是每个x[i]经过encode之后的隐层表示，而c[i]就是综合考虑前n个z(z[i-n+1]-z[i])的context representation，也即模型的音频嵌入输出，此处的n是一个名为receptive field size的超参数，越大则包含越多的上下文信息

以上两个模型连接层都采用的是CNN，激活是ReLU

训练过程中，其预训练目标就是根据当前z预测将来某个z的损失，同时仅仅有正例是不够的，也采用了类似word2vec中的负采样方法，采样负例用于计算contrastive loss，具体loss公式如下

![image-20231011220820568](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231011220820568.png)

wav2vec可以显著提高ASR等任务的准确性



#### MFCC

梅尔频率倒谱系数（Mel Frequency Cepstral Coefficents）

一篇文章讲得很好：https://zhuanlan.zhihu.com/p/350846654

可以理解为对音频的人工特征，尤其注重于在**语音识别和说话人识别**这两个任务

背景知识：

+ 梅尔频谱：人耳对于声音频率的感知敏锐度是非线性的，在低频处敏感，高频处相对不敏感。通过梅尔标度滤波器变换为梅尔频谱，相当于处理成了线性关系（相对人耳感知为线性）
+ 倒谱：对一段时域音频信号，我们关注的是其共振峰部分，可以使用频谱包络线来描述共振峰的趋势。因此可以采用通过傅里叶变换和逆傅里叶变换，得到信号的频谱包络的方法，得到的包络线描述特征就是Mel频率倒谱系数，简称MFCC

获取方式：预加重、分帧、加窗、短时傅里叶变换（STFT）、mel滤波、离散余弦变换（DCT）



#### FBank

其实就是和MFCC类似的特征，将FBank特征进行DCT变换（倒谱）就得到了MFCC

获取方式：预加重、分帧、加窗、短时傅里叶变换（STFT）、mel滤波



### Introduction

语音信号的特点、难点：

+ 每个输入话语中可能包含多个声音，难以（使用类似CV中自监督的方法，实例分类等）进行分类
+ 缺少类似NLP中离散的声音token，没有词典，因此难以进行mask的预测损失计算
+ 声音单元间的边界模糊，难以切割



#### Self-Learning method

也称为Pseudo-labeling (PL)伪标签方法，与自监督的方法一样，也是用于特征学习提取的一类方法

其思路是，先利用一些标签数据，通过监督学习的方法训练一个教师模型，然后利用该教师模型对剩下的无标签数据打“伪标签”，作为实际student模型学习的数据

相比于PL方法，自监督的方法有以下的优点：

+ PL方法仅仅是模仿教师模型，因此其监督数据的大小和质量就非常重要；而自监督的方法无需标签数据，且其训练任务可以迫使模型学习如何将更多信息压缩到representation中
+ PL方法由于有标签数据，因此得到的特征可能更针对一个下游任务；而自监督方法的特征具有更强的泛化性



### Method

#### Model Structure

![image-20231013134336590](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231013134336590.png)

+ CNN Encoder：可以视作语音预处理的层，作用应该是将连续的语音信号分割成多个frame
+ Acoustic Unit Discovery System：一个聚合模型，可以视作无监督标签生成器（聚合类别数目也就是标签数），作用是根据声音特征来通过某种方式（如聚类）来产生标签，该标签会用于HUBERT的预训练任务中
+ Transformer：12层transformer，根据输入序列x来提取特征，该特征也即HuBERT模型的输出特征

#### Pretrain task

模型的预训练任务就是将输入序列中的一部分step进行mask，然后预测mask和非mask区域的标签z，计算其和真实标签z的差距（交叉熵损失函数），细节如下

+ mask方式采用和wav2vec 2类似的方法，首先选择p%的time-steps作为开始序号，然后从各个序号后面数l个steps作为要mask的区域，也即一次mask长为l的一段

+ 损失函数和在mask区域的损失（$L_m$）和非mask区域的损失（$L_u$）的加权和

  $L=\alpha L_m+(1-\alpha)L_u$

  其中$\alpha$是一个超参数，越接近于1代表越考虑mask区域

#### Learning the Hidden Units

这部分也就是Acoustic Unit Discovery System要完成的工作，提取离散的特征，这里需要分为两个部分去考虑

+ 特征提取方法：对语音提取特征，如MFCC
+ 聚类方法：如k-means，GMM等

更特殊一些，还可以采用诸如神经网络来提取离散特征



一个trick：Cluster Ensembles

聚类模型如k-means，当只有一个聚类模型时可能表现比较差（如k-means就对聚类中心数目或初始中心位置很敏感），因此考虑用多个不同参数设置的聚类模型来集成，loss直接修改为多个聚类模型总的预测损失之和即可

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231013170458241.png" alt="image-20231013170458241" style="zoom:80%;" />

模型集成还可以和向量量化一起使用（Product Quantization，PQ），感觉他的意思应该是可以将不同的聚类模型的中心视作PQ中的组，可以并行计算



Product Quantization

针对一个d维的向量，分为m组进行quantization，每组的维度为d/m

![image-20231013171755995](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231013171755995.png)



#### Iterative Refinement of Cluster Assignments

这也是一个很重要的方法，迭代优化

具体而言，HuBERT预训练中并不是从头到尾都使用初始MFCC然后k-means得到的聚类标签。当训练到一定程度后，通过将各层transformer的输出与ASR模型（一个外部模型）得到的真实音素序列进行比较，有**三个评价指标**，寻找最接近的一层输出重新进行聚类

第一次refine是使用第六层输出；第二层refine是使用第九层输出

