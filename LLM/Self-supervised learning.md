## Self-supervised learning in Audio

自监督学习是一种在大规模无标签语料上，**通过数据本身构造监督信号**（比如对比学习或者cv中的拼图等等），来指导模型获取数据表征的方法。可以理解为，它没有人工标注的监督信号，但是有人工设计的从数据中自动获取监督信号的方法/模式。







## Contrastive Predictive Coding（CPC）

由google团队于2018年提出，开启了语音领域的自监督学习

该方法的核心思想是，将音频信号x抽取得到的表征特征z输入一个RNN，进一步得到语义特征c，要求语义特征c能预测得到后续时间步的表征特征z，也即要求$c_t$和$z_{t+1}$尽可能相近，于是该训练方法中可以认为，RNN可以根据当前表征$z_t$预测得到$z_{t+1}$，也即z确实对语音信息具有表征作用。

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112122331247.png" alt="image-20231112122331247" style="zoom: 67%;" />



该思路目前还有个问题，若是每次特征提取器和RNN都生成相同的z和c，也可以满足这个目标，因此还需要设计一个对比学习的部分，构建正负样本来完成区分。如将$z_{t+2}$，$z_{t+3}$，$z_{t+4}$等作为$c_t$的负样本，将$z_{t+1}$作为正样本

Loss设计为：

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112143536455.png" alt="image-20231112143536455" style="zoom:50%;" />

其中分子为正样本的相似度（概率），分母为负样本的相似度（概率）之和。





## Wav2vec

原理和CPC基本相同，不同的是全部采用了CNN作为特征提取器

模型架构如下

![image-20231112153958735](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112153958735.png)

和CPC模型基本一致，也是两个堆叠在一起的特征提取层

+ f: X->Z

  五层的CNN，可以理解为一个将高频音频信号X压缩为低频表示Z的特征提取器

+ g: Z->C

  九层的CNN，会同时考虑多个z（$z_i$，$z_{i-1}$，$\dots z_{i-v}$，），其中v是感受视野大小参数，加上CNN层数够多，所以可以看到过去比较大范围内的信息



Loss的设计：

相比于CPC，wav2vec设计的目标函数有如下变化：

+ 包含对未来多个z的预测，总的目标是每个预测步长k的预测目标之和
+ 由CPC中的分式改为了对抗的形式，一个正样本和多个负样本
+ 将相似度进行了sigmoid，规范化到了0-1之间

![image-20231112145346504](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112145346504.png)



训练结束后，wav2vec模型的**输出是语义表征向量c**



## VQ-Wav2vec

在Wav2vec的基础上，增加了对连续表征z进行离散化（VQ）的模块，可以支持将需要离散化输入的NLP模型用于语音领域

模型架构

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112172224685.png" alt="image-20231112172224685" style="zoom:50%;" />

其中第一层X->Z和第三层Z'->C都和wav2vec中是一样的，只不过增加了一层Z->Z'的量化层



### Quantization

本论文采用了两种不同的量化方式：Gumbel-Softmax和online k-means clustering

![image-20231112173325043](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112173325043.png)

+ Gumbel-Softmax

  是一种完全可求导的按概率采样方法（消除了argmax的不可导性），其算法的思路可以理解为将采样的过程前置。

  采样是为了在训练初期给予一定的随机性（不完全相信模型的输出）。传统的采样方法通常是$\epsilon$采样，一定的概率按照logits进行softmax后的概率取argmax，否则就随机采样。但这种方法存在两个问题，一是无法求导，二是在随机采样时完全抛弃了logits的信息。而gumbel-softmax可以解决这两个问题，算法如下：

  ![image-20231112180449196](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112180449196.png)

  这里的t是温度系数，温度越低则生成的分布越接近离散的one-hot向量，也即可以调节和one-hot的逼近程度（所以训练过程可以温度先高后低）

  采样过程前置的含义就是，此处是先加上随机变量后再进行softmax

  本方法中使用Gumbel-softmax进行离散化时，采用了直通估计器STE的方法：

  + 前向时，对gumbel-softmax得到的logits进行argmax得到one-hot向量，也即完成了离散化
  + 后向时，直接使用gumbel-softmax得到的logits的真实概率梯度

+ online k-means clustering

  采用欧式距离作为相似度度量，取距离最近（argmin）的codebook离散向量z‘作为z离散后的结果，依然也需要直通估计器来设计反向的导数。反向传播时的导数设计为$\frac{dL_k^{wav2vec}}{d\hat{z}}$，也即将二三层之间的导数直接作为一二层之间的导数（因为z和z'是接近的）。

  此时整体的目标函数设计为：

  ![image-20231112182152942](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112182152942.png)

  也是常规的加上了对z和z‘的约束，并使得z’更多的向z靠近



此外，这个VQ的过程还可以进一步变成分组VQ，也就是将d维度的z，切分成g段，每段单独进行VQ。这可以提高表现力（每个z现在可以变成g个离散化整数变量/codebook向量），同时还可以缓解mode collapse问题（只有个别z‘被经常使用，而一部分z’被忽视）。

再进一步，分组VQ的codebook变量甚至可以共享，这可以减少codebook的参数量，并且作者实验后发现这可以取得和不共享相接近的结果

在推理过程中，对于以上两个方法，都是直接argmax得到one-hot就行了



### Using discretized representation

当得到了离散化的表示后，vq-wav2vec可以进一步通过argmax的方式得到该离散化one-hot向量对应的下标（如[0,0,1,0]->2），作为该小段语音x的离散化表征（一个整数），因此一整段语音就可以转化为一个离散化的序列sequence。许多NLP模型的输入就是离散化的序列，比如BERT，通过这个方式可以将这些NLP模型应用到语音领域。（当然，应当也可以如原始wav2vec一样输出连续的语义特征吧？）

使用方式如下，将vq-wav2vec得到的离散序列，输入BERT等语言模型，进一步得到对应的输出表征，再输入到下游任务的Audio Model中

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231112202420557.png" alt="image-20231112202420557" style="zoom:50%;" />





## Wav2vec2.0

综合了wav2vec和vq-wav2vec的特点，主要是为了解决vq-wav2vec的使用pipeline中两个步骤割裂的问题：先获取到discrete speech units（离散编码），再放到下游BERT等模型中获取contextualized representations（语义表示）

Wav2vec2.0设计了一个pipeline，使得其可以同时对离散编码和语义表示对两个模块进行端到端训练，其显著成果是可以使用非常少量的标签数据就在下游任务上达到一个与以前SOTA结果接近





















