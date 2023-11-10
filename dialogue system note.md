# Dialogue Systems

## 1 Introduction

主要分为两类：task-oriented dialogue systems（TOD） 和 open-domain dialogue systems（OOD）



### 1.1 task-oriented dialogue systems（TOD）

传统的TOD，流水线结构，四个功能模块

+ Natural Language Understanding
+ Dialogue State Tracking
+ Policy Learning
+ Natural Language Generation

一些较新的TOD，采用端到端的方式，效果更好



### 1.2 open-domain dialogue systems（OOD）

主要分为三类：

+ generative systems，生成式的系统，通常采用seq2seq方式，根据用户句子和对话历史生成新句子，句子可能在训练集中从未出现过
+ retrieval-based systems，基于检索的系统，拥有一个response set，只需要从中找到合适的回应句子即可。受限于response set的大小，回应句子可能缺少上下文一致性
+ ensemble systems，集成系统，集成了上面两种系统



### 1.3 实现方式

+ rule-based，基于规则的方式，易于实现，但是灵活性差，只能用于特定领域
+ Non-neural machine learning based，基于非神经的机器学习的方式，使用template filling的方式来实现，灵活性得到了增强，但是依然不适用于大量不同的情境
+ deep learning-based（neural systems），基于深度学习的方式



## 2 Neural Models in Dialogue Systems

### 2.1 Convolutional Neural Networks（CNN）

包含三类层：

+ convolutional layers，卷积层，可以用于捕获局部特征
+ pooling layers，池化层，可以用于产生分层特征
+ feed-forward layers，前馈层

![image-20230216130957419](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216130957419.png)

CNN比较适合用于进行特征提取，但不适用于作为编码器（encoder）进行编码，因为其输入长度是固定的，卷积跨度有限

最近的研究不使用CNN作为对话编码器的主要原因是，CNN无法连续灵活地**跨时间序列步骤**提取信息



### 2.2 Recurrent Neural Networks（RNN）and Vanilla Sequence-to-sequence Models

#### RNN

CNN假设数据点之间相互独立，但实际上很多对话都需要结合上下文信息。RNN通过利用隐层状态，记录了历史信息，可以处理顺序信息流。

RNN的详解见[这里](https://blog.csdn.net/bestrivern/article/details/90723524)

其主要原理就是：$h_t=f(Ux_t+Wh_{t-1}+b)$，$y_t=SoftMax(Vh_t+c)$

RNN理论上来讲可以解决任意长序列的记忆问题，但在实际中，记忆长程信息的时候会出现梯度消失和梯度爆炸的问题，需要其他方式来解决，如LSTM



经典结构的RNN是输入输出等长的

<img src="C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216133533428.png" alt="image-20230216133533428" style="zoom:50%;" />

若要实现输入输出不等长，可以使用Encoder-Decoder结构（seq2seq），在Encoder模型中将整个句子处理为一个语义向量，表示整个句子的含义，再将该语义向量输入另一个Decoder模型，得到输出序列。



#### LSTM

LSTM详解见[这里](https://blog.csdn.net/qq_39478403/article/details/117928428)

特殊设计的RNN，除了隐层状态h外，还记录了称为细胞状态的C（代表过去的记忆），单个细胞内部由三个门组成：输入门，遗忘门（记忆门），输出门

![image-20230216141506835](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216141506835.png)

其中pointwise操作指的即点对点的操作（A*B即A中的每个元素与B中的对应元素相乘）；而门指的是让数据选择性通过的一个方法，由一个Sigmoid层和一个pointwise的乘法来组成（即每个元素乘上一个sigmoid产生的0-1的数）

<img src="C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216141754529.png" alt="image-20230216141754529" style="zoom:50%;" />

故三个门：

![image-20230216141837131](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216141837131.png)

![image-20230216141855274](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216141855274.png)

总结：

+ 输出门：决定当前时刻的单元状态 $c_t $有多少作为当前时刻的隐藏状态输出 $h_t$ (或者说输出有多少取决于记忆单元)
+ 输入门：决定当前时刻的隐藏状态输入$ x_t$ 和上一时刻的隐藏状态输出$ h_{t-1} $有多少保存到当前时刻的单元状态 $c_t$

![image-20230216141917536](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216141917536.png)





#### GRU

GRU的详解见[这里](https://blog.csdn.net/qq_39478403/article/details/118608699)

相比于LSTM，**减少了细胞状态C的信息流动，只保留了隐层状态h的信息传递**；将三个门简化为了两个门：更新门 (Update Gate) $z_t$ 和 重置门 (Reset Gate) $r_t$，参数更少，训练更快，同时能保持效果相当

![image-20230216151459617](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216151459617.png)

+ 公式 3：当前时间步重置门 $r_t$ 与上一时间步隐藏状态 $h_{t-1}$ 按元素乘法 $\odot$ 时，若重置门中的元素值接近 0，那么意味着重置对应隐藏状态元素为 0，即大部分丢弃上一时间步的隐藏状态$h_{t-1}$ ；若重置门中的元素值接近 1，那么表示大部分保留上一时间步的隐藏状态$h_{t-1}$ 。然后，将按元素乘法的结果与当前时间步的隐藏状态输入 $x_t$ 相加，再通过含激活函数 $tanh$ 的全连接层计算出候选隐藏状态 $\tilde{h_t}$，其所有元素的值域为 $[−1,1]$。由公式 3 可知，**重置门$r_t$控制了上一时间步的隐藏状态 $h_{t-1}$ 如何流入当前时间步的候选隐藏状态**。而上一时间步的隐藏状态可能包含了时间序列截至上一时间步的全部历史信息。**因此，重置门可以用来丢弃与预测无关的历史信息。**
+ 公式 4：更新门可以 **控制当前时间步的隐藏状态 $h_t$ 应如何被包含当前时间步信息的候选隐藏状态 $\tilde{h_t}$ 所更新**。假设先前的更新门 $z_{t'}$ 在好几个时间步 $t'$ 到当前时间步 $ t(t'<t) $之间一直近似为 1  ( $1-z_{t'}$就总近似为 0 )。那么，在时间步 $t'$ 到 $t$ 之间的输入信息几乎没有流入时间步 $t$ 的隐藏状态 $h_t$。实际上，这可视为是 **较早时刻 $t'$ 的隐藏状态 $h_{t'}$ 一直随时间保存并传递至当前时间步$t$** 。该设计可应对 RNN 中的梯度衰减问题，并更好地捕捉时间序列中长期依赖关系 (长期：时间步距离/跨度较大)。



GRU图示

![image-20230217124634691](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230217124634691.png)

![image-20230217124706961](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230217124706961.png)

![image-20230217124718803](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230217124718803.png)





#### Bidirectional Recurrent Neural Networks

双向RNN，使用双向信息，原理较为简单，就是增加一个反向的隐层信息即可

![image-20230217124939809](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230217124939809.png)



#### Vanilla Sequence-to-sequence Models (Encoder-decoder Models)

$Encoder : h_t = E(h_{t−1}, x_t)$

$Decoder : y_t = D(h_t, y_{t−1})$

输入输出不需要等长，但是也会存在一系列问题

![image-20230216133820792](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230216133820792.png)



###  2.3 Hierarchical Recurrent Encoder-Decoder (HRED)

是一个具有上下文感知能力的Encoder-Decoder模型，标准的HRED使用了三个RNN，其中两个分别作为Encoder和Decoder，按token level来处理每一个轮次中的对话信息；另一个RNN用来处理对话轮次间的信息传递，其输入是Encoder的最后隐层状态，输出作为Decoder的输入，本质上就是**在上述提到的Encoder-Decoder模型中间加上一个保存句子间信息的RNN**

![](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230217132640249.png)



### 2.4 Memory Networks

其特点就是设计了一个专用的内存模块，用来存储记忆，以解决RNN、LSTM等网络存在的，仅靠hidden states来进行薄弱记忆的问题。

经典的内存网络由五个模块组成：

![image-20230219110539390](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230219110539390.png)

+ 内存模块：有点类似数组，会对每一句话的记忆编码为向量后存入
+ Input模块：将各个输入都进行embed，转化为向量
+ Generalization模块：将输入的记忆存储在memory模块的对应位置（更新memory）
+ Output模块：根据输入的问题向量，在所有memory模块的记忆中，选出最相关的top k个记忆，组合得到输出向量，作为R模块的输入
+ Response模块：根据输出向量，编码生成一个自然语言的答案

![image-20230219110325857](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230219110325857.png)



### 2.5 Attention and Transformer

#### Attention

注意力机制，广泛应用于CV和NLP中，可以解决RNN等记忆上下文时将句子信息全部压缩到一个hidden vector中时信息丢失和梯度消失的问题。

attention的计算：

![image-20230219114821709](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230219114821709.png)



#### Transformer

见信息抽取论文阅读报告



### 2.6 Pointer Net and CopyNet

针对Pointer Net and CopyNet的详解文章见[这里](https://blog.csdn.net/qq_44766883/article/details/111995364)

#### Pointer Net

提出该模型时所使用的示例任务是凸包问题，输入一系列的点，要寻找一部分的点，可以包住所有的输入点，本质上就是个seq2seq任务（一列点，输出另一列点）

指针网络的设计参考了attention的思路，本质上也是对输入的不同部分计算注意力，区别在于：

+ 传统的seq2seq模型中，计算得到attention后，需要归一化后对所有的hidden states向量按attention加权求和得到context vector，该context vector输入Decoder中，解析生成对应的概率（针对一个固定的输出词汇表的概率），得到输出
+ 指针网络中，计算得到attention后，不再进行归一化，而是直接进行softmax，得到不同部分对应的概率（也即“指针”），这些指针可以用于指定原输入序列中的元素，作为输出。即指针网络的输出都是来源于输入，因此**非常适用于直接复制输入序列中的某些元素给输出序列**。
+ 可以看到，**传统seq2seq模型的输出来源于一个固定输出词汇表，而指针网络的输出来源于输入序列**。



由于指针网络的性质，当某些任务，输出可能直接来源于输入的时候（如摘要生成），就可以在模型上加入指针网络的部分，并根据一定概率与传统模型所生成的输出比较，取优者输出。





#### CopyNet

类似指针网络，只不过很多时候对话系统的知识仅来源于对话输入是远远不够的，还需要外部知识库的帮助。CopyNet就是一种将传统seq2seq网络和指针网络结合起来的网络，拥有从原句输入中复制以及生成不在输入中的知识的能力。

CopyNet中，计算一个词的概率的方式为，计算其生成概率（Pg）以及复制概率（Pc），并直接相加，如下。

![image-20230219180922771](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230219180922771.png)



### 2.7 Deep Reinforcement Learning Models and Generative Adversarial Networks

深度强化学习指的是，利用深度神经网络对强化学习框架的价值函数或策略进行建模。深层模式”是与“浅层模式”相对的。**浅模型通常指的是传统的机器学习模型**，如决策树或KNN。特征工程通常基于较浅的模型，耗时费力，而且过于指定和不完整。**深模型指的是深度神经网络模型**，易于设计，具有较强的拟合能力，自动利用数据中的层次特征，显著增强了语义表达能力和领域相关性。

下面讨论两种典型的强化模型，Deep Q-Networks和REINFORCE，分别属于Q-learning和策略梯度，这是强化学习的两个家族。



#### Deep Q-Networks

就是将传统强化学习中的Q-value计算，通过深度神经网络来进行建模，输入state S，输出一个Policy $\pi$ 

由于Reward是固定的，因此设计使得模型实际拟合的是Reward，公式如下：

![image-20230219191511586](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230219191511586.png)

其中外部大括号表示计算均方差，括号内第一部分target指的即进行该action可以获得的reward r和未来最大Qvalue的和，第二部分即为模型认为当前状态的Q value，它们的差就是实际reward减去现有模型认为在s下采取a时能得到的reward值，越小代表模型拟合的reward越准确。

deep Q-network的架构图：

#### ![image-20230219191058978](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230219191058978.png)

deep Q-network的训练技巧：

+ **Experience Reply**

  由于state的转移是彼此之间有相关性的，而如果每次进行一个动作，得到一个四元组（s, a, r, s’）后就拿来训练，则训练的样本间彼此相关，会影响训练效果，导致过拟合等问题。同时逐个获取四元组投入训练，也不利于高效计算。因此解决方法是，设立一个缓冲池，先缓冲一批四元组后，再每次随机抽取一个batch的四元组进行训练，加快训练速度。

+ **Target Network**

  如果只有一个DNN网络，则模型参数在训练过程中不断更新，那么它所追求的目标也是在不断改变的，即$\theta$改变时，不仅$Q(s,a)$改变了，$maxQ(s',a')$也改变了，容易不稳定，模型不容易收敛。

  解决方法是设立两个相同的DNN网络，一个称为Q-网络，用来进行参数更新；另一个称为目标网络（target network），其参数在训练过程中冻结，用来进行计算。在一定时间后，用Q-网络的参数更新目标网络中的参数，使得target network的参数在一定时间内都是稳定的。



#### REINFORCE

一篇针对policy-based 的深度强化学习网络的教程见[这里](https://blog.csdn.net/Pony017/article/details/81146374)

本质上就是策略迭代，基于策略更新的强化学习算法，此时不再有值网络，而是直接使用策略网络进行策略的计算，计算过程不加赘述。

REINFORCE的训练效率其实较低，问题很多，上述教程中介绍了另外两种改进后的算法，Actor-Critic和PPO算法



以上value-based和policy-based的算法都各有优势，Deep Q-Networks的样本效率更高，而REINFORCE更稳定。现代研究涉及更大的action空间，或者action是连续的，这些都使得value-based的模型不太适用，而policy-based的算法则直接通过策略网络预测动作，不限制动作空间，更适合于涉及较大动作空间的任务。



#### GANs

不加叙述





### 2.8 Knowledge Graph Augmented Neural Networks

涉及到图神经网络，没看明白



## 3 Task-oriented Dialogue Systems

面向任务的系统解决某一领域的特定问题，如电影票预订、餐厅餐桌预订等。

系统设计思路分为**模块化系统**和**端到端系统**

基于模块化的系统框架如下：

![image-20230219204842833](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230219204842833.png)

+ Natural Language Understanding (NLU)：理解用户的自然语言，将其抽取成为语义槽值，同时对领域和用户意图进行分类。当前的有些系统已经开始取消掉该模块，目的是减少错误在模块间的传播（比如第一个模块就出错了导致后面全部出错）
+ Dialogue State Tracking (DST)：根据用户action和槽值对，跟踪对话系统当前状态
+ Dialogue Policy Learning(PL)：根据DST的状态，决定下一步action
+ Natural Language Generation (NLG)：将PL生成的action，生成自然语言加以表述

通常，面向任务的系统还与外部知识库(知识库)交互，以检索关于目标任务的基本知识。



#### 3.1 NLU

NLU模块管理三个任务:域分类、意图检测和槽值填充

+ 域分类和意图检测是分类问题，它们使用分类器预测从输入语言序列到预定义标签集的映射。域指的是任务域，意图指的是该域中，用户请求所对应的意图标签（如电影系统中，域为“movie”，user intent为“find_movie”）
+ 槽值填充是一个序列标注问题（或也可视作序列生成问题），将可用信息转化为槽值对的形式

NLU的输入是ASR产生的文本信息；输出是一个N-best列表，表示act-slot-value三元组的概率分布



#### 3.2 DST

DST知乎教程见[这里](https://zhuanlan.zhihu.com/p/40988001)

两份DST相关论文：“The Dialog State Tracking Challenge Series: A Review”、“MACHINE LEARNING FOR DIALOG STATE TRACKING: A REVIEW”



DST的输入为NLU输出的N-best列表和agent已发出的action；输出为系统的状态State，用于作为下一步action的指导



传统的DST实现方式包括：人工制定规则、生成式模型（对数据集中存在的模式进行挖掘，学习出对话状态的条件概率分布，常见的方法包括贝叶斯网络和POMDP）



现在的DST实现方式：**使用神经网络**，有两种实现思路：

+ 使用预定义的，固定的槽值对

  可以被视作一个分类任务，有两种实现思路，一是一个模型在所有的类别中进行分类预测，二是分别设立多个模型，对每个slot进行单独的分类预测（称为**多跳分类**）

+ 不使用固定的槽值对

其中研究趋势为第二种方式，原因是该方式训练的模型，鲁棒性更强，同时灵活性也更好



#### 3.3 Policy Learning

根据DST生成的系统状态State，结合action集合，决策得到下一步action，即学习一个映射$f:S_t \rarr a_i \in A$

任务定义较为简单，训练较为简单，但是实际任务难度较大，如需要考虑一些诸如时间安排的细节（用户预定了两个小时的电影档位，然后打算去吃饭，那么agent应该意识到电影档位和餐厅档位之间的时间间隔必须超过两个小时，用于通勤）

两种主流的训练方式：

+ 监督学习：训练效果好，但是标注数据集需要大量人工劳动，决策能力受特定任务和领域的限制，迁移能力较弱
+ 强化学习



#### 3.4 NLG

传统的NLG采用流水线的设计方式，分为多个模块如下：

![image-20230222111132274](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230222111132274.png)

而深度学习模型出现后，流水线方式的NLG被端到端的单模块NLG所取代，同时，更多的方法被提出用以提高响应的可靠性和语义正确性等



#### 3.5 End-to-end Methods

流水线方式存在两个很明显的问题：

+ 流水线中的模块，有些时候并不是可微的，也即末端的错误难以传播到整个网络中的每个模块
+ 模块间的独立性使得单个模块的改进并不一定能提高整个系统的精度和质量，而且可能导致需要对其他模块进行重新训练等

此外，由于管道任务导向系统(如对话状态)中的手工特性，通常很难将模块化系统转移到另一个领域，因为预定义的本体需要修改。



为了解决这些问题，有两个主要的解决方式

+ 使管道系统的每个模块可微，那么整个管道可以看作一个大的可微系统，通过端到端的反向传播对参数进行优化
+ 只使用一个端到端的模块来执行知识库检索和响应生成，这通常是一个多任务学习神经模型



