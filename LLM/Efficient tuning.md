## Parameter-efficient fine-tuning（PEFT）

### Introduction

全量微调（Full- fine-tuning）的资源需求太高，太过昂贵，且全量微调还有可能损失多样性，造成灾难性遗忘的问题（性能下降），因此考虑只微调一小部分参数，高效微调，来达到与全量微调接近的效果



高效微调大致可分为三类：

+ 增加额外参数（Additive）
  + Adapters类
    + Adapters
    + AdaMix
  + soft prompt类
    + Prompt Tuning
    + Prefix Tuning
    + Intrinsic Prompt Tuning（IPT）
+ 选取一部分参数更新（Selective）
+ 引入重参数化（Reparameterization-based）

下面是各个方法的性能

![image-20230918142742473](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230918142742473.png)



### Additive

#### Adapters

思路是在Transformer的每个子层模块（attention和全连接层）后，加入一些全连接层组成的Adapters（适配器），训练时只对这些Adapters进行梯度更新

以下是Adapters的结构

![image-20230917203934519](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230917203934519.png)

上图中，左侧是在Transformer模块中加入Adapters的图例，右侧是Adapters层的内部结构，包含两个线性层和一个非线性激活，其中线性层映射到一个较低的隐层维度（为了减少参数数目），这两个线性层分别称为down-project和up-project

相关研究还表明，只在attention后加Adapters（全连接层后就不加了），也可以达到相似的性能



#### AdaMix

在Adapters的基础上发展得到的方法，主要思路是利用混合专家模型的方法（MoE，Mixture of Experts），将Adapters方法中的每个Adapters，改为一个由多个Adapters（experts）组成的门控模块，细节如下

![image-20230918151843205](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230918151843205.png)

+ 门控模块并不是传统的需要计算来选择的，AdaMix直接使用了随机的方法，减少了计算量而不影响效果
+ Adapters层中的up-project层和down-project层，此处是相互独立的进行随机选择的
+ 为了使得训练更稳定，采用了一致性正则化（防止experts之间选择差异过大），即训练时每个批次会进行两次前向过程（上图的左和右），随机选择Adapters，





## Distributed parallel training

### Theory

分布式原语

看[这篇教程](https://zhuanlan.zhihu.com/p/478953028)，含代码

+ Scatter：分发，从一个rank将一个scatter list拆分后分发到不同rank
+ Gather：集合，将所有rank的数据集合拼接到某一个目标rank中**（重点是拼接）**
+ Reduce：归约，将所有rank的数据集合并运算（如求和）到某一个目标rank中**（重点是运算）**
+ All Gather和All Reduce：就是将结果发送到所有rank中



#### Data parallel

数据并行，简单而言就是多个GPU上分别保持一个模型的副本，然后将训练数据分batch部署到不同的GPU上，每个GPU独立进行本批次数据的计算



Pytorch DP、DDP和FSDP

+ DP是单进程多线程实现，只能支持单机情况，不支持分布式

  DP数据传输过程：

  1. 前向传播得到的输出结果gather到主cuda计算loss
  2. scatter上述loss到各个cuda
  3. 各个cuda反向传播计算得到梯度后gather到主cuda后，主cuda的模型参数被更新。
  4. 主cuda将模型参数broadcast到其它cuda设备上，至此，完成权重参数值的同步。

  四次数据传输

+ DDP是多进程实现，支持分布式，支持Apex混合精度训练，支持模型并行，每个进程都是独立的python解释器，可以避免GIL带来的性能开销（GIL是python全局解释器锁，在计算密集型的时候尽量不要使用多线程，会导致有些线程闲置不进行计算）

  DDP数据传输过程：

  1. 前向传播的输出和loss的计算都是在每个cuda独立计算的，梯度all-reduce到所有的CUDA(传输梯度)，这样初始参数相同，para.grad也相同，反向传播后参数就还是保持一致的，其他没有数据传输了。

+ FSDP（Fully Sharded data parallel）

  完全数据分片并行，将DDP中每个GPU单独的模型副本进行分片，每个GPU仅拥有模型权重参数的一部分，在需要前后向计算的时候All-gather收集全部权重，计算后再抛弃，减少显存占用



#### Pipeline parallel

流水线并行，属于模型并行的一种，层间并行，是将模型按层分割后分到不同的GPU上，业界常见的流水线并行方法有GPipe 和 PipeDream

朴素的流水线并行方式，一个严重的问题就是会产生Bubble，由于一次将整个批量的数据进行计算，因此当前GPU完成计算将结果传送到下一层所在GPU后，本GPU即开始进入**空闲状态**，且越早开始计算的GPU的空闲等待时间越长，当模型层数较多时会使得GPU利用率非常低（本质上可以视作只增加了显存，而GPU计算资源不增加，且还额外增加了数据传输的开销，训练反而还变慢了）

![image-20230927212537999](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230927212537999.png)

解决方式是采用微批次（Microbatch）的流水线并行，每个数据批次分割的很小，可以尽可能减小bubble

![image-20230927213031370](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230927213031370.png)



##### 流水线并行策略

分为 F-then-B 和 1F1B 两种模式，分别表示：

+ F-then-B：对于一个batch的数据，先全部进行前向计算，再全部进行后向计算，需要保存整个batch数据的激活值和梯度等变量，显存占用高；优点是模型权重参数版本一致，不需要额外保存权重

+ 1F1B：将微批次数据的前向计算和后向计算交叉进行，及时释放不必要的中间激活等变量，提高显存利用率；缺点是一个后向计算后假如更新权重，可能还有未完成后向计算的流水线模块，会导致权重版本不一致，需要额外保存

  1F1B策略，有两种不同的调度方案，区别在于每个设备上的模型层是否有交错，如下图

  + 非交错式调度（non-interleaved schedule）：一个设备拥有一段连续的模型层（如1-4,5-8）
  + 交错式调度（interleaved schedule）：一个设备拥有交错的模型层（如1-2和9-10），相当于分配到了多个流水线阶段（虚拟阶段，virtual stages），既节省内存又节省时间

  ![image-20230927220953321](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230927220953321.png)



##### Gpipe

Google最先发布的流水线并行方案，采用F-then-B策略，通过微批次方式减小bubble大小，采用**流水线定期刷新**的方式确保权重参数在一个周期内保持一致，不需要保存多个权重版本



##### PipeDream

非交错式的1F1B策略，由于1F1B带来的异步性，需要维持多个权重参数版本

![image-20230928100918478](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230928100918478.png)

PipeDream有两个变体

+ PipeDream-2BW，2BW指的是double-buffered weights，即通过即时计算，维持最多**不超过两个**权重版本

+ PipeDream-Flush，即采用和GPipe一样的思路，通过定期刷新流水线，保证只需要一个权重版本，代价是性能下降（定期刷新）



下面我们来看看几个知名的分布式训练框架中采用的流水线并行方案：

- 在 PyTorch 中，采用的是GPipe方案。使用的是F-then-B调度策略。
- 在 DeepSpeed 中，采用的是PipeDream-Flush，使用的是非交错式1F1B调度策略。使用这个调度方案，是为了促进最大规模的模型进行训练，在模型训练过程中中，存储多个权重缓冲可能会令人望而却步，我们的首要目标希望是一个“精确”的方法，而不需要收敛权衡。当然，DeepSpeed 引擎组件抽象出了流水线调度，你也可以自行实现其他的流水线调度方案。
- 在 Megatron-LM 中，基于PipeDream-Flush进行了改进，提供了一种交错式1F1B方案。
- 在 Colossal-AI 中，基于Megatron-LM的交错式1F1B方案，提供了非交错(`PipelineSchedule`) 和交错(`InterleavedPipelineSchedule`) 调度策略。



#### Tensor parallel

张量并行，属于模型并行中的层内并行，也就是将模型权重参数矩阵进行分解，分解后放到不同的设备上



#### Sequence parallel

序列并行，指的是对于输入序列进行长度上的划分，其中Colossal AI和Megatron-LM的实现思路和重点不同

+ Colossal AI：目的是为了缓解过长输入序列的计算问题。在transformer模块中，内存使用量是输入序列长度的二次方，因此对序列长度进行分割可以显著降低显存占用

  具体而言，将输入序列分割成多个块，并将每个块输入到其相应的设备（即 GPU）中。为了计算注意力输出提出了环自注意力（RSA），将环状通信与自注意力计算相结合，如下图

  ![image-20230928112657279](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230928112657279.png)

+ Megatron-LM：为了进一步分摊在张量并行中无法分摊的显存，具体即针对LayerNorm和Dropout的计算，按照输入序列长度进行分割，使得各个设备上面只需要做一部分的 Dropout 和 LayerNorm 即可



#### 多维混合并行

业界通常会组合多种并行训练方法，提高效率，常见组合如：

+ DP+PP

+ DP+PP+TP：也称作3D并行



### Framework

#### Pytorch



#### Deepspeed-ZeRo

代码内需要import deepspeed，wrap一下model，对engine求损失



DeepSpeed最重要的就是ZeRO，可以视作一个数据并行的Plus版本，详情可以看[这篇文章](https://zhuanlan.zhihu.com/p/394064174)

模型训练过程中，占用显存的部分可以分为Model States和Activation两部分，其中Model State可以分为以下三部分

+ Optimizer States
+ Gradient
+ Model Parameter

其思想是针对训练过程中的Model State进行切分然后分摊到多个GPU上，如下图所示

![image-20230928155931110](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230928155931110.png)

+ ZeRo-1：进行Optimizer States的切分，其原理是Optimizer States只在参数更新的时候被用到（参数和Optimizer States运算得到新的参数），而在前后向传播的时候都不用到，因此切分到不同GPU后，GPU可以只对自己那部分Optimizer States对应的参数进行更新，然后所有GPU**通过All-gather获取完整的更新后的模型参数**

+ ZeRo-2：进行Optimizer States和Gradient的切分，其实就是在一的基础上，由于一段Optimizer States只需要由一段Gradient进行运算，因此完全可以将Gradient也进行切分，每次单个GPU反向得到梯度后，只需要增加一个All-reduce将所有GPU上该段Gradient平均即可，然后**按需保留**

+ ZeRo-3：进行模型参数的切分，实现非常复杂，因为需要保证在需要的时候及时拿到参数，不需要的时候又要立马释放。通过submodule的hook操作来实现



#### Megatron-LM

张量并行，实现简单，只需要在原生pytorch代码中增加少量代码即可，不需要新的编译器



#### Megatron-DeepSpeed

微软的框架，其实就是字面意思，将Megatron的张量并行和DeepSpeed中ZeRo的数据并行、管道并行结合，可以视作3D并行的一个具体实现
