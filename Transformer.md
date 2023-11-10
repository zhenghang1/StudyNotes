### Attention

可以见[这篇文章](https://zhuanlan.zhihu.com/p/149490072)



#### hard attention和soft attention

计算softmax归一化后的权重后，hard是根据这个概率权重进行随机选择，而soft是根据概率进行加权求和

一般采用soft，因为hard方法太生硬且无法求导，需要通过Monte Carlo采样估计等方式计算梯度



#### global和local

指的是是否可以看到全局的信息，计算全局的注意力；还是只能看到附近的一部分信息，针对这部分计算注意力

local方式的一个方案：local-p，通过一个预估的计算，预计应当关注的位置pt，然后在softmax的结果中加了一个以 pt 为中心的高斯分布来调整 alignment 的结果

实际测试中，local-p的结果可能是最好的，但是一般还是采用global attention，因为复杂化的local attention 带来的效果增益感觉并不大。



#### self-attention

注意力的计算公式：$Atten(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})\cdot V$

其中，QKV都是Word的嵌入和QKV矩阵相乘得到后的结果矩阵

![image-20230930112245200](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230930112245200.png)



多头注意力

将输入数据X的embedding进行切分（按照头的数目），因此单个头的输入维度其实是$d_{model}/h$，将得到的h个z向量后进行拼接，再经过一个投影矩阵O矩阵，映射到目标长度



#### 时间复杂度分析

详见[这篇文章](https://zhuanlan.zhihu.com/p/264749298)

定义：n指的是数据序列长度，d指的是embedding的维度

矩阵乘法时间复杂度分析：

+ 矩阵A(n\*m),B(m\*n)，复杂度为**O(n\*m\*n)**,即O(n^2m)
+ 三个矩阵相乘，可以视作前两个相乘，其结果矩阵再和第三个相乘，**复杂度是相加而不是相乘**

Self-Attention包括**三个步骤：相似度计算，softmax和加权平均**，它们分别的时间复杂度是：

+ 相似度计算可以看作大小为(n,d)和(d,n)的两个矩阵相乘，得到一个(n,n)的矩阵：$O(n^2d)$

+ softmax直接计算：$O(n^2)$

+ 加权平均可以看作大小为(n,n)和(n,d)的两个矩阵相乘，得到一个(n,d)的矩阵：$O(n^2d)$

因此，Self-Attention的时间复杂度是$O(n^2d)$



多头注意力的时间复杂度和单个self-attention是一样的，原因是多头只是把输入维度切分了，令$a=d/h$，单个头的复杂度为$O(n^2a)$，则整体复杂度为$O(n^2a\times h)=O(n^2d)$



#### 代码

~~~python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.output_linear(x)
~~~

+ `view()`之前记得要加`contiguous()`，原因是`transpose()`操作转置矩阵，会使得矩阵数据内存地址不连续

~~~python
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
~~~



#### Attention优化加速

+ KV cache：字面意思，就是在GPT等Decoder模型推理过程中，由于是自回归的，每生成一个token后就需要将该token**拼接**后重新作为模型输入，seq长度会越来越长，每次需要读取的KV数据量也越来越大，因此可以考虑将以前计算过的KV都保存下来，以后输入只需要输入最新的那个token即可

+ MQA：Multi-query Attention，也就是Query继续保持多头，但是KV都只有一个头，会带来性能的些许下降，但是吞吐率能提高百分之三四十

+ GQA：Group-query Attention，和MQA一样，只不过此时是KV保留少部分的头，一组Query对应一个KV头，所以叫group

  ![image-20230930171421204](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230930171421204.png)

+ Flash Attention：主要是为了加速和节省内存，主要贡献包括：
  + 计算**softmax**时候不需要全量input数据，可以分段以增量方式进行计算；
  + 反向传播的时候，不存储attention matrix (N^2的矩阵)，而是只存储softmax归一化的系数。
+ Paged Attention：借鉴了操作系统分页的思想，将KV cache中的内容使用分页方式存储和检索，允许在非连续空间中进行存储，可以缓解原有方法中过度保留显存空间而造成浪费的问题（序列长度不稳定）



### Norm

使用正则化方法的原因：ICS问题（Internal Covariate Shift）

深度学习难以训练的一个重要原因在于神经网络中层与层之间存在着极强的**关联性**，并且目前绝大部分使用的都是基于**梯度下降**的方法进行反向传播训练。当网络的底层发生微弱变化时，这些变化会随着层数的增加被放大，意味着对于高层网络要不断进行参数更新以适应底层参数的变化，如此导致了训练的困难，很容易会出现**梯度爆炸或者梯度消失**的情况，导致模型训练失败。即：

+ 上层参数需要不断适应新的输入数据分布，降低学习速度。
+ 下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止。

+ 每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。

因此使用Norm，本质上就是使得层与层之间的输入保持在一个**稳定的分布**中，便于训练



缓解梯度消失：还可以选择使用非饱和的激活函数如Relu等



#### L1和L2

L1正则化可以使得模型参数尽可能稀疏，更多值为0

L2正则化可以使得模型参数绝对值尽可能小，保证其泛化性能



#### Dropout

解决过拟合问题，可以视作集成多个模型进行训练，而推理的时候是不进行Dropout的，故在训练的时候需要将权重乘以$\frac{1}{1-p}$



#### BatchNorm

沿着数据批次，整个数据批次中的每个特征做规范化

其中有两组需要学习的参数，均值和方差是直接根据batch数据计算即可，两组可训练参数（都是向量，针对每个特征）是线性变换参数$\gamma$和$\beta$，将规范化后的数据重新进行线性映射，防止由于规范化损害特征的表现能力

测试过程中，均值和方差使用**训练阶段记录下来的无偏估计值**

思考：

- BN适用于每个MiniBatch的数据分布差距不大的情况，并且训练数据要进行充分的shuffle，不然效果反而变差，增大训练难度；
- BN不适用于动态的网络结构和RNN网络。
- BN由于基于MiniBatch的归一化统计量来代替全局统计量，相当于在梯度计算中引入了噪声，因此一般不适用与生成模型，强化学习等对噪声敏感的网络中。



#### LayerNorm

针对单个样本，对其所有特征进行规范化，学习参数等和BN类似

思考：

- LN针对当个训练样本进行规范化，不像BN那样依赖于一个MiniBatch的数据，因此避免了不同样本之间的相互影响，可以适用于动态网络场景，RNN和自然语言处理领域
- LN不需要保存MiniBatch的均值和方差，相比BN计算量更小
- 对于那种特征明确的任务，LN是不合适的，如图像中的颜色、形状等特征，分别有特定具体的含义，使用LN会降低表现能力
- 研究表明，LN对于梯度的规范化可能才是其有效性的真正原因

```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```





#### WeightNorm

将权重参数向量分解为向量方向和向量模长，对其分别进行优化

对于矫正梯度更新方向有一定作用，同时具有自稳定作用（当更新值中噪音较多时，更新值就会变小），因此可以使用较大的学习率



正则化为什么有效：

+ 权重伸缩不变性
+ 数据伸缩不变性
+ 提高反向传播的效率：防止梯度爆炸或梯度消失
+ 具有参数正则化的效果，可以使用更高的学习率



### Transformer

[这篇文章](https://zhuanlan.zhihu.com/p/85221503)写的非常细致有用，下面整理贴一下具体的内容

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231005124558120.png" alt="image-20231005124558120" style="zoom:150%;" />

+ <1> Inputs是经过padding的输入数据，大小是[batch size, max seq length]。 

+ <2> 初始化embedding matrix，通过embedding lookup将Inputs映射成token embedding， 大小是[batch size, max seq length, embedding size]，然后**乘以embedding size的开方**。

+  <3> 通过sin和cos函数创建positional encoding，表示一个token的绝对位置信息，并加入到 token embedding中，然后**dropout**。

+  <4> multi-head attention 
  + <4.1> 输入token embedding，通过Dense生成Q，K，V，大小是[batch size, max seq length, embedding size]，然后按第2维split成num heads份并按第0维concat，生成新的Q，K，V，大小 是[num heads*batch size, max seq length, embedding size/num heads]，完成multi-head的操作。 
  + <4.2> 将K的第1维和第2维进行转置，然后Q和转置后的K的进行点积，结果的大小是[num heads\*batch size, max seq length, max seq length]。 
  + <4.3> 将<4.2>的结果除以hidden size的开方(在transformer中，hidden size=embedding size)，完成scale的操作。 
  + <4.4> 将<4.3>中padding的点积结果置成一个很小的数(-2\*\*32+1)，完成mask操作，后续 softmax对padding的结果就可以忽略不计了。 
  + <4.5> 将经过mask的结果进行softmax操作。
  + <4.6> 将softmax的结果和V进行点积，得到attention的结果，大小是[num heads\*batch size, max seq length, hidden size/num heads]。 
  + <4.7> 将attention的结果按第0维split成num heads份并按第2维concat，生成multi-head attention的结果，大小是[batch size, max seq length, hidden size]。Figure 2上concat之后还有 一个linear的操作，但是代码里并没有。
+  <5> 将embedding和multi-head attention的结果相加，并进行Layer Normalization。 
+ <6> 将<5>的结果经过2层Dense，其中第1层的activation=relu，第2层activation=None。 
+ <7> 功能和<5>一样。 
+ <8> Outputs是经过padding的输出数据，与Inputs不同的是，Outputs的需要在序列前面加上一 个起始符号“<\\s>”，用来表示序列生成的开始，而Inputs不需要。
+ <9> 功能和<2>一样。 
+ <10> 功能和<3>一样。 
+ <11> 功能和<4>类似，唯一不同的一点在于mask，<11>中的mask不仅将padding的点积结果置 成一个很小的数，而且将当前token与之后的token的点积结果也置成一个很小的数。 
+ <12> 功能和<5>一样。 
+ <13> 功能和<4>类似，唯一不同的一点在于Q，K，V的输入，<13>的Q的输入来自于Outputs 的 token embedding，<13>的K，V来自于<7>的结果。 
+ <14> 功能和<5>一样。 
+ <15> 功能和<6>一样。 
+ <16> 功能和<7>一样，结果的大小是[batch size, max seq length, hidden size]。 
+ <17> 将<16>的结果的后2维和embedding matrix的转置进行点积，生成的结果的大小是[batch size, max seq length, vocab size]。 
+ <18> 将<17>的结果进行softmax操作，生成的结果就表示当前时刻预测的下一个token在vocab 上的概率分布。 
+ <19> 计算<18>得到的下一个token在vocab上的概率分布和真实的下一个token的one-hot形式的 cross entropy，然后sum非padding的token的cross entropy当作loss，利用adam进行训练



总结：

+ Transformer可以理解为一个seq2seq模块，其中Encoder将可变长的输入转化为一个定长的中间表示（Encoder的输出），而Decoder是将固定长度的中间表示转化为可变长的输出
+ FeedForward层是一个两层的MLP，第一层的激活函数为 Relu，第二层不使用激活函数。其输出维度就是d_model，也即transformer block中间隐层的维度，而隐层维度是4*d_model
+ Embedding是token embedding和positional embedding的和
+ MHA：抽象形式中，是采用h个不一样的$W_Q,W_K,W_V$参数矩阵，对embedding的维度$d_{model}$切分为h份，视作h个子空间然后分别进行attention，最后再拼接；实际实现中，$W_Q,W_K,W_V$矩阵维度依旧是$d_{model}\times d_{model}$，映射得到QKV之后再进行最后一个维度的切分（切分为h份），分别计算得到attention之后，再将其拼接起来进行映射

+ 为什么<2>要乘以embedding size的开方？

  可能是因为embedding matrix的初始化方式是**xavier init**，这种方式的方差是1/embedding size，因此乘以embedding size的开方使得embedding matrix的方差是1，在这个scale下可能更有利于embedding matrix的收敛。

+ 为什么<4.2>的结果要scale，也即为什么scaled dot-product attention要进行scaled？

  为了防止当向量维度过大进行矩阵乘法时，得到的向量方差太大（假设原向量维度为dk，每个特征均值为0，方差为1，则点积后的结果均值为0，方差为dk），在softmax时容易趋向0和正无穷，使得梯度很小，训练速度很慢，因此可以除以$\sqrt{d_k}$使得其方差变回1,

+ Decoder中的source attention模块，输入的Q是Decoder的输入output经过第一层self-attention后，与$W_Q$计算得来的，而K和V是**整个Encoder模块**的输出结果C和$W_K$和$W_V$计算得到的，因此包含了Encoder输入的所有信息

  <img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231008122055069.png" alt="image-20231008122055069" style="zoom:50%;" />

+ Decoder的训练和测试的GAP

  + 训练时，Decoder的第一个输入是一个特殊token，如[BEGIN]，往后每一次的输入都是前一次输入加上输入序列下向后移一位的**ground truth token**的embedding（如I have an apple，第二个时间序列的输入就是第一个时间序列的输入（[BEGIN]的embedding+I的embedding）加上have的embedding）
  + 测试时，Decoder的第一个输入是一个特殊token，如[BEGIN]，往后逐个预测下一个单词，并将其对应embedding加到原输入中，重新作为新的输入

  区别在于新的time step加入的embedding，是ground truth的还是Decoder自己在上一个time step预测的







#### Mask

transformer里的mask，是在进行attention计算Q和K的乘积之后进行的（也即进入softmax之前），mask方式是根据mask值（是否为True）决定是否将对应位置处的值置为一个很小的数

Encoder中，mask只是为了将补齐到max_seq_length的padding部分mask掉

Decoder中，由于其设定是模型只能看到当前token之前的tokens（为了**保持训练和推理时的一致性**，训练时不应该能看到当前step后的token），所以需要将当前时间步后面的所有token也mask掉，具体方式为构建一个下三角矩阵，在主对角线右上的都是false（mask\[i]\[j]可以理解为第i个step时是否可以看到第j个token）

![image-20231006114133726](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231006114133726.png)



#### Positional Embedding

计算公式

![image-20231006115713048](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231006115713048.png)

可以看到，此处奇偶位置分别使用余弦和正弦函数，这种表示方法的好处是可以**表示相对位置信息**

![image-20231006115943577](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231006115943577.png)

代码

```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
```

先计算出最大长度的embedding值然后储存起来，每次forward就根据对应长度取pe对应的部分即可

实际实现中，也可以采用前半段计算sin，后半段计算cos然后拼接起来作为pe的方法，后接的线性层可以视作会对其进行下标重排



#### Label Smoothing

训练时的trick，目的是牺牲训练误差，提升泛化的性能，做法是：

当训练时，Decoder的loss是预测输出的softmax概率和实际输出label的softmax概率的偏差（最经典的如计算交叉熵），如下图

![image-20231007181201246](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231007181201246.png)

而label smooth指的就是将target output label，从上面的one-hot形式进行修改，比如，在 position #1 ，将1变成0.9，剩下的0.1 平均分配给其他的位置，也就是从[0 , 0 , 1 , 0 , 0 , 0]变成了 [0.025 , 0.025 , 0.9 , 0.025 , 0.025]，可以提升泛化性能



#### Transformer-XL

解决输入序列过长的问题，将输入序列进行分段，同时段与段之间采用类似RNN的循环机制传递信息，即有一个类似段之间的隐层信息的部分，用来在段与段之间建立长期依赖性

由于分段，因此段内不能再使用绝对位置编码，否则在第3段看来，第1段和第2段的第一个token位置是相同的，因此这里采用了一种根据词之间的相对距离来建模的相对位置编码（搞不懂）



### Bert

其实就是Transformer的Encoder模块，主要用于自然语言理解任务

Bert和Transformer中Encoder模块的区别：

+ Embedding不同，Bert新加了一个segment embedding，同时修改了positional embedding，改为可训练的embedding形式，测试时对于不同的token_id，直接在positional embedding中lookup即可

  因此，在embedding后面新加了一个LN（原始transformer中没有），目的就是为了防止梯度消失或者梯度爆炸，促进positional embedding的训练

+ Bert的输入可以是一个句子（不加[SEP]符号），也可以是一个句子对

+ Bert的输出同时有两种形式

  + pooler output：对应的是[CLS]的输出。
  + sequence output：对应的是所有其他的输入字的最后输出。

+ 使用Gelu激活代替了Relu



#### Bert的预训练任务

+ Mask Language Model(MLM)：类似于完词填空，随机mask输入中的一些词，然后预测这个词

  为了减轻pre-training 和 ﬁne-tuning的不匹配，Bert预训练时提出的一个方法是，对于被选中的待mask的词：

  - (1) 80%的情况是替换成[MASK]
  - (2) 10%的情况是替换为随机的token
  - (3) 10%的情况是保持不变 具体的code snippet如下

  loss是mask后的单词的Encoder输出，重新乘以Embedding Matrix的转置会得到在词表中的概率分布，计算其和真实token的one-hot概率分布的交叉熵

+ Next sentence order(NSP)：下一句子预测，根据CLS预测Sentence B是否是sentence A的下一句

  loss就是分类损失

Bert的预训练loss就是上面这两个任务的loss之和



Bert其实可以看做一个**强大的词嵌入模型**，相比于传统的Word2Vec和Glove（不提供任何上下文），Bert可以做到上下文建模



#### Bert处理长文本

由于Google初始设置的Positional embedding是固定512长度且预训练好的，因此无法突破512的长度，解决方法有多种，比较经典的有：

+ 重新设置一个更大的positional embedding，前512个用已有的进行填充，后面的从头训练
+ 滑动窗口法，固定窗口大小和步长，对源段落进行切分，保证其长度在512之内，多个切分样本需要保证一定的重合。该方法的缺点在于无法直接微调BERT，因为loss此时是不可微的；同时直接切分也会导致上下文信息缺失



#### Bert的权重共享

两个部分进行了权重共享

Transformer在两个地方进行了权重共享：

**（1）**Encoder和Decoder间的Embedding层权重共享；

**（2）**Decoder中Embedding层和FC层权重共享。

其中，（2）中Decoder的FC层作用在于由隐层状态计算输出词表对应单词的预测概率（要softmax），而embedding层作用则是从one-hot到嵌入表示，两个过程**类似逆过程**，可以权重共享，只要使用的时候进行转置即可，可以减少参数量，加快收敛



#### xlnet

创新融合了Encoder的auto-encoder的上下文信息能力（mask），和Decoder的auto-regressive方式中不会引入上下文gap的优势（只能看到当前位置之前的token）

引入了一个新概念叫因子分解序，可以理解为就是虚拟的句子排序方式，假设使用了1->4->2->3->5的因子分解序，意味着此时在模型计算2的attention时，会mask掉3和5，但是不会mask掉1和4，所以其实就是修改了Decoder模块中mask的方式

同一个句子在训练时可以采用不同的因子分解序，相当于是多个不同的训练语料

又引入了一个双流自注意力的方法，目的就是为了在实现中融入因子分解序（否则会产生不合理现象，如2个不同的因子分解序1->3->2->4->5和1->3->2->5->4，第1个句子的4和第2个句子的5在auto-regressive的loss下的attention结果是一样的，因此第1个句子的4和第2个句子的5在vocab上的预测概率分布也是一样的）



#### AlBert

主要创新点是两个参数减少的技术以及修改了预训练任务（SOP任务）

+ token embedding词向量矩阵的分解，在bert以及诸多bert的改进版中，embedding size都是等于hidden size的，这不一定是最优的。因为**bert的token embedding是上下文无关的**，而经过multi-head attention+ffn后的**hidden embedding是上下文相关的**，bert预训练的目的是提供更准确的hidden embedding，而不是token embedding，因此token embedding没有必要和hidden embedding一样大。

  AlBert将token embedding进行了分解，将embedding维度降低，然后再采用一个线性层将维度映射到hidden embedding的大小，token embedding参数量对比

  + Bert中是vocab size * hidden size
  + AlBert中是vocab size \* embedding size + embedding size \* hidden size

  只要embedding size << hidden size，就能起到减少参数的效果。

+ transformer encoder block参数共享，也就是将bert的12层不同的block进行参数共享，共用一套参数，显著降低参数量

AlBert虽然减少了参数量，但是并不会怎么降低推理时间，因为只是从串行进行12个不同的block，变成循环计算12次同一个block

SOP任务指的是sentence order prediction，正例就是NSP任务，反例是预测A是B的下一句，所以其任务其实得到了简化，两句话本身就有顺序关系了，只需要判断谁在前谁在后即可



#### RoBERTa

使用了更多的数据（160G），去掉了NSP的loss，采用了动态mask的新方法，同时修改了input format

+ 动态mask：Bert初始的mask是采用静态mask的，即在构造训练句子时就已经mask了，同一个句子可能采用多种不同的mask方式；而动态mask是每个句子要进入模型时才随机进行mask，效率更高
+ input format：由于取消了NSP任务，RoBERTa的input不再是两个用[SEP]分开的句子对，而是尽可能用同一个document内的句子塞满512的长度





## Note

### Additive & Dot-product Attention

Additive attention是将query和key拼接起来，然后将拼接向量输入一个线性层得到分数输出，而Dot-product是将query和key进行点积得到其分数输出

为什么称为Additive：

线性层的公式其实就是$y=f(w_1x_1+w_2x_2+\cdots+w_nx_n+b)$，因此对query和key拼接后输入线性层，本质上就可以写作$y=f(w_1q+w_2k+b)$，也就是加的形式

![image-20231007173922159](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20231007173922159.png)

两种Attention的比较：

+ Additive Attention可以处理key和query维度大小不同的情况（调整两个权重矩阵的维度即可）
+ Additive Attention的效果相比Dot-product Attention效果较好，因其是加法其得分方差不会随着输入向量维度大小改变而剧烈变化
+ 两种Attention的理论计算复杂度相似，但是Dot-product方法可以使用矩阵并行化加速，实际上会更快



### Weight Initialization

比较简单的有constant init和random init

Constant Initialization是将所有参数初始化为一个常数，显然是不行的，会使得每个计算单元对同一样例的输出和反向更新的梯度存在某种对称关系或甚至完全相同，导致神经网络的灵活性大打折扣

Random Initialization是按照某个概率分布（如高斯分布，均匀分布）将参数初始化，但是如何选择概率模型中的超参数便是一个问题，**Xavier Initialization和Kaiming Initialization正是为了解决这个问题而提出的**。



#### Xavier Initialization

思路是保持神经网络在训练过程中的信号强度不变（信号强度用Variance进行度量），推导可得需要使得参数的标准差为输入维度和输出维度的调和平均值



#### Kaiming Initialization

Xavier Initialization假设网络中没有激活，但实际中肯定是有激活的，激活会改变流动数据的分布，因此根据不同的激活，Kaiming initialization会有不同的初始化方式，但是其核心思想依旧是为了保持方差不变



### Word Embedding

#### Word2Vec

第一个火起来的embedding，采用简单的计算方法，极大加速了以往SVD矩阵分解等方法获取词向量的速度，使得在大规模语料上训练embedding成为可能



#### Elmo

+ 是一个被预训练好的多层双向LSTM语言模型，而不是一个词向量模型

- 它的词向量是在真实下游任务中产生的，所以根据输入不同，任务不同，同一个词获取的词向量是不同的
- 可以看作是特征提取的过程，在实际任务中，对于输入的句子，使用Elmo这个语言模型处理他，得到输出的向量，拿来做词向量。

Elmo的双向是由Bi-LSTM来的，这其实是一种伪双向，只是将两个方向的LSTM拼接起来，单个LSTM还是只能看到单向信息，并非同时获得前后向信息，其训练目标就是两个LSTM训练目标之和



### 缓解梯度消失、梯度爆炸

+ 传统方法：
  + 梯度剪切、正则化，限制在某个范围内防止爆炸，或者通过正则化惩罚过大的梯度，都是用来防止梯度爆炸的
  + 逐层预训练，每次冻结其它层，只训练一层，然后最终再合并起来联合微调（Hinton训练深度信念网络）
+ 使用relu、leakrelu、elu等激活函数，相比Sigmoid和tanh，计算量小的同时，还不会出现梯度消失的问题
+ batch norm，相当于将梯度通过规范化的过程，拉回到一个合理的范围内，防止梯度消失和爆炸
+ 残差结构
+ LSTM也是一个不容易发生梯度消失的网络（不是直接传递，长期依赖可以通过细胞状态保存）



