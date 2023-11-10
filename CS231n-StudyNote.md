## Lecture2 Image Classification

### Nearest Neighbor Classifier

大概思想就是，在数据集中找到1张和待预测图像距离最相近的图片，将其label作为预测结果

此处，一个非常重要的点就在于，如何界定两张图片的距离，两种常见的距离如：

+ L1distance，曼哈顿距离

  ![image-20220923151051492](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923151051492.png)

+ L2distance，欧氏距离

  就是各个像素点欧氏距离差的和



该类型算法存在的主要问题：

+ 准确率低
+ 训练过程简单（仅仅是储存下训练集的所有信息），但是预测过程需要进行大量计算（与训练集中所有图片进行比较），不符合应用层面的一般需求



### k - Nearest Neighbor Classifier（KNN）

Nearest Neighbor Classifier算法的优化版本，找到训练集中的k张最相近图片，并用其label进行综合预测，相比最近邻算法，其预测边界会更加顺滑

![image-20220923152130164](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923152130164.png)



### Validation sets for Hyperparameter tuning

#### **hyperparameters**

超参数，可以理解为选用的方法，而非实际上的某个参数，比如在距离选择中，我们在“L1 norm, L2 norm, there are many other choices we didn’t even consider (e.g. dot products)”中进行选择



#### Validation sets

> 一个原则：*Evaluate on the test set only a single time, at the very end.*

也就是说，只在整个模型训练结束的时候，才会涉及到测试集，除此之外整个训练过程中，包括验证的过程，都不应该利用测试集进行模型效果的验证，这样才不会导致模型实际上是对该测试集的过拟合的现象

因此一个合理的方式是在训练集中再拿出一部分作为验证集（Validation sets），该验证集可用来进行超参数选择等的性能测试中

验证集的方式还可以拓展为交叉验证（cross-validation），但是一般不常使用，因为其训练过程的计算开销太大![image-20220923153237881](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923153237881.png)

是否选择交叉验证的可能因素：

For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation. 



### Linear Classification

![image-20220923162006648](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923162006648.png)

![image-20220923162020193](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923162020193.png)

其中，W矩阵通常称为weight，b通常称为bias，偏置的设定通常是因为训练集中不同的类别图片的数目不同，或者其他我们的训练偏好等



线性分类器，可以理解为在高维空间中的一根线（图片可视作高维空间中的一个点），将一个类别以如下方式与其他类区分开：

![image-20220923162447484](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923162447484.png)



根据线性分类器的表达式，我们可以理解为，W矩阵的每一行就是对应一个类别的分类器。当训练结束后，W矩阵的每一行的参数值，实际上可视作给定了一个类别的一个模板，接近这个模板的图片会得到较高的得分：

![image-20220923162643270](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923162643270.png)

因为每一个类别对应的只有一个模板，因此事实上线性分类器对于区分同一个类别中的多模态现象表现是不佳的，在后续的网络中我们可以看到，有方式可以使得一个类别可以对应多个不同的模板



对于bias的一个trick：

为了防止多次矩阵相乘再相加，可以通过提高矩阵维度的方式将bias纳入矩阵，如下：

![image-20220923164304294](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923164304294.png)





## Lecture3 Loss Function and Optimization

### Loss function

#### Multiclass Support Vector Machine loss

![image-20220923170137478](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923170137478.png)

其思想是，对于网络给出来的各个类别的预测分数，若其正确结果得分最高且比其他得分高出delta的量，则该测试的loss为0，否则其loss值为最高得分减去正确结果得分+delta（即预测正确且高于一个给定安全范围delta，则认为该预测完全正确，loss为0，否则就还有改进空间）

![image-20220923170830907](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923170830907.png)



#### Regularization

为了防止过拟合，损失函数L中可以加入一个penalty项，不同的penalty项具有不同的选择倾向，可以使得使用该损失函数L所得的模型倾向于简单化，避免过拟合



#### Softmax classifier

![image-20220923171637763](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923171637763.png)

由损失函数的表达式可得，其是将所有的类别得分求e指数（转化为正数）后归一化，得到一个类似代表各类别的概率的结果，将该值取负对数即得该次测验的Li值（取负对数是为了使得其越大表示偏差越大，即结果越差，loss越大）

其目标是尽可能将正确结果的概率值趋向于1，而其他非正确结果的概率值和趋向于0



#### example：

![image-20220923172600476](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923172600476.png)



### Optimization

优化的原则就是选择梯度下降的方向进行优化，降低loss值

#### 计算grad值

+ 数值法：有限差分法

  ![image-20220923173312404](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923173312404.png)

​			其问题在于计算量过大，每次计算梯度都需要对所有的维度进行计算，不可行

+ 解析法：微分计算法，利用微积分知识，计算给定Loss Function的梯度表达式，并每次直接计算其梯度即可

  ![image-20220923173536051](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220923173536051.png)

实际运用中，我们通常使用解析法，然后用数值法来验证我们的解析解的正确性



#### Gradient Descent

每次计算损失值和梯度的时候，一般采用**Mini-batch gradient descent**的方式，每次随机挑选一部分样本，而不是对训练集中所有的样本都进行计算，伪代码如下：

~~~python
while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
~~~



#### Stochastic Gradient Descent (SGD)

指的是，Mini-batch gradient descent，挑选mini-batch的过程中，仅挑选一个随机样本进行loss和梯度的计算，不常使用



## Lecture4 Backpropagation Intuitions

### Backpropagation

![image-20220925110342575](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220925110342575.png)



![image-20220925110428397](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220925110428397.png)

给定各个变量的值，先前向传递得到所有计算节点的值，再利用链式法则，反向递归的计算每个计算结点的梯度值（最末一个的梯度为1），每个节点处的梯度计算都是非常简单的（上游传递的梯度*本地梯度）



计算公式

![image-20220925141650025](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220925141650025.png)









![image-20220925225709409](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220925225709409.png)









![image-20220925232516955](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220925232516955.png)

图像方面的化，不经常做归一化（除以标准差），因为各特征的差异并没有很大（归一化主要是为了将取值差异过大的多的特征统一一下





Weight的初始化：

+ 全部置为零：不行，相同的参数会使得许多神经元在后续进行完全相同的动作，得到完全一样的神经元，这是我们不想看到的
+ 将所有参数随机取为某些较小的值：也不好，因为一个输入x，在多次乘以一个值较小的Wi之后，所有的数值都会迅速缩小至0附近，经过一个较深的网络之后，可能得到很多0的输出。同时反向传播的时候，得到的梯度值也会很小，即参数几乎不会进行update
+ 将所有参数随机取为某些较大的值：也不好，类似tanh等的激励函数，当输入较大的时候，会发生饱和现象，即图像趋于平缓，梯度很小接近于0，导致反向传播的时候参数得不到有效更新







![image-20220926103253027](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220926103253027.png)

批量归一化，该层通常放置于全连接层和非线性层的中间，对全连接层的神经元输出进行调整

可以理解为，我们通常将全连接层的输出数据进行归一化是有好处的，但是在某些情况下也可能不是，我们可能希望数据能拥有自我调节的能力，所以在归一化得到x后，采用两个新的参数变量γ和β来得到y（放缩和平移），其中γ和β是可以通过学习来不断调整的





关于CNN的资料：https://github.com/cs231n/cs231n.github.io/blob/master/convolutional-networks.md
