## PyTorch 学习笔记

### 图片操作

~~~python
from PIL import Image

img = Image.open(path)
img.show()
~~~



### os模块中两个常用函数

+ `os.path.join()`：可以将后续几个字符串路径按照对饮操作系统的要求进行拼接
+ `os.listdir(path)`：返回路径对应的文件夹下所有文件/目录的名称组成的列表，可用来对一个目录中所有对象作批量处理



### Dataset类

pytorch中的数据集类

~~~Python
from torch.utils.data import Dataset
~~~

其使用方式为：自定义一个类，并继承Dataset，类中必须重写以下两个方法：

+ `__getitem__(self, index)`：根据index返回一个对象item，即支持`x = my_dataset[idx]`
+ `__len__(self)`：返回该数据集的大小

一般是在`__init__()`函数中读取数据集并存储，在`__getitem__`中按索引返回即可



### 补充：批量生成label文件

~~~python
import os

root_dir = r"data/train"
keyword = input()
source_dir = keyword + "_image"
imgs_path = os.listdir(os.path.join(root_dir, source_dir))
label = source_dir.split("_")[0]
target_dir = keyword + "_label"

for i in imgs_path:
    file_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir, target_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)
~~~





### TensorBoard的使用

可以理解为一个画图工具，其工作方式为：读取logs文件夹中的事件文件，并根据其内容绘图，命令如下：

`tensorboard --logdir=logs`

其中`logs`为自定义的目录文件夹，默认将使用本地6006端口，若多用户同时使用为了防止占用的话，可以使用`--port=port_number`的参数修改端口号



使用：

~~~python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")		# 参数为自定义的log文件夹名字

'''
......
'''

writer.close()
~~~

主要学习两个方法：

+ `add_scalar(tag, scalar_value, global_step=None,...)`，添加一个数据，主要有三个参数，其中`tag`可理解为当前图的标题，`scalar_value`可理解为y轴数据，`global_step`可理解为x轴数据

  ![image-20220914193606433](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220914193606433.png)

+ `add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`，添加一张图片，其中`tag`依旧是当前标题，`img_tensor`是添加的图片（格式要求，下面详细叙述），`global_step`是当前图片是当前测试的第n步的结果，`dataformats`是指添加的图片的shape（高度H，宽度W和通道数C），如与默认不同则需要指明

  `img_tensor`的格式：`torch.Tensor, numpy.array, or string/blobname`中的其中一种，常见的jpg，png格式是不行的，可以如下方式将其转化为`numpy.darray`格式：

  ~~~python
  import numpy as np
  from PIL import Image
  
  img_PIL = Image.open(img_path)
  img_array = np.array(img_PIL)
  ~~~

  注意`numpy.darray`格式的shape为：HWC，与`add_image`参数`dataformats`默认值不同，需要额外指定如下：

  ~~~python
  writer.add_image("test", img_array, 1, dataformats="HWC")
  ~~~

  可通过`print(np.array.shape)`的方式来查看shape



### 使用OpenCV读取图片

~~~python
import cv2

img_cv = cv2.imread(img_path)
~~~

使用`imread()`读入的图片格式为numpy格式



### transforms

> 用于进行多种数据类型之间的转换

#### ToTensor

将一个PIL Image或者numpy.ndarray对象转换为tensor对象，使用方式：

~~~python
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
~~~



#### ToPILImage

和ToTensor的使用方式基本一致，只是转换对象不同而已



#### Normalize

其构造函数：`def __init__(self, mean, std, inplace=False)`

需传入两个列表，分别表示标准化的均值和标准差，列表维度为图片的维度（一般PILImage使用RGB，就为3），标准化公式类似概率统计中随机变量的标准化：`output[n]=(input[n]-mean[n])/std[n]`

返回标准化后的对象



#### Compose

连续进行多个transform操作，构造函数为：`def __init__(self, transforms)`，其中transforms为多个transforms的实例对象的列表，就是对某个对象按序进行多个transform操作

根据其call方法的实现，可知其原理：

~~~python
def __call__(self, img):
    for t in self.transforms:
        img = t(img)
    return img
~~~

使用方式：

~~~python
img = Image.open("data/train/ants_image/0013035.jpg")
trans_tensor = transforms.ToTensor()
trans_resize = transforms.Resize((512, 512))
trans_compose = transforms.Compose([trans_resize, trans_tensor])
img_resize_tensor = trans_compose(img)
~~~



#### Resize

改变图片尺寸



其余类不加赘述，都可以通过官方文档等轻松掌握其用法



### torchvision中的数据集使用

>torchvision中为我们提供了一系列标准的数据集，详见[这里](https://pytorch.org/vision/stable/datasets.html#built-in-datasets)



以CIFAR10数据集作为示例：
![image-20220915090042030](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915090042030.png)

其中，`root`指的是需要指定数据集的存储位置；`train`指的是下载训练集or测试集；`transform`指的是一个transforms变换，即对数据集中所有的图片统一进行某个变换；`target_transform`是指对target进行一个transforms变换；`download`是指是否自动下载该数据集（通常设置为True）

一个比较常见的使用方法：
~~~python
import torchvision

train_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=True, download=True)

print(train_dataset[0])

'''
Output:
(<PIL.Image.Image image mode=RGB size=32x32 at 0x18C5C8B5BE0>, 6)
'''

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False,
                                            transform=dataset_transform, download=True)
#将数据集中的PIL Image图片先转换为tensor类型
~~~

即得到的Dataset默认也是重写了`__item__()`方法的，可以直接根据索引返回一个item，包含一个img和一个target，其中target指的是在classes属性中的下标（即用数字代表某个label），可按如下查看数据集的classes：

~~~python
print(train_dataset.classes)
'''
Output:
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
'''
~~~



### DataLoader

![image-20220915094403592](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915094403592.png)

几个较为常用的参数：

+ dataset：指定要加载的数据集
+ batch_size：一次加载并打包的数据量
+ shuffle：是否打乱加载顺序（为False的话，两次不同的加载取出的数据顺序都会是一样的）
+ num_workers：加载使用的线程数
+ drop_last：是否将最后不足batch_size大小的数据忽略

用法：

~~~python
from torch.utils.data import DataLoader

test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, drop_last=False)
for data in test_loader:
    imgs, targets = data
    # some operation on imgs and targets
~~~



### 神经网络

主要的部分都包含在torch的`nn`模块中，nn指的是neural networks

#### 基类nn.Module

![image-20220915204317384](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915204317384.png)

所有的自定义神经网络，都需要继承自`Module`类，并实现其`forward`方法

+ `__init__`方法中，调用基类的`__init__`方法，然后定义所需的各层操作
+ `forward`方法中，对输入进行各卷积操作，并返回结果对象



#### 卷积操作

> 主要解释各参数的含义

![image-20220915205717029](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915205717029.png)

其中：

+ input：一个tensor数据类型的待处理对象，需满足shape的要求

+ weight：卷积核

+ stride：可以理解为卷积核移动的步长，一个数字n则默认水平和竖直方向都为n，也可以是一个包含两个数据的元组

  ![image-20220915210255948](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915210255948.png)

+ padding：在原输入的tensor周围添加的元素（默认为0）的多少

  ![image-20220915210418368](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915210418368.png)



#### nn.functional

包含各个卷积操作的函数，与nn中各卷积操作的类的可以视作一一对应的，一般按如下方式引入：

~~~python
import torch.nn.functional as F

output = F.conv2d(input, kernal, stride=1, ...)
~~~



#### reshape函数

`input = torch.reshape(input, target_shape_tuple)`

注意target_shape_tuple是一个tuple，其中若有某一项无法确定的话，可以写为-1，函数会自动根据其它项进行计算



#### flatten函数

`torch.flatten(input, start_dim=0, end_dim=-1) `，将一个高维度的tensor摊平到低维度，其中：

![image-20220916144444965](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220916144444965.png)

eg：

~~~python
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
torch.flatten(t)
#output: tensor([1, 2, 3, 4, 5, 6, 7, 8])

torch.flatten(t, start_dim=1)
#output: tensor([[1, 2, 3, 4],
#		        [5, 6, 7, 8]])
~~~



#### 卷积层convolution layers

主要使用`nn.Conv2d`类作为例子

![image-20220915232503407](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915232503407.png)

其中，dilation指的是，卷积核映射时的元素间距离

shape的计算公式：

![image-20220915233125062](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915233125062.png)

主要用法：实例化一个对象，作为神经网络的其中一层





#### 池化层Pooling layers

以最大池化`MaxPool2d`类为例，其通常用于保留图片特征的同时压缩其大小（即将一个kernel_size大小的部分映射为一个点）

![image-20220915235412054](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915235412054.png)

+ stride：默认为kernel_size大小
+ ceil_mode：指的是计算shape时，以7->3为例，若为floor模式，则shape为2；若为ceil模式，则shape为3，即向上or向下取整的区别

![image-20220915235759987](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220915235759987.png)



#### Padding Layers

其实就是各种padding的方式，如用0填充等



#### 非线性激活Non-linear Activation

为神经网络引入一些非线性的特质，下面以`nn.ReLU`为例：

`ReLU(x) = (x)+ = max(0,x)`

![image-20220916001733953](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220916001733953.png)

#### 正则化层Normalization Layers

有论文表明，正则化可以加快神经网络的训练速度，`BatchNorm2d`类的使用也较为简单



#### 线性层Linear Layers

![image-20220916143617638](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220916143617638.png)

其中k和bias是根据我们输入的参数在某个范围内进行采样得到的，下面以`nn.Linear`类为例：

`CLASS torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`

其中：

![image-20220916151330402](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220916151330402.png)

线性层可以将输入的最后一个维度的n个数据通过线性变换映射到另一个维度的m个数据（最后一个维度的shape改变），其他维度的shape都保持不变



#### nn.Sequential

也是`nn.container`中的一类，其作用类似与前面所讲的`Compose`类，`Compose`类是对一系列transforms实例按序整合到一起，而`Sequential`是对一系列神经网络层实例按序整合到一起，使用较为简单，实例如下：
~~~python
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        return self.model(input)

~~~



#### 损失函数

pytorch提供了许多不同的损失函数，用于计算实际模型所得和目标之间的差距，目标就是损失函数值越低越好，使用方式较为简单，只需要设置好input和target就可以自动计算了，如下：

~~~python
import torch
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))
#L1Loss对于输入输出的shape有要求

loss = L1Loss()
result = loss(inputs, targets)

print(result)
#output: tensor(0.6667)
~~~

有许多其他的损失函数，区别只是在于计算公式不同，用法相似，在此不加赘述



#### 反向传播

利用损失函数求得的结果，若不进行反向传播，是不会对原模型中的参数产生影响的，若要利用损失函数进行参数调整，则需利用反向传播求得其梯度，并在后续利用合适的优化器进行优化

在此利用我们之前搭建的神经网络为例：

~~~python
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False,
                                            transform=dataset_transform, download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, drop_last=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        return self.model(input)

model = Model()
loss = nn.CrossEntropyLoss()

for data in test_loader:
    imgs, targets = data
    output = model(imgs)
    loss_result = loss(output, targets)
    loss_result.backward()
~~~



#### 优化器torch.optim

官方文档见[这里](https://pytorch.org/docs/stable/optim.html)

使用方式：选用合适的优化器 → 构造优化器，设置好学习速率等参数 → 调用优化器的`step()`方法 → 新循环开始，调用`zero_grad()`方法，清除梯度，如下：

~~~python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
~~~

这只是对一份数据进行一次调优，真实训练中，以上部分也需要放在一个更大的循环中，多次训练进行参数调优

其中，构造优化器的一般方式如下：

~~~python
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #调用模型的parameters()方法
~~~

对优化器的learning rate参数也可以进行调整，通过Scheduler进行调整，具体用法见[这里](https://pytorch.org/docs/stable/optim.html)



### 使用现有网络模型

pytorch提供了许多已有的模型，见[这里](https://pytorch.org/vision/stable/models.html)

使用方式举例（以vgg16为例）：

~~~python
import torchvision
from torch import nn

vgg_false = torchvision.models.vgg16(pretrained=False, progress=True)
# vgg_true = torchvision.models.vgg16(pretrained=True, progress=True)
# pretrained 设置为True的时候，会从网上下载该模型对应的已训练好的参数集

print(vgg_false)

vgg_false.add_module('add_linear', nn.Linear(1000, 10))  # 在网络中添加一层
vgg_false.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 在某一层中添加一层

vgg_false.classifier[6] = nn.Linear(4096, 10)   # 修改某一层中的某一项的内容
~~~



### 模型的保存和加载

主要有两种方法：

~~~python
# model_save.py

import torch
import torchvision

vgg_false = torchvision.models.vgg16(pretrained=False, progress=True)
# 保存方式1，保存了模型的结构+模型的参数
torch.save(vgg_false, 'vgg_method1.pth')	# 保存对象+保存路径

# 保存方式2，仅保存模型的参数
torch.save(vgg_false.state_dict(), 'vgg_method2.pth')	# 保存对象字典化+保存路径
~~~

~~~python
# model_load.py

import torch
import torchvision

# 保存方式1对应的加载方式
model = torch.load('vgg_method1.pth')

# 保存方式2对应的加载方式
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg_method2.pth'))
~~~

官方推荐的是使用方式2，原因如下：

+ 方式2保存占用的空间更小
+ 使用方式1的话，必须将对应网络模型的定义import进来



### 完整的模型训练过程

分为以下几个步骤：

+ 准备数据集，创建DataLoader

+ 创建网络模型

+ 指定device

+ 创建损失函数和优化器

+ 设置训练的参数

+ 训练：

  + 训练步骤

    + 训练

    + 调优

  + 测试步骤

    + 测试
    + 计算测试结果



示例代码如下：

~~~python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./datasets", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./datasets", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 开始时间
start_time = time.time()

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("total time :{}".format(end_time - start_time))
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(model, "model_{}.pth".format(i))
    print("模型已保存")

writer.close()
~~~

要点：

+ 46行指定device：

  一般采用如上的写法，可以适用于不同（有无GPU）的环境

  注意指定device后，**网络模型实例、损失函数实例以及数据和对应targets**，都需要调用to方法，即`to(device)`将其迁移到GPU中去。其中，数据和对应targets需要将其返回值重新赋值回自身

+ 76行和97行：

  训练步骤开始前，调用`model.train()`；测试步骤开始前，调用`model.eval()`，可视作开始训练模式和测试模式

+ 93行：

  loss调用`item()`方法，其实就是将tensor类型转化为数值类型进行输出

+ 100行：

  `with torch.no_grad():`测试代码放于该块内，可以防止测试过程中对参数进行误调整

+ 108行：

  该行用于计算本次总的命中数，其中`argmax()`方法用于计算当前数据的最大值所对应的下标，数据可以是一个多维的tensor，参数为1表示横向按行比较，参数0为表示纵向按列比较



### 使用GPU训练

代码已包含在上一部分中，若本机无GPU，可使用Google Colab的免费GPU（需要Google账号），详见[这里](https://colab.research.google.com/)

要使用GPU，需进入笔记本，选择“修改”→“笔记本设置”→“硬件加速器”→“GPU”



### 模型使用

其实过程就类似于前面**【完整的模型训练过程】**中代码的test步骤部分，读入一个待检测的图片，加载已训练好的模型，作用于该图片，得到其output，使用`argmax()`方法得到其最大值的下标，即为模型给出的预测结果

一个示例代码如下：

~~~python
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "imgs/dog.png"
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("model_39.pth", map_location=torch.device('cpu'))
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

print(output.argmax(1))
~~~

注意要点：

+ 第8行：`image = image.convert('RGB')`

  由于png格式的图片是4通道的，除了RGB之外还有一个透明度通道，因此需要通过`convert()`方法将其转换为三通道，才符合该模型的输入格式

+ 第34行：`model = torch.load("model_39.pth", map_location=torch.device('cpu'))`

  使用GPU训练的模型，在加载时需要将其映射回CPU模型





