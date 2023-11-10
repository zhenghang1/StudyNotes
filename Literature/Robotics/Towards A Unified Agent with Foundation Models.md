## Towards A Unified Agent with Foundation Models

### Introduction

以语言作为核心，将LLM和VLM（Vision Language Model）应用于**强化学习agent的训练中**，**尤其是从零开始学习的强化学习系统**，解决一系列强化学习挑战如高效探索、重用经验数据、调度技能和从观察中学习

![image-20230730165953909](https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230730165953909.png)



### Terminology

+ trained from scratch：从头开始训练
+ caption：视觉语言模型训练数据集中，图像和文本对中的文本，也称为标题



### Related work

+ LLM和VLM的机器人应用，如长期规划的子目标分解，VLM来理解场景等，聚焦于部署和调度技能
+ RL中的稀疏奖励任务，方法如课程学习（curriculum learning），内在动机（intrinsic motivation），分层分解（hierarchical decomposition）等
+ 多任务学习中，通过为每个新任务学习一个新的奖励模型，来重用之前任务中的经验、trajectory



### Framework

需要使用一个LLM和一个VLM，具体流程：

+ VLM将视觉信息映射到文本
+ LLM根据文本描述和任务描述，提供语言指令
+ agent将语言指令ground到具体action



#### Bridging Vision and Language using VLMs

CLIP：一个大型对比语言模型（large contrastive visual-language model），包含一个text encoder 和一个 image encoder，可以嵌入到同一个domain

使用CLIP，将所有场景图像和可能的caption进行嵌入，计算其相似度（点积），若大于某个阈值则认为其描述是正确的

<img src="https://raw.githubusercontent.com/zhenghang1/Image/main/img/image-20230731114435911.png" alt="image-20230731114435911" style="zoom:80%;" />

#### Reasoning through Language with LLMs

使用LLM进行推理，完成子任务分解，**细节还需要再了解，论文中缺少具体描述**



#### Grounding Instructions into Actions

将LLM产生的子任务指令，通过一个language-conditioned policy network（基于Transformer），ground到一个具体的action

具体而言，这个模型是task特定的，在RL loop中从头开始训练，以sub-goal，MDP state等作为输入，输出特定action







### 应用和实验结果

（5.1）

#### 连续任务学习（5.2）

在同一个环境中，存在一系列的任务，agent在前期任务中得到的知识或经验应当可以用于后面的任务，提升后续任务的学习速度

具体方法为：

+ agent有一个lifelong buffer，存储所有经验，同时针对每个具体任务有一个task buffer
+ 针对每个新任务$T_n$，LLM将其分解为一系列子目标$[g_0,\dots,g_{L-1}]$
+ agent从其lifelong buffer中，根据每个episode的observation $o_{t,n}$，利用VLM与子目标计算相似度，具体即计算两个嵌入的点乘$\phi_T(g_l)\cdot\phi_I(o_t)$，超过一个阈值即认为其对本任务有价值，于是将该episode复制到新任务初始化的task buffer中

通过以上方法，新任务开始训练时，buffer中已经包含了一定数量的有价值的trajectory，可以帮助新任务的学习，减少训练时间

意义：可以视作解锁了agent终身学习的能力，在其生命周期中学习能力不断增强
