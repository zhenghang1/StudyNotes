## Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation

### Introduction

主要聚焦于大模型的自我矛盾幻觉，提出了一个框架对其进行评估、检测和缓解，适用于**黑盒大模型**

自我矛盾是幻觉中的一种。解决幻觉的一种常见方式是利用外部知识进行事实检测，这依赖于高质量的外部知识库；而自我矛盾幻觉的检测则不需要外部知识，仅仅需要逻辑推理即可，较为容易实现。



### Method

本方法中设计两个LLM模型，一个作为生成模型GLM，一个作为分析模型ALM

+ Trigger

  首先使用GLM生成一段文本x，然后将x中的每个句子，抽取上下文列表c，在c的约束下重新生成一个新的句子，最终与x构成一个句子对

+ Detection

  直接使用ALM进行分析

+ Mitigation

  使用ALM，对矛盾的信息进行删除，同时尽可能保留非矛盾的信息



### Dataset

使用自己从Wikipedia找的topics，不同topic间覆盖不同领域，Google search受欢迎程度不同，并且平衡了人类和非人类话题的比例



