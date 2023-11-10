# CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing

## Introduction

![image-20230801171222984](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20230801171222984.png)

提出了CRITIC框架，根据不同的任务，设计不同的prompt引导LLM使用特定的外部工具对response进行验证，直到其符合期望



### 实现

主要分为interaction，verification和correction三个部分

+ interaction：将任务特定的工具（如Google API，python解释器等）构建为文本到文本函数（方便LLM调用）
+ verification：对每个input query和预先生成的response，与对应的工具交互后得到一个反馈评价，LLM根据该评价验证前面生成的response的准确性
+ correction：LLM根据评价，重新生成一个新的response



## Experiment

进行了三个任务上的实验：free-form question answering，mathematical program synthesis和toxicity reduction，使用的LLM包括**text-davinci-003**和**gpt-3.5-turbo**



### free-form question answering

为了提高通用性，使用Google search的API作为工具，抓取top-1的HTML网页文件

数据集：AmbigNQ，TriviaQA和HotpotQA

对比的baseline包括：Vanilla，CoT和ReAct

实验结果

+ CRITIC有效
+ 工具的交互起着关键作用
+ 模型自我批评的帮助很小



### mathematical program synthesis

生成python程序，运行后根据其结果验证原有response的准确性，其中python程序提供其错误信息和result变量的值

数据集：GSM8k

baseline：Vanilla和PoT



### **Toxicity Reduction**

毒性检测和缓解，目标是生成流畅且无攻击性的文本，使用现有的PERSPECTIVE API作为评估细粒度毒性信息的工具，提供了总体毒性评分和六个细粒度属性（如insult、profanity和identity attack）的评分

数据集：REALTOXICITYPROMPTS，从中选择了1k个无毒性prompt用来引起毒性反应

评估：使用两个维度的评分：①25代response中的最大毒性；②25代中至少有一代response毒性超过50%的概率

结果：CRITIC显著降低了生成文本的毒性，与SOTA的有监督方法相当。同时，外部反馈对解毒非常重要，因为LLM本身难以有效减轻自己产生的response的毒性

