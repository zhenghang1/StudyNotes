# Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback

## **Introduction**

![](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20230721172719101.png)

通过基于规则的方法或者打分模型，针对每个回复决定要进行的action，直接回复or调用外部知识库采集信息并重新生成回复，期间在memory中记录得到的信息以及历史query等

各个模块：

- Working Memory：记录历史状态，如query，收集到的信息，每次response的打分情况等
- Policy：决定下一步的策略，如直接回复或采集信息重新生成回复，可以基于规则或基于可训练的模型来实现
- Action Executor：分为多个子模块
  - Knowledge Consolidator：知识整合器，根据query和历史query信息，调用外部api进行信息查询，并将收集到的信息进行整合
  - Prompt Engine：根据以上各种信息，重新生成一个prompt
- Utility：对生成的response进行效用性等的打分，并生成反馈用于提示LLM更好地response（反馈可以使用另一个生成模型来实现），效用函数应当是**task-specific**的

## **Specific tasks**

分为两个任务

(1) information seeking dialog：需要基于各种外部知识来源生成回答

(2) Wiki question answering：需要拼凑分散在多个wiki文档中的各种模式的信息来回答问题

### **information seeking dialog**

侧重于信息真实性，同时要尽可能提高信息量，避免闲谈

数据集：

- News Chat，基于DSTC7 Track 2任务构建
- Customer Service，基于DSTC11 Track 5任务构建

实验设置：

- LLM：使用ChatGPT

- Knowledge Consolidator，有两种设定：

  - 使用BM25检索器（BM25 retriever）从对应知识库进行检索
  - 使用Groundtruth知识库，也称golden knowledge

- Prompt Engineer：提供了对应的prompt

- Utility：设计了两种不同的效用函数和反馈方法

  - 效用函数：KF1（Knowledge F1），衡量prediction和evidence之间的重合（evidence即Knowledge Consolidator提供的）

    反馈生成：基于模板的自然语言生成器

  - 效用函数：ChatGPT进行自我评价（self-criticism）

    反馈生成：ChatGPT生成

- Policy：使用一个可训练的Policy模型（基于T5-Base，Google提出的seq2seq预训练语言模型，**[论文链接](https://arxiv.org/abs/1910.10683)**），其使用off-line的RL训练方式从头开始训练

- Evaluation：一系列评价指标，包括KF1，BLEU，ROUGE，chrF，METEOR，BERTScore，BARTScore和BLEURT

实验结果：

- 使用外部知识（BM25检索的方式或golden knowledge的方式）可以带来幻觉的显著减轻
- 使用反馈也能有效提高性能
- 使用可训练的policy可以有效选择系统动作，对性能提升有所帮助

消融实验：

- 三种关于知识整合器（KC）的使用设定：①不使用外部知识；②仅在LLM建议时使用外部知识；③一直使用外部知识

  实验结果表明，使用KC的KF1 score显著高于不使用的情况，但总是使用KC会带来较高的额外开销，因此最好采用可训练的Policy来决定何时采用KC

- 关于Utility的反馈设定：①不使用反馈；②使用基于规则的反馈；③使用基于ChatGPT的self-criticism反馈

  实验结果表明，使用了反馈要优于不使用，而使用ChatGPT的self-criticism的反馈会包含更详细的建议

### **Wiki question answering**

侧重于多跳推理，从多个信息来源中获取信息并拼凑得到最终结果

由于LLM通常是在单个网页的文本上进行训练的，其对于分布在多个页面的复杂多步推理容易产生幻觉

数据集：OTT-QA，一个开放域问答的benchmark，包含了多步联合推理（13%的单跳问题，57%的两跳问题和30%的多跳问题）

实验设置：

- Knowledge Consolidator：使用了两种不同的KC
  - DPR：开放域问答检索技术，采用包含语义信息的稠密向量表示段落和question，只需对其嵌入向量计算内积即可
  - 额外使用CORE中的中间模块如linker和chainer来拼接证据链
- Utility：使用召回率作为效用函数，使用模板的自然语言生成模型生成反馈
- Evaluation：使用Precision，Recall和F1 score

实验结果：

LLM-AUGMENTOR可以有效提高F1 score等指标，但是与之前的使用top-50  consolidated evidence的微调模型相比，依然存在明显差距