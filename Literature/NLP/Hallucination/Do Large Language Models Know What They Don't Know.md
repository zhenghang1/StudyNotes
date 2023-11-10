## Do Large Language Models Know What They Don't Know?

提出了一种自动化的方法，来检验LLM评估自我认知能力（了解自己对未知的局限的能力），构建了一个数据集SelfAware，由五个不同类别的无法回答的问题及其可回答的对应问题组成，对包括GPT-3、InstructGPT和LLaMA在内的20个llm进行了广泛的分析。

![image-20230815140109524](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20230815140109524.png)

### Dataset

SelfAware，包含1032个无法回答的问题和2337个可以回答的问题，其中可回答问题来源于SQuAD、HotpotQA和TriviaQA 三个数据集，并选择了语义上最接近不可回答问题的部分

### Method

自动评测：提前选定了一批参考句子（16条），类似“The answer is unknown.”，然后对所有的输出，计算与参考句子的相似度，若相似度大于某个阈值则认为该输出表明问题不可回答（阈值消融实验后，设定为0.75），否则认为其可回答

### Experiment

对二十几个LLM进行评测，根据其模型大小、是否经过指令调优来进行比较

此外，对每个问题设计了三种输入形式：

+ direct：直接输入问题
+ instruction： prompt中先包含一段指令，指导其若是不确定，可以输出“The answer is unknown”
+ In-context learning： prompt中先给出六个QA的示例，然后输入待测试的问题

### Result

+ LLM的自我认知能力，随着模型大小增大而提高
+ 经过指令调优的LLM，自我认知能力显著提高
+ 输入形式中，instruction和in-context learning的输入形式都有助于提高自我认知能力，尤其是in-context learning的方式
+ 最先进的LLM，自我认知能力与人类相比也有显著的差距