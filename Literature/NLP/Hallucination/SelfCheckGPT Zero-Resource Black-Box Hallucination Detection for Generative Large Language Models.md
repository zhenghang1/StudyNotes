## SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models

### Introduction

提出了一个**针对黑盒模型**生成的内容的幻觉情况进行评判的流程/方法，基本思想是针对同一个query进行**多次重复采样**，根据这些样本来对Response的幻觉情况进行评价

主要针对的是句子级别和篇章级别的幻觉情况

### Method

句子级别：

设计了几种对幻觉情况进行评分的方式：

- BERTScore：计算句子间相似度的方法，对于Response中的每个句子，与每个sample中的每个句子计算BERTScore，按如下方式进行计算得到其得分，可以看到若样本中越多地出现相关的句子，则BERTScore的分数会越低，即认为该句子更可能是事实性的，否则认为其带有幻觉的可能性较大。其中BERTScore的计算式如下：

  $\bold S_{BERT}(i) = 1-\frac{1}{N}\sum\limits_{n=1}^{N}\mathop{max}\limits_{k}(\bold B(r_i,s_k^n)) $

- MQAG：多项选择，使用多项选择题生成模型和问答模型来进行评判，Response和Samples的答案选项重合率越高即代表其越具有事实性，本质上也就是判断Response和Samples中句子的相似程度

- n-gram：这里主要采用1-gram，统计Response和Samples中的n-gram数，计算Response中每个n-gram的负对数概率的平均值或最大值，其实也就是判断句子中出现的词汇的重合率，应该是判断句子相似程度的一个最简单直观的方法

篇章级别：

基本思想就是直接对句子级别的得分进行平均

### Experimental result

Sentence-level Hallucination Detection

- LLM本身的token概率就可以作为幻觉检测的基线，通过实验验证了结论：“当模型对事实不够确定时，生成的token概率具有更大的不确定性”
- 使用代理LLM方法的性能非常差，代理LLM即使用一个可访问的灰盒LLM的token概率来近似一个待研究的黑盒LLM的token概率
- SelfcheckGPT的方法与灰盒方法持平或者更优，使用最简单的n-gram方法通常就可以取得一个很好的结果

Passage-level Factuality Ranking

SelfcheckGPT方法得到的结果与人类评估结果具有很强的相关性，验证了其有效性