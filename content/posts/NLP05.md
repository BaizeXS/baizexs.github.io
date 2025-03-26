---
title: "RNN and NLP Study Notes 05"
date: "2025-03-24T11:41:39+08:00"
tags: [""]
author: "Tristan"
draft: true
description: ""
# canonicalURL: "https://canonical.url/to/page"
---

## Making RNNs More Effective

包含三个方面：Stacked RNN、Bidirectional RNN、Pretrain

### Stacked RNN

可以将很多全连接层堆叠起来，构成一个MLP；可以把很多卷积层堆叠起来，构成一个深度卷积网络；同样的道理，你可以将很多RNN堆叠起来，构成一个多层RNN网络；神经网络每一步都会更新状态h，新算出来的h有两个copies；一个送到下一个时刻，另一个作为输出，这一层的输出h成为了下一层的输入；

最底层的RNN的输入为词向量x；这些词向量的输出h会作为下一层RNN的输入；

![image-20250325234633209](../../../../Library/Application Support/typora-user-images/image-20250325234633209.png)

最上层的状态为RNN的输出；最上层的最后一个状态h_t为最终的状态h_t，

### Bidirectional RNN

Standard RNN和人的阅读习惯一样，从左往右阅读；人阅读的过程在脑子里积累阅读的信息；RNN在状态向量h_t中积累阅读到的信息；读完一份电影评论则可以知道是正面评价还是负面评价；我们人类总是从前往后，从左往右进行阅读；但是这只是人类的阅读习惯，从后往前阅读依旧可以判断电影评价是正面的还是负面的；对RNN来说，从前往后阅读或者从后往前阅读，是没有太大区别的，训练一个从后往前阅读的RNN，也会有一个比较不错的结果；所以比较自然的想法就是训练两条RNN，一条从左往右读，一条从右往左读；两条RNN是独立的，不共享参数，也不共享状态；两条RNN各自输出自己的状态向量，然后把它们的状态向量做cat；记做向量y；如果有多层RNN，就将这些y作为更上一层RNN的输入；如果只有一层RNN，则可以将y丢弃，只保留两条链最后的状态h_t和h_t‘；把这两个向量的cat作为从输入文本中提取出的特征向量；把这个向量用于判断是正面还是负面；

![image-20250326000815290](../../../../Library/Application Support/typora-user-images/image-20250326000815290.png)

双向RNN总是比单向RNN效果好；原因可能是这样的，无论哪个方向，或多或少会忘记最开始的输入；从左往右阅读可能忘记最左边的内容；反之亦然；

### Pretrain

预训练在深度学习中非常好用，比如在卷积神经网络中，如果网络太大而训练集不够大，那么可以在ImageNet大数据上做预训练；这样可以让神经网络有比较好的初始化，也可以避免Overfitting；训练RNN同理；例如这里Embedding层有320000参数，而我们只有2w个训练样本；这个Embedding太大了，会导致overfitting；解决办法就是对Embedding层做预训练；

![image-20250326001446063](../../../../Library/Application Support/typora-user-images/image-20250326001446063.png)

预训练具体如下；首先找一个更大的数据集，可以是情感分析的数据，也可以是其他类型的数据；但是任务最好是接近原来情感分析的任务；最好学习到的词向量带有正面或者负面情感；两个任务越相似，预训练之后的transfer就会越好；有了大数据集之后，要搭建一个神经网络，这个神经网络的结构是什么样都可以，甚至不用是RNN，只要这个神经网络有embedding层即可；然后就是在大数据集上训练这个神经网络；

![image-20250326001629650](../../../../Library/Application Support/typora-user-images/image-20250326001629650.png)

训练好之后，丢掉上面的层，只保留Embedding层和训练好的模型参数；

![image-20250326002121770](../../../../Library/Application Support/typora-user-images/image-20250326002121770.png)

然后再搭我们自己的RNN网络；这个新的RNN网络和之前预训练的可以有不同的结构，搭好之后，新的RNN层和全连接层都是随机初始化的，而下面的Embedding层的参数是预训练出来的，要固定住Embedding层的参数，不训练Embedding，只训练其他层；

### Summary

- SimpleRNN and LSTM are two kinds of RNNs; always use LSTM instead of SimpleRNN.
- Use Bi-RNN instead of RNN whenever possible.
- Stacked RNN may be better than a single RNN layer (if $n$ is big).
- Pretrain the embedding layer (if $n$ is small).

还有一些其他的RNN替代，例如GRU，Gated R Unit；但是不一定比LSTM好。
