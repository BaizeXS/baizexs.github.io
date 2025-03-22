---
title: "My 1st post"
date: 2020-09-15T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["first"]
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Desc Text."
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

### Text to Sequence

文本处理

Step1: Tokenization

- Tokenization breaks a piece of text down into a list of tokens.

- Here, a token is a word. (A token can be a character in some applications.)

Tokenization讲究很多，例如是否应该把大写改成小写呢？进行typo correction等等

![image-20250322150322238](./NLP-2.assets/image-20250322150322238.png)

Step2: Build Dictionary

- Use a dictionary (hash table) to count word frequencies.

- The dictionary maps word to index.

可以首先统计词频去掉低频词，然后让每个词对应一个正整数；这样的话，一句话就可以用一个正整数列表表示，这个列表就叫sequence；如果必要就还得做one-hot encoding。![image-20250322150620890](./NLP-2.assets/image-20250322150620890.png)

结果

![image-20250322150754528](./NLP-2.assets/image-20250322150754528.png)

但是长度不统一

Step4: 对齐sequence

![image-20250322150854986](./NLP-2.assets/image-20250322150854986.png)

解决方案是这样的：可以固定长度为w，加入一个序列长度太长，就砍掉前面的词，只保留最后w个词；如果一个序列太短，就做zero padding，用0填充

![image-20250322151030119](./NLP-2.assets/image-20250322151030119.png)

这样一来，所有的序列长度就统一了

![image-20250322151117125](./NLP-2.assets/image-20250322151117125.png)









文本处理完成了，接下来是word embedding，把单词表示为一个低维向量；

![image-20250322151252837](./NLP-2.assets/image-20250322151252837.png)

然而这样做的话，如果是10000个单词，维度就是10000维，维度太高了；因此要做word embedding；

![image-20250322151511757](./NLP-2.assets/image-20250322151511757.png)

具体做法就是把one-hot向量e_i和参数矩阵p相乘；矩阵P转置的大小是d x v；d是词向量的维度；v是字典里词汇的数量；结果为x_i；x_i就是一个词向量；如果one-hot向量的第三行为1；那么x_i就是P转置矩阵的第三列（其余列都为0）；所以P转置每一列都是一个词向量；所以矩阵P本身的每一行就是一个词向量；其行数为v；即vocabulary，词汇量；每一行对应一个单词；矩阵的列数为d；d是用户决定的；d的大小会决定机器学习模型的表现；应该用xxx来选择一个比较好的d；

我们的任务是看评论是正面的还是负面的；参数矩阵是从训练数据中学习出来的，所以学出来的词向量是带有感情色彩的；假设这些词向量都是二维的，则可以在平面坐标系中标出下面词向量

![image-20250322152340636](./NLP-2.assets/image-20250322152340636.png)

keras提供embedding层；用户可以指定vocabulary大小和d的大小；以及每一个sequence的长度；d是根据算法选出来的
![image-20250322152454854](./NLP-2.assets/image-20250322152454854.png)

到这一步，我们已经完成了文本处理和word embedding。接下来就是用Logistic Regression for Binary Classification做二分类；

![image-20250322152657058](./NLP-2.assets/image-20250322152657058.png)

![image-20250322153044970](./NLP-2.assets/image-20250322153044970.png)