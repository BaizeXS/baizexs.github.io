---
title: "RNN and NLP Study Notes 02"
date: "2025-03-24T11:41:20+08:00"
tags: [""]
author: "Tristan"
draft: false
description: ""
# canonicalURL: "https://canonical.url/to/page"
---

## Text Processing and Word Embedding

### Text to Sequence

Processing text data is crucial for natural language processing (NLP) and machine learning applications. This section outlines the key steps in processing text data.

#### Step 1: Tokenization

The first step in text processing is **tokenization**, which involves **breaking down a text string into a list of individual words**. For example, given the text:

```
S = "Machine learning is an important branch of artificial intelligence"  
```

Breaking this string into a word list:

```
L = ["machine", "learning", "is", "an", "important", "branch", "of", "artificial", "intelligence"]
```

This shows the simplest form of tokenization, but in reality, many factors need to be considered, such as:

- Converting upper case to lower case (e.g., changing "Apple" to "apple").
- Removing stop words, such as "the," "a," "of," etc.
- Correcting typos (e.g., changing "goood" to "good").

Nowadays, commonly used tokenization methods include **BPE (Byte Pair Encoding)**, **WordPiece**, and **SentencePiece**.

#### Step 2: Build Dictionary

After obtaining the word list, the next step is to **build a dictionary that maps each word to a unique index** and **counts the frequency of each word**. This can be accomplished **using a dictionary (hash table)** that records each word along with its corresponding frequency.

- Initially, the dictionary is empty.
- For each word ($w$):
  - If $w$ is not in the dictionary, add $w$ and set its frequency to 1.
  - If $w$ is already in the dictionary, increment its frequency counter.

After processing some text, the dictionary might look like this:

| Word                    | Frequency |
| ----------------------- | --------- |
| machine learning        | 219       |
| artificial intelligence | 200       |
| deep learning           | 180       |
| model                   | 131       |
| algorithm               | 120       |
| training                | 52        |
| prediction              | 31        |

#### Step 3: One-Hot Encoding

**Using a dictionary to map words to indices**

Utilizing the previously constructed dictionary, map each word in the text to its corresponding index (integer). Then, these indices form a sequence. For example, suppose the dictionary contains the following entries derived from the frequency table:

| Word                    | Index |
| ----------------------- | ----- |
| machine learning        | 1     |
| artificial intelligence | 2     |
| deep learning           | 3     |
| model                   | 4     |
| algorithm               | 5     |
| training                | 6     |
| prediction              | 7     |
| an                      | 8     |
| important               | 9     |
| branch                  | 10    |
| of                      | 11    |
| intelligence            | 12    |
| is                      | 13    |

The corresponding sequence for the sentence would be:

```
sequences = [1, 13, 8, 9, 10, 11, 2]
```

**Generating one-hot vectors**

For each word in the sequence, a one-hot vector is generated based on its index. A one-hot vector is a $v$-dimensional vector, where $v$ is the size of the vocabulary.

Given the vocabulary size of 13 (assuming distinct words from the example), each word's one-hot vector will be structured as follows:

```(空)
"machine learning" = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"artificial intelligence" = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"is" = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
"an" = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
"important" = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
"branch" = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
"of" = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

For the example sentence "Machine learning is an important branch of artificial intelligence", the corresponding one-hot encoding results would be:

```
[  
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // "machine learning"  
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  // "is"  
  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  // "an"  
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  // "important"  
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  // "branch"  
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  // "of"  
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // "artificial intelligence"
]  
```

In this way, all words in the example sentence are transformed into corresponding one-hot encodings, which can be used for subsequent machine learning tasks.

#### Step 4: Align Sequences

In the process of text processing, Aligning Sequences is a crucial step to ensure that **all input samples have the same length**. After the previous processing steps, we found that there are differences in the lengths of the sequences, which poses a challenge for building machine learning models. Most machine learning models require uniform input shapes, so we must perform sequence alignment.

To address this issue, the following measures can be implemented:

- **Fixed Length**: Choose a fixed length $w$.
- **Truncation**: If a sequence exceeds the length $w$, retain only the last $w$ words.
- **Padding**: If a sequence is shorter than $w$, pad it with zeros until it reaches the specified length $w$.

![image-20250325141534011](/images/image-20250325141534011.png)

By doing this, all sequences will be adjusted to the same length, allowing them to be effectively stored in a matrix and facilitating subsequent training of machine learning models. This process will help improve the model's performance and accuracy.

![image-20250325141950386](/images/image-20250325141950386.png)

### Word Embedding: Word to Vector

#### Why map words to vectors?

The main reasons for mapping words to vectors in natural language processing (NLP) include:

1. **Capturing semantic relationships**

   Traditional word representations (like one-hot vectors) only indicate the presence or absence of words, lacking the ability to express similarities between words. By mapping to a low-dimensional vector space, word embeddings can capture semantic similarities between words. For example, "king" and "queen" would be closer in vector space, while "king" and "apple" would be farther apart.

2. **Dimensionality reduction**

   One-hot encoding generates a sparse vector with a dimension equal to the size of the vocabulary, leading to significant storage and computational costs. Word embeddings represent words as low-dimensional dense vectors, significantly reducing the required storage space and computation.

3. **Improving model performance**

   Using low-dimensional embedding vectors allows machine learning models to learn and predict more efficiently. Embedding vectors enable models to better understand and process complex patterns in language, thus enhancing performance in tasks like sentiment analysis and text classification.

4. **Facilitating transfer learning**

   Pre-trained word embedding vectors can be transferred between different tasks and datasets, accelerating model training and improving prediction accuracy. This makes word embeddings widely used in various NLP tasks.

#### How to map word to vector?

1. **Represent words using one-hot vectors**

   First, represent words using one-hot vectors.

   - Assume the dictionary contains $v$ unique words (vocabulary = $v$).
   - Then the one-hot vectors $e_1$, $e_2$, $e_3$, …, $e_v$ are $v$-dimensional.

2. **Map one-hot vectors to low-dimensional vectors**

   Next, map the one-hot vectors to low-dimensional vectors. The mapping formula is:
   $$
   x_i = P^T \cdot e_i
   $$

   - $x_i$ is the low-dimensional vector, with dimensions $d \times 1$.
   - $P$ is parameter matrix which can be learned from training data, with dimensions $d \times v$.
   - $e_i$ is the one-hot vector of the 𝑖-th word in dictionary, with dimensions $v \times 1$

   {{< figure src="/images/image-20250325143845492.png" alt="image-20250325143845492" >}}

#### Interpretation of the parameter matrix

- The parameter matrix $P$ contains the embedding representations of each word in the low-dimensional space. **Each row corresponds to a word**.

- By visualizing, you can see the relative positions of different words in low-dimensional space. **Similar words are close to each other in vector space**, reflecting their semantic similarity.

  For example, the position of the word "fantastic" is close to "good," "fun," etc., while "boring" and "poor" are relatively far apart.

  {{< figure src="/images/image-20250325143922568.png" alt="image-20250325143922568" >}}

## Conclusion

Text processing converts raw text into structured formats suitable for analysis and consists of several key steps:

1. **Tokenization**: Breaking down text into individual words.
2. **Building a Dictionary**: Mapping each word to a unique index and counting frequencies.
3. **One-Hot Encoding**: Converting words to one-hot vectors.
4. **Aligning Sequences**: Ensuring uniform input lengths for machine learning models.

Mastering these steps is essential for efficient feature extraction and improved model performance in natural language processing tasks.

Word embeddings transform words into low-dimensional vectors, capturing semantic relationships and enhancing expressiveness for various natural language processing tasks. The process involves:

1. **One-Hot Encoding**: Representing words as sparse vectors.
2. **Mapping to Low-Dimensional Vectors**: Using a parameter matrix to project one-hot vectors into a reduced space.

Understanding how to map words to vectors and interpret the parameter matrix is crucial for developing effective NLP applications.

## References

For a deeper understanding of these concepts, you can refer to the following course video:[RNN模型与NLP应用(2/9)：文本处理与词嵌入](https://youtu.be/6_2_2CPB97s?si=dUyiiUyIeKITKTxN).
