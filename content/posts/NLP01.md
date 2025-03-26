---
title: "RNN and NLP Study Notes 01"
date: "2025-03-23T21:43:18+08:00"
tags: ["Machine Learning", "NLP", "ML Basics"]
author: "Tristan"
draft: false
description: ""
# canonicalURL: "https://canonical.url/to/page"
---

## Data Processing Basics

In machine learning, data types can be classified into several forms, including **numeric features**, **categorical features**, and **binary features**. The table below provides concrete examples to better understand these data types:

| Age  | Gender | Nationality |
| ---- | ------ | ----------- |
| 35   | Male   | US          |
| 31   | Male   | China       |
| 29   | Female | India       |
| 27   | Male   | US          |

### Numeric Features

Numeric features refer to data that possess **additive properties** and **can be compared in magnitude**. For example, a person's age serves as a numeric feature, where 35 is greater than 31.

### Binary Features

Binary features typically represent scenarios with **only two possible values**, such as gender. Gender can be represented by 0 (female) and 1 (male).

### Categorical Features

Categorical features are used to represent **different categories**, such as nationality. Due to the diversity of countries, numeric vectors are needed for representation. A dictionary can be created to map nationalities to corresponding indices, such as `US=1`, `China=2`, `India=3`, and so on.

### Issue with Categorical Features

Representing categorical features using integer values may lead to **incorrect numerical relationships**. For instance, the expression `US + China = India` is ridiculous.

### Solution: One-Hot Encoding

One-Hot encoding is a method used to represent categorical variables as binary vectors, effectively mitigating the issues associated with ordinal relationships. For example, nationalities can be encoded as follows:

- The U.S. is represented as `[1, 0, 0, 0, ..., 0]`.
- China is represented as `[0, 1, 0, 0, ..., 0]`.
- India is represented as `[0, 0, 1, 0, …, 0]`.

In this way, if a person holds both U.S. and Indian citizenship, their encoded representation could be a combination of those two vectors: `[1, 0, 1, 0, ..., 0]`. This indicates that they possess both nationalities.

---

## One-Hot Encoding

In machine learning, One-Hot encoding is a crucial technique used to represent categorical variables as binary vectors. This method helps to preserve the integrity of the data while avoiding misleading numerical relationships.

### Why Use One-Hot Encoding

- **Avoiding Pseudo-Numeric Relationships**: One-Hot encoding prevents **false numeric relationships** that can arise from integer encoding. By mapping categories to vectors, this method ensures that the model does not make incorrect inferences based on arbitrary numerical values.
- **Complete Information Retention**: This encoding method retains all possible category information by assigning a value of 1 at the position corresponding to each category, with 0s elsewhere. This structure simplifies the learning process for the model, allowing it to effectively distinguish between different categories.

### Steps of One-Hot Encoding

One-Hot encoding is typically performed in two main steps:

1. **Establishing Mapping: From Category to Index**

   - Assign a unique index to each category, typically starting from 1. 
   - Reserve index 0 to represent "unknown" or "missing," which corresponds to a vector of all 0s. This enhances the model's tolerance for missing data.

2. **Converting to One-Hot Vector**

   - Each category is transformed into a One-Hot vector, where a value of 1 is placed in the index corresponding to that category and 0s are placed elsewhere.

   - For example, if there are a total of 197 categories (such as countries), each category would be represented by a 197-dimensional vector:

     - The United States can be represented as `[1, 0, 0, 0, ...]`, indicating its position in the vector.

     - China can be represented as `[0, 1, 0, 0, ...]`, indicating its position as well.

These steps ensure that each category can be uniquely identified without introducing misleading numeric relationships.

### Example: Representing a Person's Features

When analyzing a person's characteristics, several key aspects come into play, including **age**, **gender**, and **nationality**. For our purposes, let's assume there are 197 nationalities.

To calculate the total number of feature dimensions, we can break it down as follows:

- **Age**: 1 dimension
- **Gender**: 1 dimension
- **Nationality**: 197 dimensions

This gives us a total of **199 dimensions** to represent a person's features.

For example, consider a person with the attributes `(28, Female, China)`. The features can be represented by the following One-Hot vector:

```
[28, 0, 0, 1, 0, ..., 0]  
```

In this vector:

- The age retains its original value of 28.
- Gender is represented by 0 (indicating female).
- The One-Hot encoding for nationality indicates a 1 at the position corresponding to China, ensuring that the categorical integrity is preserved.

This structured approach allows us to effectively model and analyze individual characteristics in a machine learning context.

---

## Processing Text Data

Processing text data effectively is essential for applications in natural language processing (NLP) and machine learning. This section outlines the key steps involved in processing text data.

### Step 1: Tokenization (Text to Words)

The first step in processing text data is **tokenization**, where a string of text is broken down into a list of individual words. For example, given the text:

```
S = “… to be or not to be…”
```

We break this string into a list of words:

```
L = [..., to, be, or, not, to, be, ...]
```

### Step 2: Count Word Frequencies

Once we have the list of words, the next step is to count the frequency of each word. This can be done using a dictionary (or hash table) that keeps track of each word and its corresponding frequency.

- Initially, the dictionary is empty.
- For each word ($w$):
  - If the $w$ is not in the dictionary, add $w$ with a frequency of 1.
  - If $w$ is in the dictionary, increment its frequency counter.

For example, after processing some text, the dictionary might look like this:

| Key(word) | Value(frequency) |
| --------- | ---------------- |
| a         | 219              |
| to        | 398              |
| hamlet    | 5                |
| be        | 131              |
| not       | 499              |
| prince    | 12               |
| kill      | 31               |

### Step 3: Limit Vocabulary Size

If the vocabulary (the number of unique words) is too large (e.g., over 10,000), it’s advisable to keep only the most frequent words. This is necessary because:

1. Infrequent words often add little meaningful information, like typos or rare names.
2. A larger vocabulary leads to higher-dimensional one-hot vectors, which increases computation time and the number of parameters in the word embedding layer.

### Step 4: One-Hot Encoding

In the final step, we map each word to its index using one-hot encoding. For instance, if the words are:

```
Words: [to, be, or, not, to, be]
```

the corresponding indices might be:

```
Indices: [2, 4, 8, 1, 2, 4]
```

Each word can then be represented as a one-hot vector. If a word cannot be found in the dictionary (such as a typo), it can either be ignored or encoded as 0.

## Conclusion

In this section, we explored essential concepts of data processing, which are foundational for effectively applying Recurrent Neural Networks (RNNs) in Natural Language Processing (NLP). We examined different data types, emphasizing numerical, binary, and categorical features. Additionally, we highlighted preprocessing techniques like One-Hot encoding and tokenization, which are crucial for transforming textual data into a format suitable for RNNs and other machine learning models.

## References

For a deeper understanding of these concepts, you can refer to the following course video:[RNN模型与NLP应用(1/9)：数据处理基础](https://youtu.be/NWcShtqr8kc?si=OobtdifZSl41e_Mn).
