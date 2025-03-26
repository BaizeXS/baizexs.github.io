---
title: "RNN and NLP Study Notes 03"
date: "2025-03-24T11:41:32+08:00"
tags: [""]
author: "Tristan"
draft: false
description: ""
# canonicalURL: "https://canonical.url/to/page"
---

## Recurrent Neural Networks (RNNs)

### How to model sequential data?

In deep learning, the key to processing sequential data lies in **understanding the relationship between inputs and outputs**. Generally, the input-output relationships of data can be divided into three categories: **one-to-one**, **one-to-many**, and **many-to-many**. A one-to-one relationship means that each input has a corresponding output, which is commonly used in image classification. In contrast, a one-to-many relationship indicates that a single input can produce multiple outputs, suitable for generation tasks, such as a chatbot generating multiple responses based on user input. Meanwhile, a many-to-many relationship involves multiple inputs leading to multiple outputs, typically used in sequence transformation tasks, such as machine translation.

- **One to One**:
  - **Definition**: Each input corresponds to one output.
  - **Application Example**: Commonly used in image classification tasks, where the input is an image and the output is the label of that image.
- **One to Many**:
  - **Definition**: One input can generate multiple outputs.
  - **Application Example**: In a chatbot, multiple possible responses can be generated after inputting a question from the user, or multiple descriptions can be generated for an image.
- **Many to Many**:
  - **Definition**: Multiple inputs can produce multiple outputs, and the lengths of the sequences are usually different.
  - **Application Example**: In machine translation tasks, multiple words from the input sentence correspond to multiple words in the translated output.

Fully connected networks (FC Nets) and convolutional neural networks (ConvNets) perform well when handling fixed-size inputs and outputs, but they struggle with variable-length sequential data. Since these two models typically only allow for one-to-one or one-to-many mappings, they cannot capture the temporal dependencies and contextual information between inputs.

**Limitations of FC Nets and ConvNets**:

- Process a paragraph as a whole.
- Fixed-size input (e.g., image).
- Fixed-size output (e.g., predicted probabilities).

In comparison, recurrent neural networks (RNNs) are more suitable for modeling sequential data. RNNs can handle variable-length inputs and outputs while continuously preserving contextual information through hidden states, which enables them to demonstrate greater flexibility and capability in tasks such as sequential generation and time series analysis. RNNs are better ways to model sequential data (e.g., text, speech, and time series).

{{< figure src="/images/image-20250326104312899.png" alt="image-20250326104312899" >}}

Recurrent Neural Networks (RNNs) are designed to handle sequential data, where the relationships between inputs and outputs are often many-to-one or many-to-many rather than one-to-one. In an RNN, each word in a sequence contributes to the accumulation of information in the state vector $h_t$. Initially, each input word is transformed into a word embedding, producing a corresponding vector $x_t$. As each word vector is fed into the RNN, the network updates the state $h$ to incorporate the new information. For instance, $h_0$ encapsulates the information from the first word $x_0$, while $h_1$ contains information from the first two words $x_0$ and $x_1$, and so on. **By the time the last word is processed, the final state $h_t$ represents a feature vector that summarizes the entire sentence.** The state updates rely on a parameter matrix $A$, which remains constant throughout the RNN, regardless of the length of the sequence being processed. This design allows RNNs to effectively capture dependencies across sequences and utilize contextual information.

### Simple RNN Model

#### Recurrent Neural Networks (RNN) Models and Working Principles

Recurrent Neural Networks (RNNs) are a type of neural network model specifically designed for handling sequential data. **Their structure allows the state at each moment to depend not only on the current input but also on the previous state, creating a dynamic memory mechanism.** In an RNN, the two inputs to the model are the previous state $h_{t-1}$ and the current input word vector $x_t$.

{{< figure src="/images/image-20250326105711767.png" alt="image-20250326105711767" >}}

During the state update, the RNN first concatenates $h_{t-1}$ and $x_t$ to generate a higher-dimensional vector. This vector is then multiplied by the model's parameter matrix $A$ to obtain a new vector. This new vector is processed through an activation function, which is typically the hyperbolic tangent function $\tanh$, serving to compress each element's value within the range of -1 to 1. In this way, the state $h_t$ is effectively updated, allowing us to interpret the new state $h_t$ as a function of the current input $x_t$ and the previous state $h_{t-1}$.

The computation of state $h_t$ relies on three factors:

- the current input $x_t$,
- the previous state $h_{t-1}$,
- and the parameter matrix $A$.

The parameter matrix $A$ plays a crucial role in the calculations of the entire RNN, driving the process of each state update.

#### Choosing the Activation Function: Why Use the Tanh Function?

The activation function plays a key role in deep learning models. Without an appropriate activation function, the model may encounter issues of **gradient vanishing** or **explosion**. The function of tanh is to perform "normalization" after each weight update. By readjusting the values to a reasonable range of -1 to 1, the tanh function helps **maintain the stability of the model** and **enhances training efficiency**. This normalization process ensures that the model can learn and update the weights of each layer more accurately, thereby improving overall performance.

#### Model Parameters of Simple RNN

When discussing the parameters of RNNs, attention should be paid to the dimensions of the parameter matrix $A$. Specifically, the number of rows in matrix $A$ corresponds to the dimension of the hidden state $shape(h)$, while the number of columns is the sum of the hidden state dimension and the input dimension ($shape(h) + shape(x)$). Therefore, the total number of parameters in this matrix can be represented as:

- #rows of $A$: $shape(h)$
- #cols of $A$: $shape(h) + shape(x)$
- Total #parameter: $shape(h) \times [shape(h) + shape(x)]$.

{{< figure src="/images/image-20250326112207607.png" alt="image-20250326112207607" >}}

This means that as the input scale increases, the number of parameters will also grow, thereby influencing the model's complexity and training requirements.

#### Limitations of Simple RNN

Although Simple RNNs perform well in handling short-term dependencies, they exhibit significant **shortcomings in addressing long-term dependencies**. This is because, in RNNs, the current state $h$ is functionally related to all previous states $h$. Theoretically, if the early input $x_1$ is changed, it should lead to changes in all subsequent states $h$. However, in practice, Simple RNNs do not fully demonstrate this property. By taking the derivative of state $h_{100}$ with respect to $x_1$, it can be observed that the derivative approaches zero, indicating that when $x_1$ changes, $h_{100}$ hardly changes at all. This shows that state $h_{100}$ has almost no relation to earlier inputs $x_0$, meaning that state $h_{100}$ has forgotten information from many previous steps, highlighting the limitations of RNNs in handling long-term dependency issues.

These designs and limitations point to the use cases of Simple RNNs as well as the directions for future model improvements, leading to the development of more complex RNN variants such as Long Short-Term Memory networks (LSTMs) and Gated Recurrent Units (GRUs), aimed at addressing issues of long-term dependencies.

### Summary

- RNN for text, speech, and time series data.

- Hidden state $h_t$ aggregates information in the inputs $x_0, \dots, x_t$.
- RNNs can forget early inputs.

  - It forgets what it has seen early on.
  - If $t$ is large, $h_t$ is almost irrelevant to $x_0$.

- SimpleRNN has a parameter matrix (and perhaps an intercept vector).

- Shape of the parameter matrix is

  $$
  shape(h) \times [shape(h) + shape(x)].
  $$

- Only one such parameter matrix, no matter how long the sequence is.
