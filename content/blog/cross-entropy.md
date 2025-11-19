---
title: "Entropy, Cross-Entropy, and NLL Loss"
date: 2025-09-27
---

Cross-Entropy and Negative Log-Likelihood (NLL) loss are fundamental concepts in ML. Though often used interchangeably, they are different. This post explains the relationship and difference between them.

### 1. Information Theory Concepts 

To understand these concepts, let's consider a concrete example: encoding daily weather reports based on a known **true probability distribution (P)**.

> **Example Scenario:**
> * **P(Sunny)** = 1/2 (50%)
> * **P(Cloudy)** = 1/4 (25%)
> * **P(Rainy)** = 1/8 (12.5%)
> * **P(Snowy)** = 1/8 (12.5%)

#### Entropy: The Ideal Cost

**Entropy (H)** is the theoretical minimum average number of bits required to encode data from a specific probability distribution. We use shorter codes for high-probability events and longer codes for low-probability events.

For our data, the optimal codes are:
* **Sunny**: `0` (1 bit)
* **Cloudy**: `10` (2 bits)
* **Rainy**: `110` (3 bits)
* **Snowy**: `111` (3 bits)

The average cost, or entropy, is:
$$
H(P) = -\sum_i P(i)\log_2 P(i) = \frac{1}{2}(1) + \frac{1}{4}(2) + \frac{1}{8}(3) + \frac{1}{8}(3) = 1.75 \text{ bits}
$$

#### Cross-Entropy: The Actual Cost

If we use an inaccurate **predicted distribution (Q)** to create an encoding scheme, the resulting average number of bits needed to encode the *true* data is the **Cross-Entropy (H(P, Q))**. Consider a model (Q) that overestimates rain:

* **Q(Sunny)** = 1/8
* **Q(Rainy)** = 1/2
* **Q(Cloudy)** = 1/4
* **Q(Snowy)** = 1/8

The average cost using this model is:
$$
H(P, Q) = -\sum_i P(i)\log_2 Q(i) = \frac{1}{2}(3) + \frac{1}{4}(2) + \frac{1}{8}(1) + \frac{1}{8}(3) = 2.5 \text{ bits}
$$

#### KL Divergence: The Penalty

**KL Divergence (D_KL)** is the penalty for using an inaccurate model. It quantifies the extra bits wasted on average because the predicted distribution (Q) differs from the true distribution (P). It is the difference between the cross-entropy and the entropy.

$$
D_{KL}(P||Q) = H(P, Q) - H(P) = 2.5 - 1.75 = 0.75 \text{ bits}
$$

### 2. Connection to Machine Learning Loss 

In machine learning, the goal is to make the model's predicted distribution (Q) as close as possible to the true data distribution (P). This is equivalent to **minimizing the KL Divergence**.

Since the entropy of the true data, $H(P)$, is a fixed constant, minimizing KL Divergence is equivalent to minimizing Cross-Entropy. Therefore, Cross-Entropy is used as the loss function.

#### The Special Case: NLL Loss

The connection to **Negative Log-Likelihood (NLL) Loss** becomes clear in classification tasks, where the true distribution (P) is a **one-hot vector** (e.g., `[0, 0, 1, 0]`). In this case, the Cross-Entropy formula:
$$
H(P, Q) = -\sum_c P(c)\log Q(c)
$$
collapses to a single term for the true class $k$:
$$
H(P, Q) = - (1 \cdot \log Q(k)) = -\log Q(k)
$$
This simplified form is identical to the NLL loss, which evaluates the model's predicted probability for the single correct class. For one-hot labels, **Cross-Entropy and NLL loss are mathematically identical**.

### 3. A Look at PyTorch 

This distinction is reflected in PyTorch loss functions:

* **`nn.CrossEntropyLoss`**: This function takes raw **logits** as input and internally applies `LogSoftmax`. It accepts two label formats:
    1.  **Hard Labels**: Class index for each data - One-hot label 
    2.  **Soft Labels**: Probabilities for each class

* **`nn.NLLLoss`**: It **only supports hard labels** (class indices).