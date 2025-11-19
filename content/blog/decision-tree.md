---
title: "Decision Tree"
date: 2025-09-08
---

This post comes from my rough notes, covering the core concepts of decision tree, from how to build a tree to analyze the problem of overfitting.

## Acknowledgments

The structure and most of the content in this post follow the lecture slides of the [10-601: Introduction to Machine Learning](https://www.cs.cmu.edu/~mgormley/courses/10601/) course at Carnegie Mellon University, taught by Professors Geoff Gordon and Matt Gormley. Google Gemini also helps me in drafting the proof and formulating parts of this blog.

---

## Splitting Criterion

When building a decision tree, we need to select a splitting criterion to decide each split point. The idea is that we choose the best feature at each node by our splitting criterion to build the tree. Here are three common criteria:

### 1. Error Rate
A simple metric, just choose the feature that minimize training error.

### 2. Gini Gain
The **Gini Gain** evaluates a split by measuring the reduction in impurity.

1.  The **Gini Index** for a given node $t$ is calculated.
    $$
    Gini(t) = 1 - \sum_{k} p_k^2
    $$

2.  **Gini Gain** ($\triangle Gini$) is the impurity of the parent node minus the weighted average of the impurities of the child nodes.
    $$
    \triangle Gini = Gini(\text{parent}) - \sum_{i \in \text{children}} \frac{|t_i|}{|t|} Gini(t_i)
    $$

### 3. Information Gain

**Information Gain** evaluates a split by measuring the reduction in uncertainty, or **entropy**.

1.  **Entropy** measures the amount of uncertainty in the data.
    $$
    H(t) = -\sum_{k} p_k \log p_k
    $$

2.  The **Information Gain** ($\triangle H$) is the entropy of the parent node minus the weighted average of the entropies of the child nodes.
    $$
    \triangle H = H(\text{parent}) - \sum_{i \in \text{children}} \frac{|t_i|}{|t|} H(t_i)
    $$

#### Alternative Formulation (Mutual Information)

Information Gain is equivalent to the **Mutual Information** between the features $X$ and the target variable $Y$. It quantifies how much information about $Y$ is gained by knowing $X$.

$$I(Y;X) = H(Y) - H(Y|X)$$

In this formula, $H(Y|X)$ is the **conditional entropy**, which is the average entropy of $Y$ after the values of $X$ are known:
$$H(Y|X) = \sum_{v \in V(X)}P(X=v)H(Y|X=v)$$

---

## Key Concepts

* **Generalization**: How well does our Decision Tree perform on data it has never seen?
* **Inductive Bias**: The set of assumptions a learning algorithm uses to make predictions.
* The **ID3 Algorithm** has an inductive bias that favors trees placing features with high information gain closer to the root.

---

## The Problem of Overfitting

A challenge with decision trees is **overfitting**. This occurs when a model learns the training data too well and have poor performance on unseen data.

* **An Example (Selection Bias)**:
    If we roll a die, the expected value is 3.5. If we roll three dice, the expected max of those three dice is larger than 3.5!
    The same apply to DT. At each step we greedily choose the *best* splitting point. The performance we see on our training data is like getting the maximum value from multiple dice rolls—it's optimistically biased upwards compared to the true performance on real data.

* **Solving Overfitting with Pruning**:
    * **Pruning** is the process of simplifying a tree by removing branches to improve its generalization. If removing a split doesn't significantly harm performance on a validation set, then we remove the splitting point to make the tree structure simplier.

---

## A Non-Rigorous View of DT Overfitting

A different angle for the selection bias mentioned above

### 1. Definitions and Problem Formulation

* **Random Training Set (S)**: A random subset of data from a real data distribution.
* **Fixed Set of Models**: A group of $n$ models to choose from, $M_1, \ldots, M_n$.
* **Sample Accuracy $X_i(S)$**: The accuracy of model $M_i$ on a *specific* training set $S$.
* **True Accuracy $\mu_i$**: The expected accuracy of model $M_i$ over *all possible* training sets, i.e., $\mu_i = E_S[X_i(S)]$.
* **The Training Process**: Selecting the model with the maximum accuracy on our given set $S$.

**Goal**: To prove that the performance we *see* is optimistically biased compared to the performance we *actually get*.

$$
E_S[\max(X_1(S), \ldots, X_n(S))] \ge \max(\mu_1, \ldots, \mu_n)
$$

### 2. The Proof

1.  For any single training set $s$, the best model's performance is, by definition, greater than or equal to any other model's performance:
    $$
    \max(X_1(s), \ldots, X_n(s)) \ge X_i(s)
    $$

2.  **Applying Monotonicity of Expectation**: Since this holds for any $s$, it holds over the expectation of all $S$. Because $E[\cdot]$ is monotonic, we get:
    $$
    E_S[\max(X_1(S), \ldots, X_n(S))] \ge E_S[X_i(S)]
    $$

3.  This relationship holds for *any* model $i$. Therefore, the left side must be greater than or equal to the **maximum** of all possible terms on the right, which proves our goal.

### 3. Interpreting the Formula

The expected performance of the champion model we *see* during training (the left side) will always be systematically **higher** than the true performance of the best model we actually have (the right side). This difference is the "optimism bias," which is overfitting.

---

## Summary: Decision Tree Overview

Despite the risk of overfitting, decision trees remain a valuable tool.

### Pros:
* **Easy to understand and interpret**. The tree structure is transparent.
* **Computationally efficient** for both training and inference.