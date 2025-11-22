---
title: "Linear Regression"
date: 2024-01-30
---

### Simple Case

Linear regression is a simple method to solve a regression problem. It has a **closed form** solution. Let me first illustrate how to get the closed form solution.

Suppose we have a dataset $\{(x^1, y^1), \ldots, (x^n, y^n)\}$, where $x^i \in \mathbb{R}^d$ and $y^i \in \mathbb{R}$.

We want to find a linear function $f(x) = w^T x + b$ to fit the data.

To make the question simple, we first discuss the case where $d = 1$, i.e., $x^i$ is a real number. In this case, we can write the linear function as $f(x) = w_1 x + w_2$.

Therefore, what we want to find is the optimal $w_1$ and $w_2$ such that:

$$
\underset{w_1, w_2}{\operatorname{argmin}} \frac{1}{2} \sum_{i=1}^n \left( y^i - w_1 x^i - w_2 \right)^2
$$

Translate the problem into matrix form, we have:

$$
\underset{w_1, w_2}{\operatorname{argmin}} \frac{1}{2} \left\|
\begin{bmatrix}
y^{1} \\
\vdots \\
y^{n}
\end{bmatrix} - 
\begin{bmatrix}
x^{1} & 1 \\
\vdots & \vdots \\
x^{n} & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
w_1 \\
w_2
\end{bmatrix}
\right\|^2_2
$$

Then, we denote the matrices above as $X^T, Y, w$ respectively, where $X^T \in \mathbb{R}^{n \times 2}$, $Y \in \mathbb{R}^n$, and $w \in \mathbb{R}^2$. The problem becomes:

$$
\underset{w}{\operatorname{argmin}} \frac{1}{2} \|Y - X^T w\|_2^2
$$

To solve this problem, we take the derivative of the objective function with respect to $w$ and set it to $0$. Using matrix calculus, we have:

$$
L = \frac{1}{2} \left( Y - X^T w \right)^T \left( Y - X^T w \right) = \frac{1}{2} \left( Y^T Y - Y^T X^T w - w^T X Y + w^T X^T X w \right)
$$

$$
\frac{\partial L}{\partial w} = -X Y - X Y + 2 X^T X w = 0
$$
We derive the Normal equation
$$
X^TXw =X^T Y
$$ 
If rank(X) = d, we have
$$
w = (X^T X)^{-1} X^T Y
$$ 
Otherwise, we can use pseudo-inverse of $X$.

<!-- --- -->

### General Case

Let's go back to the general case. We also discuss higher-order polynomial regression where $x^i$ is no longer a real number.

Suppose we have a dataset $\{(x^1, y^1), \ldots, (x^n, y^n)\}$, where $x^i \in \mathbb{R}^d$ and $y^i \in \mathbb{R}$.

We want to find a polynomial function $f(x) = w_0 + w_1 x + w_2 x^2 + \ldots + w_d x^d$ to fit the data:

$$
\underset{w_0, w_1, \ldots, w_d}{\operatorname{argmin}} \frac{1}{2} \left\|
\begin{bmatrix}
y^{(1)} \\
\vdots \\
y^{(N)}
\end{bmatrix} - 
\begin{bmatrix}
(x^{(1)})^d & \cdots & x^{(1)} & 1 \\
\vdots & \ddots & \vdots & \vdots \\
(x^{(N)})^d & \cdots & x^{(N)} & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
w_d \\
\vdots \\
w_1 \\
w_0
\end{bmatrix}
\right\|^2
$$

Here, $X^T \in \mathbb{R}^{n \times d}$, $Y \in \mathbb{R}^n$, and $w \in \mathbb{R}^d$. The problem is also:

$$
\underset{w}{\operatorname{argmin}} \frac{1}{2} \|Y - X^Tw\|_2^2
$$

Similarly, we take the derivative of the objective function with respect to $w$ and set it to 0. We get the same closed-form solution as above:

$$
w = (X^T X)^{-1} X^T Y
$$

<!-- --- -->

### Regularization

In practice, we may encounter the problem where $n < d + 1$. In this case, the matrix $X^T X$ is not invertible.

To address this, we can add a regularization term to the objective function. The objective function becomes:

$$
\underset{w}{\operatorname{argmin}} \frac{1}{2} \|Y - X^T w\|_2^2 + \frac{\lambda}{2} \|w\|_2^2
$$

The closed-form solution is modified to:

$$
w = (X^T X + \lambda I)^{-1} X^T Y
$$

Regularization also helps to make the parameters smaller and avoid overfitting.

---
### Interpretation
Updataed on Nov 21, 2025

#### *Empirical Risk Minimization Perspective*
$$
\underset{w}{\operatorname{argmin}} \ \frac{1}{2} \sum_{i=1}^n (y^{(i)} - w^T x^{(i)})^2
$$
#### *Statistical Perspective*
We assume the input $x$ and target $y$ follows a linear relationship with noise:
$$
y^{(i)} = w^T x^{(i)} + \epsilon^{(i)}
$$
If we assume these errors are **Independent and Identically Distributed (i.i.d.)** and follow a **Gaussian distribution** with mean 0 and variance $\sigma^2$:
$$
\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)
$$

Given an input $x^{(i)}$, the target $y^{(i)}$ follows a Gaussian distribution centered at the predicted value $w^T x^{(i)}$:
$$
p(y^{(i)} | x^{(i)}; w) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - w^T x^{(i)})^2}{2\sigma^2} \right)
$$

With **Maximum Likelihood estimation**, we want to find $w$ that maximize the Likelihood of the entire dataset. Since the samples are independent, the Likelihood function $L(w)$ is the product of their individual probabilities:
$$
L(w) = \prod_{i=1}^n p(y^{(i)} | x^{(i)}; w) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - w^T x^{(i)})^2}{2\sigma^2} \right)
$$

To simplify the calculation, we maximize the **Log-Likelihood** $\ell(w)$ instead:

$$
\begin{aligned}
\ell(w) &= \log L(w) \\
&= \sum_{i=1}^n \log \left[ \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - w^T x^{(i)})^2}{2\sigma^2} \right) \right] \\
&= \sum_{i=1}^n \left[ \log \left( \frac{1}{\sqrt{2\pi}\sigma} \right) - \frac{(y^{(i)} - w^T x^{(i)})^2}{2\sigma^2} \right] \\
&= n \underbrace{\log \left( \frac{1}{\sqrt{2\pi}\sigma} \right)}_{\text{constant}} - \frac{1}{2\sigma^2} \sum_{i=1}^n (y^{(i)} - w^T x^{(i)})^2
\end{aligned}
$$

To maximize $\ell(w)$, we need to minimize the negative term. Dropping the constants ($n, \sigma$), this becomes:
$$
\underset{w}{\operatorname{argmax}} \ \ell(w) \iff \underset{w}{\operatorname{argmin}} \ \frac{1}{2} \sum_{i=1}^n (y^{(i)} - w^T x^{(i)})^2
$$

**Conclusion**:
Under the assumption of Gaussian noise, **Maximum Likelihood Estimation (MLE)** is mathematically equivalent to minimizing the **Empirical Risk**.