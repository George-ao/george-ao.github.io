---
title: "Linear Regression"
date: 2024-01-30
---

### *Simple Case*

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
\underset{w}{\operatorname{argmin}} \frac{1}{2} \|Y - X w\|_2^2
$$

To solve this problem, we take the derivative of the objective function with respect to $w$ and set it to 0. Using matrix calculus, we have:

$$
L = \frac{1}{2} \left( Y - X^T w \right)^T \left( Y - X^T w \right) = \frac{1}{2} \left( Y^T Y - Y^T X^T w - w^T X Y + w^T X^T X w \right)
$$

$$
\frac{\partial L}{\partial w} = -X Y - X Y + 2 X^T X w = 0
$$

$$
w = (X^T X)^{-1} X^T Y
$$

---

### *General Case*

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
\underset{w}{\operatorname{argmin}} \frac{1}{2} \|Y - X w\|_2^2
$$

Similarly, we take the derivative of the objective function with respect to $w$ and set it to 0. We get the same closed-form solution as above:

$$
w = (X^T X)^{-1} X^T Y
$$

---

### *Regularization*

In practice, we may encounter the problem where $n < d + 1$. In this case, the matrix $X^T X$ is not invertible.

To address this, we can add a regularization term to the objective function. The objective function becomes:

$$
\underset{w}{\operatorname{argmin}} \frac{1}{2} \|Y - X w\|_2^2 + \frac{\lambda}{2} \|w\|_2^2
$$

The closed-form solution is modified to:

$$
w = (X^T X + \lambda I)^{-1} X^T Y
$$

Regularization also helps to make the parameters smaller and avoid overfitting.