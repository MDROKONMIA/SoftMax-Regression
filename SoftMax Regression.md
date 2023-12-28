---
title: SoftMax Regression
---
#SoftMaxRegression And Its Derivation

Introduction

In the previous chapter we introduce logistic regression that only
worked for binary class dataset. But in the real world exist multiclass
dataset there are doesn't applicable binary logistic regression. For
this problem we introduce a linear regression that are solve this
problem and work with multiclass dataset called SoftMax regression.
SoftMax regression nothing but specialization of Logistic regression.

## SoftMax Function

In logistic regression (chapter 5) we use a function that called sigmoid
function. Sigmoid function calculates probability of each data point to
satisfying one class. The SoftMax function calculates all probability
for all classes of each data point.

### Derivation of SoftMax Equation

Consider a classification problem which involved $k$ number of classes,
x feature vector and $y$ is the corresponding class, where
$y \in \left\lbrace 1,2,3,4,5,\ldots.,k \right\rbrace$

Now we would like a probability of $y$ given $x$ , $P\left( y|x \right)$
which is a vector of probabilities of $y$ given features $x$.

$$P\left( y|x \right) = \begin{bmatrix}
P\left( y = 1|x \right) \\
P\left( y = 2|x \right) \\
\begin{matrix}
P\left( y = 3|x \right) \\
 \vdots \\
P\left( y = k|x \right) \\
\end{matrix} \\
\end{bmatrix}$$

Recall that in [logistic
regression](https://medium.com/datadriveninvestor/fundamental-of-logistic-regression-4244cbc9aa3c), log-odd
for y=1 with respect to y=0 is assumed to have a linear relationship
with the independent variable x.

Using the same analogy, we can assume that the log-odd for $y = i$ with
respect to $y = k$ is assumed to have linear relationship with
the independent variable x.

$$\ln{\frac{P\left( y = i|x;w \right)}{P(y = k|x;w)} = {xw}_{i}}$$

$$\Rightarrow \frac{P\left( y = i|x \right)}{P\left( y = k \middle| x \right)} = e^{{xw}_{i}}$$

$$\Rightarrow P\left( y = i|x \right) = P(y = k|x)e^{{xw}_{i}}$$

Since, sum of $P(y = j|x)$ for $j = 1,2,3,\ldots,k$ is 1

$$\sum_{j = 1}^{k}{P\left( y = j \middle| k \right)} = \sum_{j = 1}^{k}{P\left( y = k \middle| x \right)e^{{xw}_{j}}}$$

$$\Rightarrow 1 = P\left( y = k \middle| x \right)\sum_{j = 1}^{k}e^{{xw}_{j}}$$

$$\Rightarrow \ P\left( y = k \middle| x \right) = \frac{1}{\sum_{j = 1}^{k}e^{{xw}_{j}}}$$

By substitution, we get

<!-- $$P\left( y = i|x \right) = \frac{e^{{xw}_{i}}}{\sum_{j = 1}^{k}e^{{xw}_{j}}}$$ -->

$$ P(y=k|x)={{e^{xw_i}}\over {\sum_{j=1}^{k} e^{xw_j}}} $$

Where

<!-- $w_{i} = \begin{bmatrix}
\ldots & w_{0i} & \ldots\\
\ldots & w_{1i} & \ldots\\
\ldots & w_{2i} & \ldots\\
\ldots & \vdots & \ldots\\
\ldots & w_{mi} & \ldots\\
\end{bmatrix} -->

$$ w_{k} =\begin{bmatrix}
\begin{matrix}
\ldots & w_{0k} & \ldots \\
\end{matrix} \\
\begin{matrix}
\cdots & w_{1k} & \ldots \\
\end{matrix} \\
\begin{matrix}
\begin{matrix}
\cdots & w_{2k} & \ldots \\
\end{matrix} \\
 \vdots \\
\begin{matrix}
\cdots & w_{mk} & \ldots \\
\end{matrix} \\
\end{matrix} \\
\end{bmatrix}$$

Now we write it into following mathematical notation

<!-- $$ {\sigma(z)}_{i} = \frac{e^{z_{i}}}{\sum_{j = 1}^{k}e^{z_{j}}} $$ -->

$$ \sigma(z)_i={{e^{xw_i}}\over {\sum_{j=1}^{k} e^{xw_j}}} $$

Where $e^{z_{i}} =$ standard exponential function for output vector,
$e^{z_{j}} =$ standard exponential function of output vector, $k =$
number of classes in the multiclass classifier, $\sigma =$ softmax,
$m$ =number of features

## Mathematical intuition of SoftMax Regression

Before proceed, let's gets introduced the indication function which
output 1 if argument is true otherwise 0

$$ 1\left\lbrace . \right\rbrace = \left\{ \begin{array}{r}
1,\ if\ y = i\ is\ true \\
0,\ other\ wise \\
\end{array} \right.\ $$

To derive the SoftMax regression model, we can start from the principle
of maximum likelihood. Suppose we have a training set of $N\ $examples,
where each example $x^{(i)}\  \in$ $\mathbb{R}^{M}$ is labeled with one
of K classes, $y^{(i)} \in \ \{ 1,\ 2,\ \ldots,\ K\}.$

To get the likelihood on the training data, we need to compute all of
the probabilities of $y = y⁽ⁱ⁾$ given $x⁽ⁱ⁾$ for i=1, 2, 3, ..., N. (N
is the total number of training data)

$$P\left( y^{(i)}|x^{(i)};\beta \right) = P\left( y = y^{(i)}|x^{(i)};\beta \right) = \prod_{k = 1}^{K}{P\left( y^{(i)} = k|x^{(i)};\beta \right)}^{1\left\lbrace y^{(i)} = k \right\rbrace}$$

To simplify the notation, we can write $\beta$ as an $K \times M$
matrix, where each column $\beta^{(k)}$ corresponds to parameters for
class k. Then, we can define the linear model for the log-odds of class
k as:

$$z_{k}^{(i)} = x^{(i)}\beta_{k}$$

$$\Rightarrow z_{k}^{(i)} = \begin{bmatrix}
x_{i0} & x_{i1} & \begin{matrix}
x_{i2} & \ldots & x_{im} \\
\end{matrix} \\
\end{bmatrix}\begin{bmatrix}
\begin{matrix}
\ldots & \beta_{0k} & \ldots \\
\end{matrix} \\
\begin{matrix}
\cdots & \beta_{1k} & \ldots \\
\end{matrix} \\
\begin{matrix}
\begin{matrix}
\cdots & \beta_{2k} & \ldots \\
\end{matrix} \\
 \vdots \\
\begin{matrix}
\cdots & \beta_{mk} & \ldots \\
\end{matrix} \\
\end{matrix} \\
\end{bmatrix}$$

The SoftMax function then converts the log-odds to probabilities, by
exponentiating and normalizing them:

$${{O_{k}^{(i)} = h}_{\beta}\left( x^{(i)} \right)}_{k} = \frac{e^{z_{k}^{(i)}}}{\sum_{j = 1}^{K}e^{z_{j}^{(i)}}} = P\left( y^{(i)} = k|x^{(i)};\beta \right)$$

Find prediction's SoftMax value $ith$ row

$$\Rightarrow z^{(i)} = \begin{bmatrix}
z_{1}^{(i)} & z_{2}^{(i)} & \begin{matrix}
z_{2}^{(i)} & \ldots & z_{k}^{(i)} \\
\end{matrix} \\
\end{bmatrix} = \begin{bmatrix}
x_{i0} & x_{i1} & \begin{matrix}
x_{i2} & \ldots & x_{im} \\
\end{matrix} \\
\end{bmatrix}\begin{bmatrix}
\begin{matrix}
\beta_{01} & \beta_{02} & \begin{matrix}
\beta_{03} & \ldots & \beta_{0k} \\
\end{matrix} \\
\end{matrix} \\
\begin{matrix}
\beta_{11} & \beta_{12} & \begin{matrix}
\beta_{13} & \ldots & \beta_{1k} \\
\end{matrix} \\
\end{matrix} \\
\begin{matrix}
\begin{matrix}
\beta_{21} & \beta_{22} & \begin{matrix}
\beta_{23} & \ldots & \beta_{2k} \\
\end{matrix} \\
\end{matrix} \\
 \vdots \\
\begin{matrix}
\beta_{m1} & \beta_{m2} & \begin{matrix}
\beta_{m3} & \ldots & \beta_{mk} \\
\end{matrix} \\
\end{matrix} \\
\end{matrix} \\
\end{bmatrix}$$

We want to find the parameters $\beta$ that maximize the likelihood of
the data, given by:

$$L(\beta) = P\left( Y|X;\beta \right) = \prod_{i = 1}^{N}{P\left( y^{(i)}|x^{(i)};\beta \right)}$$

$$L(\beta) = \prod_{i = 1}^{N}{\prod_{k = 1}^{k}\left( O_{k}^{(i)} \right)^{1\left\{ y^{(i)} = k \right\}}}$$

To make optimization easier, we can take average cross entropy Loss
function:

$$\Rightarrow J(\beta) = - \frac{1}{N}\ln\left\lbrack L(\beta) \right\rbrack = - \frac{1}{N}\sum_{i = 1}^{N}{\sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\}\ln\left( O_{k}^{(i)} \right)}}$$

$$\Rightarrow J(\beta) = - \frac{1}{N}\sum_{i = 1}^{N}{\sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\}\ln\left( \frac{e^{z_{k}^{(i)}}}{\sum_{j = 1}^{K}e^{z_{j}^{(i)}}} \right)}}$$

$$\Rightarrow J(\beta) = - \frac{1}{N}\sum_{i = 1}^{N}\left\lbrack \sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\}\ln\left( e^{z_{k}^{(i)}} \right)} - \left\lbrack \sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\}} \right\rbrack\ln\left\lbrack \sum_{j = 1}^{K}e^{z_{j}^{(i)}} \right\rbrack \right\rbrack$$

$$\Rightarrow J(\beta) = - \frac{1}{N}\sum_{i = 1}^{N}\left\lbrack \sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\} z_{k}^{(i)}\ln(e)} - \ln\left\lbrack \sum_{j = 1}^{K}e^{z_{j}^{(i)}} \right\rbrack \right\rbrack\ \ $$

Here $\sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\}} = 1$, because one
condition will be true and others is false.

$$\Rightarrow J(\beta) = - \frac{1}{N}\sum_{i = 1}^{N}\left\lbrack \sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\} x^{(i)}\beta_{k}\ } - \ln\left\lbrack \sum_{j = 1}^{K}e^{x^{(i)}\beta_{k}\ } \right\rbrack \right\rbrack$$

This is the cost function that we want to minimize with respect to
$\beta$. To do so, we can use gradient descent or any other optimization
algorithm. The gradient of the cost function with respect to $\beta_{k}$
is given by:

$$\nabla_{\beta_{k}}J(\beta) = - \frac{\partial J(\beta)}{\partial\beta_{k}\ }$$

Here $\beta_{k}$ is vector so applying vector differentiate,

$$\Rightarrow \nabla_{\beta_{k}\ }J(\beta) = - \frac{1}{N}\sum_{i = 1}^{N}\left\lbrack 1\left\{ y^{(i)} = k \right\} x^{(i)T} - \frac{x^{(i)T}.e^{x^{(i)}\beta_{k}\ }}{\sum_{j = 1}^{K}e^{x^{(i)}\beta_{j}}} \right\rbrack$$

$$\left\lbrack \frac{\partial(AX)}{\partial X} = A^{T}and\ \frac{\partial(e^{AX})}{\partial X} = (AX)^{'}{e^{AX} = A}^{T}.e^{AX},\ here\ A,X\ is\ matrix\ where\ A\ is\ indepent\ of\ X \right\rbrack$$

$${\Rightarrow \nabla}_{\beta_{k}}J(\beta) = - \frac{1}{N}\sum_{i = 1}^{N}{(1\{ y^{(i)} = k\} - O_{k}^{(i)})x^{(i)T}}$$

\[ See details of the derivation go to section 6.2.1\]

To prevent overfitting, we can also add a regularization term to the
cost function, such as the L2 norm of θ:

$$J(\theta) = - \frac{1}{N}\sum_{i = 1}^{N}{\sum_{k = 1}^{K}{y_{k}^{(i)}lnO_{k}^{(i)}}} + \frac{\lambda}{2}\sum_{k = 1}^{K}{\sum_{j = 1}^{M}\beta_{jk}^{2}}$$

where λ is the regularization parameter that controls the trade-of
between the data fit and the model complexity. The gradient of the
regularization term is simply $\lambda\beta_{k},$ so the update rule for
gradient descent becomes:

$$\beta_{k}: = \beta_{k} - \alpha\left( - \frac{1}{N}\sum_{i = 1}^{N}{(y_{k}^{(i)} - O_{k}^{(i)})x^{(i)T} + \lambda\beta_{k})} \right)$$

$$\Rightarrow \beta_{k}: = \beta_{k} + \frac{\alpha}{N}\left( \sum_{i = 1}^{N}{(y_{k}^{(i)} - O_{k}^{(i)})x^{(i)T} + \lambda\beta_{k})} \right)$$

### Partial derivation calculation:

$$J(\beta) = - \frac{1}{N}\sum_{i = 1}^{N}E_{i},\ where\ E_{i} = \sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\}\ln\left( O_{k}^{(i)} \right)}\ and\ O_{k}^{(i)} = \frac{e^{z_{k}^{(i)}}}{\sum_{j = 1}^{K}e^{z_{j}^{(i)}}}$$

Calculating $\frac{\partial E_{i}}{\partial\beta_{mj}}$,
$m = \left\{ 1,2,3,4,5,\ldots\ldots\ldots,M \right\}$:

$$\frac{\partial E_{i}}{\partial\beta_{mj}} = \frac{\partial\left( \sum_{k = 1}^{K}{1\left\{ y^{(i)} = k \right\}\ln\left( O_{k}^{(i)} \right)} \right)}{\partial\beta_{mj}} = \sum_{k = 1}^{K}{\frac{\partial}{\partial\beta_{mj}}\left( 1\left\{ y^{(i)} = k \right\}\ln\left( O_{k}^{(i)} \right) \right)}$$

Now, $O_{k}^{(i)}$ are not direct function of $z_{j}^{(i)}$, we can
apply chain rule and $1\left\{ y^{(i)} = k \right\}$ is equivalent of
$y_{k}^{(i)}$

$$\Rightarrow \frac{\partial E_{i}}{\partial\beta_{mj}} = \sum_{k = 1}^{K}\left\lbrack \frac{\partial\left\lbrack y_{k}^{(i)}\ln\left( O_{k}^{(i)} \right) \right\rbrack}{\partial O_{k}^{(i)}} \times \frac{\partial O_{k}^{(i)}}{\partial z_{j}^{(i)}} \times \frac{\partial z_{j}^{(i)}}{\partial\beta_{jm}} \right\rbrack$$

Focus on $\frac{\partial O_{k}^{(i)}}{\partial z_{j}^{(i)}}$ term or
Derivatives of SoftMax function

$$\frac{\partial O_{k}^{(i)}}{\partial z_{j}^{(i)}} = \frac{\partial}{\partial z_{j}^{(i)}}\left\lbrack \frac{e^{z_{k}^{(i)}}}{\sum_{j = 1}^{K}e^{z_{j}^{(i)}}} \right\rbrack$$

$$\frac{\partial O_{k}^{(i)}}{\partial z_{j}^{(i)}} = \frac{\partial}{\partial z_{j}^{(i)}}\left\lbrack \frac{\left( \sum_{j = 1}^{K}e^{z_{j}^{(i)}} \right)\frac{\partial\left( e^{z_{k}^{(i)}} \right)}{\partial z_{j}^{(i)}} - e^{z_{k}^{(i)}}\frac{\partial\left( \sum_{j = 1}^{K}e^{z_{j}^{(i)}} \right)}{\partial z_{j}^{(i)}}}{\left\lbrack \sum_{j = 1}^{K}e^{z_{j}^{(i)}} \right\rbrack^{2}} \right\rbrack$$

Case-I: If j==k,

$$\frac{\partial O_{k}^{(i)}}{\partial z_{j}^{(i)}} = \frac{\partial}{\partial z_{j}^{(i)}}\left\lbrack \frac{e^{z_{k}^{(i)}}\sum_{j = 1}^{K}e^{z_{j}^{(i)}} - e^{z_{k}^{(i)}}.e^{z_{j}^{(i)}}}{\left\lbrack \sum_{j = 1}^{K}e^{z_{j}^{(i)}} \right\rbrack^{2}} \right\rbrack = \ O_{k}^{(i)}\left( 1 - O_{j}^{(i)} \right)$$

Case-II if $j \neq k$,

$$\frac{\partial O_{k}^{(i)}}{\partial z_{j}^{(i)}} = \frac{\partial}{\partial z_{j}^{(i)}}\left\lbrack \frac{0 - e^{z_{k}^{(i)}}.e^{z_{j}^{(i)}}}{\left\lbrack \sum_{j = 1}^{K}e^{z_{j}^{(i)}} \right\rbrack^{2}} \right\rbrack = \  - O_{k}^{(i)} \times O_{j}^{(i)}$$

Summery above all:

$\frac{\partial O_{k}^{(i)}}{\partial z_{j}^{(i)}} = \left\{ \begin{array}{r}
O_{k}^{(i)}\left( 1 - O_{j}^{(i)} \right)\ if\ j = k \\
\  - O_{k}^{(i)} \times O_{j}^{(i)}\ if\ j \neq k \\
\end{array} \right.\  = O_{k}^{(i)}\left( \delta_{kj} - O_{j}^{(i)} \right)$,
where $\delta_{kj}$ is the Kronecker delta that equals 1 if
$k\mathbf{=}j$, and 0 otherwise.

Focus
on$\frac{\partial\left\lbrack y_{k}^{(i)}\ln\left( O_{k}^{(i)} \right) \right\rbrack}{\partial O_{k}^{(i)}}$
term,

$$\frac{\partial\left\lbrack y_{k}^{(i)}\ln\left( O_{k}^{(i)} \right) \right\rbrack}{\partial O_{k}^{(i)}} = \frac{y_{k}^{(i)}}{O_{k}^{(i)}}$$

And

$$\frac{\partial z_{j}^{(i)}}{\partial\beta_{mj}} = \frac{\partial\left( \beta^{(j)T}x^{(i)} \right)}{\partial\beta_{mj}} = x_{m}^{(i)}$$

Substituting all, then we get

$$\frac{\partial E_{i}}{\partial\beta_{mj}} = \sum_{k = 1}^{K}\left\lbrack O_{k}^{(i)}\left( \delta_{kj} - O_{j}^{(i)} \right) \times \frac{y_{k}^{(i)}}{O_{k}^{(i)}} \times x_{m}^{(i)} \right\rbrack$$

$$\Rightarrow \frac{\partial E_{i}}{\partial\beta_{mj}} = \sum_{k = 1}^{K}\left\lbrack \left( \delta_{kj} - O_{j}^{(i)} \right) \times y_{k}^{(i)} \times x_{m}^{(i)} \right\rbrack$$

$$\Rightarrow \frac{\partial E_{i}}{\partial\beta_{mj}} = x_{m}^{(i)}\sum_{k = 1}^{K}\left\lbrack \left( \delta_{kj} - O_{j}^{(i)} \right) \times y_{k}^{(i)} \right\rbrack$$

$$\Rightarrow \frac{\partial E_{i}}{\partial\beta_{mj}} = x_{m}^{(i)}\left\lbrack \sum_{k = 1}^{K}{y_{k}^{(i)}\delta_{kj}} - \sum_{k = 1}^{K}{O_{j}^{(i)}y_{k}^{(i)}} \right\rbrack$$

$$\Rightarrow \frac{\partial E_{i}}{\partial\beta_{mj}} = x_{m}^{(i)}\left\lbrack \sum_{k = 1}^{K}{y_{k}^{(i)}\delta_{kj}} - O_{j}^{(i)}\sum_{k = 1}^{K}y_{k}^{(i)} \right\rbrack\ \ \ $$

$$\Rightarrow \frac{\partial E_{i}}{\partial\beta_{mj}} = x_{m}^{(i)}\left\lbrack y_{k}^{(i)} - O_{j}^{(i)} \right\rbrack\ $$

Since $\sum_{k = 1}^{K}y_{k}^{(i)} = 1$ and
$\sum_{k = 1}^{K}{y_{k}^{(i)}\delta_{kj}} = y_{k}^{(i)}$

Now, we find matrix for $\frac{\partial E_{i}}{\partial\beta_{k}}$:

$$\frac{\partial E_{i}}{\partial\beta_{k}} = \begin{bmatrix}
\frac{\partial E_{i}}{\partial\beta_{0k}} \\
\frac{\partial E_{i}}{\partial\beta_{1k}} \\
\begin{matrix}
\frac{\partial E_{i}}{\partial\beta_{2k}} \\
 \vdots \\
\frac{\partial E_{i}}{\partial\beta_{mk}} \\
\end{matrix} \\
\end{bmatrix} = \begin{bmatrix}
x_{0}^{(i)}\left( y_{k}^{(i)} - O_{j}^{(i)} \right) \\
x_{1}^{(i)}\left( y_{k}^{(i)} - O_{j}^{(i)} \right) \\
\begin{matrix}
x_{2}^{(i)}\left( y_{k}^{(i)} - O_{j}^{(i)} \right) \\
 \vdots \\
x_{m}^{(i)}\left( y_{k}^{(i)} - O_{j}^{(i)} \right) \\
\end{matrix} \\
\end{bmatrix} = \left( y_{k}^{(i)} - O_{j}^{(i)} \right)\begin{bmatrix}
x_{0}^{(i)} \\
x_{1}^{(i)} \\
\begin{matrix}
x_{2}^{(i)} \\
 \vdots \\
x_{m}^{(i)} \\
\end{matrix} \\
\end{bmatrix} = \left( y_{k}^{(i)} - O_{j}^{(i)} \right)x^{(i)T}$$

Now, we calculate derivation of $J(\beta)$ w.r.t $\beta^{(k)}$:

$$\frac{\partial J(\beta)}{\partial\beta_{k}} = - \frac{1}{N}\sum_{i = 1}^{N}{\left( y_{k}^{(i)} - O_{j}^{(i)} \right)x^{(i)T}}$$

Gradient descent update rule,

$$\beta_{k} ≔ \beta_{k} + \frac{\alpha}{N}\sum_{i = 1}^{N}{\left( y_{k}^{(i)} - O_{j}^{(i)} \right)x^{(i)T}}$$

$\alpha =$ learning rate
