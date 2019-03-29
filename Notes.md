---
title: "Notes on Fundamentals of Statistics"
author: "Romain Mondon-Cancel (skasch)"
header-includes:
   - \usepackage{bbm}
output:
    pdf_document:
        extra_dependencies: ["bbm"]
---

# Fundamentals of Statistics

## Introduction

These notes are written by [Romain Mondon-Cancel](http://github.com/skasch) for
the course [Fundamentals of Statistics](https://courses.edx.org/courses/course-v1:MITx+18.6501x+3T2018/course/).

## Unit 1: Introduction to Statistics

### Lecture 1: What is Statistics?

### Lecture 2: Probability Redux

#### Law of Large Numbers, Central Limit Theorem

Let $X_1, \ldots, X_n$ be **independent and identically distributed** (*i.i.d*)
**random variables** (*r.v.*). Let $\mu = \E[X]$ the expectation and
$\sigma^2 = \V[X]$ the variance of these random variables.

We define the **sample average** of our $X_i$ as:

$$\bar{X}_n = \frac1n \sum_{i=1}^nX_i$$

**Law of Large Numbers** (*LLN*):

$$\tag{1.1} \bar{X}_n \conv{\P, a.s.}\mu$$

Where $\P$ designs convergence in probability and $a.s.$ convergence
almost surely.

**Central Limit Theorem** (*CLT*):

$$\tag{1.2} \sqrt{n}\frac{\bar{X}_n - \mu}{\sigma} \convd \mathcal{N}(0, 1)$$

Equivalently, we also have:

$$\sqrt{n}(\bar{X}_n - \mu) \convd \mathcal{N}(0, \sigma^2)$$

If $n$ is not large enough to apply the *CLT*, we have *Hoeffding's inequality*.
If $X \in [a, b]$ almost surely, then

$$\P[|\bar{X}_n - \mu| \geq \varepsilon] \leq 2\exp\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right)$$

#### Properties of normal distributions

Linear transformation: given a random variable $X \sim \Norm(\mu, \sigma^2)$,

$$\tag{1.3} \forall a, b \in \R, aX + b \sim \Norm(a\mu + b, a^2\sigma^2)$$

We can therefore standardize any normal distribution:

$$\tag{1.4} Z = \frac{X - \mu}{\sigma} \sim \Norm(0, 1)$$

We therefore have:

$$\P[u \leq X \leq v] = \P\left[\frac{u-\mu}{\sigma} \leq Z \leq \frac{v-\mu}{\sigma}\right]$$

The standard normal probability is also symmetric, which means:

$$\P[Z \geq x] = \P[Z \leq -x]$$

The **quantile** of order $1-\alpha$ of a random variable $X$ is the number $q_\alpha$ such as:

$$\tag{1.5} \P[X \leq q_\alpha] = 1 - \alpha$$

#### Types of convergence

Let $(T_n)_n$ be a sequence of random variables, and $T$ a random variable (which can be deterministic).

**Convergence almost surely**:

$$\tag{1.6} T_n \convas T \iff \P\left[\left\{\omega: T_n(\omega) \conv{} T(\omega)\right\} \right] = 1$$

**Convergence in probability**:

$$\tag{1.7} T_n \convp T \iff \forall \varepsilon > 0, \P[|T_n - T| \geq \varepsilon] \conv{} 0$$

**Convergence in distribution**:

$$\tag{1.8} T_n \convd T \iff \E[f(T_n)] \conv{} \E[f(T)]$$

for all function $f$ continuous and bounded.

Convergence *almost surely* implies convergence *in probability*, which implies convergence *in distribution*.

If the limit has a density, then convergence *in distribution* implies convergence *in probability*.

#### Operations on convergence

Let $(T_n)_n, (U_n)_n$ but two sequences of random variables, and $T, U$ be two random variables. If

$$T_n \conv{a.s./\P} T, U_n \conv{a.s/\P} U$$

then:

$$\tag{1.9} T_n + U_n \conv{a.s/\P} T + U$$

$$\tag{1.10} T_nU_n \conv{a.s/\P} TU$$

if in addition, $U \neq 0$ almost surely, then

$$\tag{1.11} \frac{T_n}{U_n} \conv{a.s/\P} \frac{T}U$$

**Slutsky's Theorem**. Let $u \in \R$. If

$$T_n \convd T, U_n \convp u$$

then:

$$\tag{1.12} T_n + U_n \convd T + u$$

$$\tag{1.13} T_nU_n \convd uT_n$$

if in addition, $u \neq 0$, then

$$\tag{1.14} \frac{T_n}{U_n} \convd \frac1uT$$

**Continuous mapping theorem**. Let $f \in \mathcal{C}(\R)$. If

$$T_n \conv{a.s./\P/(d)}T$$

then:

$$\tag{1.15} f(T_n) \conv{a.s./\P/(d)} f(T)$$

## Unit 2: Foundation of Inference

### Lecture 3: Parametric Statistical Models

Trinity of statistical inference: *estimation*, *confidence intervals* and *hypothesis testing*.

Given $X_1, \ldots, X_n \iid \P$, the goal is to learn the distribution $\P$.

A **statistical model** is defined as:

$$\tag{2.1} \statmodel$$

where

* $E$ is the *sample space*,
* $(\P_\theta)_{\theta \in \Theta}$ is a *family of probability measures* on $E$,
* $\Theta$ is called the *parameter set*.

The model is **well specified** if

$$\tag{2.2} \exists \theta \in \Theta, \P_\theta = \P$$

The model is called:

* **parametric** if $\Theta \subseteq \R^d$ for some $d \in \N$,
* **non-parametric** if $\Theta$ is infinite-dimensional,
* **semi-parametric** if $\Theta = \Theta_1 \times \Theta_2$ where $\Theta_1 \subseteq \R^d$ is the parameter set we are interested in, and $\Theta_2$ is the *nuisance parameter* set.

The parameter $\theta$ is called **identifiable** if the mapping $\theta \mapsto \P_\theta$ is injective, i.e.:

$$\tag{2.3} \forall \theta_1, \theta_2 \in \Theta, \theta_1 \neq \theta_2 \implies \P_{\theta_1} \neq \P_{\theta_2}$$

### Lecture 4: Parametric Estimation and Confidence Intervals

A **statistic** is any measurable function of the sample.

An **estimator of $\theta$** is any statistic whose expression doesn't depend on $\theta$.

An estimator $\thn$ of $\theta$ is **weakly (resp. strongly) consistent** if:

$$\tag{2.4} \thn \conv{\P\;(resp.\;a.s.)} \theta \quad (\P_\theta)$$

An estimator $\thn$ of $\theta$ is **asymptotically normal** if:

$$\tag{2.5} \sqrt{n}(\thn - \theta) \convd \Norm(0, \sigma^2)$$

The quantity $\sigma^2$ is called the **asymptotic variance** of $\thn$, noted

$$\tag{2.6} V(\thn) = \sigma^2$$

The **bias** of an estimator $\thn$ of $\theta$ is defined as:

$$\tag{2.7} \Bias(\thn) = \E[\thn] - \theta$$

If $\Bias(\thn) = 0$, then we say that $\thn$ is **unbiased**.

The **quadratic risk** (or **risk**) of an estimator $\thn$ is

$$\tag{2.8} R(\thn) = \E\left[(\thn - \theta)^2\right] = \V[\thn] + \Bias(\thn)^2$$

Let $\statmodel$ be a statistical model with $\Theta \subseteq \R$ associated with observations $X_1, \ldots, X_n$, and $\alpha \in (0, 1)$.

A **confidence interval (*C.I.*) of level $1 - \alpha$** is any random interval $\mathcal{I}$ independent from $\theta$ such as:

$$\tag{2.9} \forall \theta \in \Theta, \P_\theta[\mathcal{I} \ni \theta] \geq 1 - \alpha$$

A **C.I. of asymptotic level $1 - \alpha$** is any random interval $\mathcal{I}_n$ independent from $\theta$ such as:

$$\tag{2.10} \forall \theta \in \Theta, \limn\P_\theta[\mathcal{I}_n \ni \theta] \geq 1 - \alpha$$

*Example*: $X_1, \ldots, X_n \iid \Ber(p)$. Our estimator is $\est{p} = \bar{X}_n$. From the CLT:

$$\sqrt{n}\frac{\bar{X}_n - p}{\sqrt{p(1-p)}} \convd\Norm(0,1)$$

Therefore:

$$\P\left[\frac{\sqrt{n}}{\sqrt{p(1-p)}}|\bar{X}_n - p| \leq q_{\alpha/2}\right] = 1-\alpha
$$

$$\P\left[\left[\bar{X}_n - \sqrt{\frac{p(1-p)}{n}}q_{\alpha/2}, \bar{X}_n + \sqrt{\frac{p(1-p)}{n}}q_{\alpha/2}\right] \ni p\right] = 1-\alpha$$

But **this is not an asymptotic C.I.** because the interval depends on $p$. We have three ways to find an actual confidence inteval:

**Convervative bounds**:

$$\forall p \in (0, 1), p(1-p) \leq \frac14 \implies \mathcal{I}_{conv} = \bar{X}_n + \left[-\frac{q_{\alpha/2}}{2\sqrt{n}}, \frac{q_{\alpha/2}}{2\sqrt{n}}\right]$$

**Solve bounds**, by solving the quadratic equation in $p$:

$$(p - \bar{X}_n)^2 \leq \frac{q^2_{\alpha/2}}{n}p(1-p) \implies \mathcal{I}_{solve} = [p_1, p_2]$$

**Plug-in bounds**; as $\est{p} \convas p$, we can apply Slutsky's theorem:

$$\sqrt{n}\frac{\bar{X}_n-p}{\sqrt{\est{p}(1 -\est{p})}} \convd \Norm(0, 1) \implies \mathcal{I}_{plug} = \bar{X}_n + q_{\alpha/2}\left[-\sqrt{\frac{\est{p}(1-\est{p})}{n}}, \sqrt{\frac{\est{p}(1-\est{p})}{n}}\right]$$

All three intervals verify:

$$\limn \P[\mathcal{I} \ni p] \geq 1 - \alpha$$

If $\mathcal{I}$ is a confidence interval of level $\alpha$, it is also a confidence interval of level $\beta$ for all $\beta \leq \alpha$.

A confidence interval of level $\alpha$ means that if we repeat the experience multiple times, then the real parameter will be in the confidence interval with frequency $\alpha$. It **does not mean** that the real parameter is in the interval with a certain confidence (as the real parameter is deterministic).

### Lecture 5: Delta Method and Confidence Intervals

The **Delta Method**. Let $(X_n)_n$ be a sequence of r.v. such as

$$\sqrt{n}(X_n - \theta) \convd \Norm(0, \sigma^2)$$

for some $\theta \in \R, \sigma^2 > 0$. Let $g: \R \rightarrow \R$ continuously differentiable at $\theta$. Then $(g(X_n))_n$ is asymptotically normal around $g(\theta)$ and:

$$\tag{2.11}\sqrt{n}(g(X_n) - g(\theta)) \convd \Norm(0, g'(\theta)^2\sigma^2)$$

### Lecture 6: Introduction to Hypothesis Testing, and Type 1 and Type 2 Errors

Let us consider a sample $X_1, \ldots, X_n$ of i.i.d r.v. and a statistical model $\statmodel$. Let us consider $\Theta_0 \amalg \Theta_1 = \Theta$.

* $H_0: \theta \in \Theta_0$ is called the **null hypothesis**,
* $H_1: \theta \in \Theta_1$ is called the **alternative hypothesis**.

We are testing $H_0$ against $H_1$ to decide whether to **reject $H_0$** by looking for evidence against $H_0$.

$H_0$ and $H_1$ do not play symmetric roles in hypothesis testing: $H_0$ is the default hypothesis when we lack evidence.

A **test** is a statistic $\psi : E^n \rightarrow \{0, 1\}$ such as:

* If $\psi(x_1, \ldots, x_n) = 0$ then $H_0$ is not rejected,
* If $\psi(x_1, \ldots, x_n) = 1$ then $H_0$ is rejected and $H_1$ is accepted.

The **rejection region** of a test $\psi$ is:

$$\tag{2.12} R_\psi = \{\xx \in E^n, \psi(\xx) = 1\}$$

The **type 1** error of a test $\psi$ corresponds to rejecting $H_0$ when it is actually true:

$$\tag{2.13} \fun{\alpha_\psi}{\Theta_0}{[0, 1]}{\theta}{\P_\theta[\psi(\XX) = 1]}$$

The **type 2** error of a test $\psi$ corresponds to not rejecting $H_0$ when $H_1$ is actually true:

$$\tag{2.14} \fun{\beta_\psi}{\Theta_1}{[0, 1]}{\theta}{\P_\theta[\psi(\XX) = 0]}$$

Type 1 and type 2 errors are summaries as follows ($R$ is real, $T$ is test):

$$\begin{array}{ccc}R\char`\\ T & H_0 & H_1 \\ H_0 & \checkmark & 1 \\ H_1 & 2 & \checkmark \end{array}$$

The **power** of a test $\psi$ is:

$$\tag{2.15} \pi_\psi = \inf_{\theta \in \Theta_1}(1 - \beta_\psi(\theta))$$

### Lecture 7: Hypothesis Testing (Continued): Levels and P-values

A test $\psi$ has **level $\alpha$** if:

$$\tag{2.16} \forall \theta \in \Theta_0, \alpha_\psi(\theta) \leq \alpha$$

A test $\psi$ has **asymptotic level $\alpha$** if:

$$\tag{2.17} \forall \theta \in \Theta_0, \limn \alpha_{\psi_n}(\theta) \leq \alpha$$

In general, a test has the form 

$$\psi = \one\{T_n > c\}$$

for some statistic $T_n$ and threshold $c$. $T_n$ is called the **test statistic** and the rejection region is $R_\psi = \{T_n > c\}$.

We have two main types of tests:

* If $H_0: \theta = \theta_0$ and $H_1: \theta \neq \theta_0$, it is a **two-sided test**,
* If $H_0: \theta \leq \theta_0$ and $H_1: \theta > \theta_0$, it is a **one-sided test**.

Example: Let $X_1, …, X_n \iid \mathcal{D}(\mu)$ where $\mu$ and $\sigma^2$ are the expectation and variance of $\mathcal{D}(\mu)$. Let's consider $\mu_0$.

We want to test $H_0: \mu = \mu_0$ against $H_1: \mu \neq \mu_0$ with asymptotic level $\alpha$.

$$\tag{2.18} T_n = \sqrt{n}\frac{\est{\mu} - \mu_0}{\sigma} \convd \Norm(0, 1)$$

Therefore, if $H_0$ is true, then

$$\P[|T_n| > q_{\alpha/2}] \conv{} \alpha$$

Let $\psi_\alpha = \one\{|T_n| > q_{\alpha/2}\}$.

Now, we want to test $H_0: \mu \leq \mu_0$ and $H_1: \mu > \mu_0$ with asymptotic level $\alpha$. Which value of $\mu \in \Theta_0$ should we consider?

Type 1 error is the function $\mu \mapsto \P_\mu[\psi(\XX) = 1]$. To control the level, we need the $\mu$ that maximises this expression over $\Theta_0$: clearly, this happens for $\mu = \mu_0$. Therefore, if $H_0$ is true, then

$$\P_{\mu_0}[T_n > q_\alpha] \conv{} \alpha$$

If $H_1: \mu < \mu_0$, then we want $\P_{\mu_0}[T_n < -q_\alpha]$.

The **(asymptotic) p-value** of a test $\psi_\alpha$ is the smallest (asymptotic) level $\alpha$ at which $\psi_\alpha$ rejects $H_0$. The p-value is random and depends on the sample.

The *golden rule*: $H_0$ is rejected by $\psi_\alpha$ at any asymptotic level $\alpha \geq \pval(\XX)$.

## Unit 3: Methods of estimation

### Lecture 8: Distance measures between distributions

Let $\statmodel$ a statistical model, associated with a sample of i.i.d r.v. $X_1, \ldots, X_n$. We assume the model is well specified, i.e. $\exists \tth \in \Theta, X \sim \P_\tth$. $\tth$ is called the **true parameter**.

We want to find an estimator $\thn$ such that $\P_{\thn}$ is close to $\P_\tth$.

The **total variation distance** between two probability measures $\P_1$ and $\P_2$ is defined as:

$$\tag{3.1} \TV(\P_1, \P_2) = \max_{A \subset E}|\P_1(A) - \P_2(A)|$$

If $E$ is discrete, and $p_1, p_2$ are the PMF associated with $\P_1$ and $\P_2$:

$$\tag{3.2} \TV(\P_1, \P_2) = \frac12\sum_{x\in E}|p_1(x) - p_2(x)|$$

If $E$ is continuous, and $f_1, f_2$ are the PDF associated with $\P_1$ and $\P_2$:

$$\tag{3.3} \TV(\P_1, \P_2) = \frac12\int_E|f_1(x) - f_2(x)|dx$$

Proprieties of the total variation distance:

* The $\TV$ is **symmetric**: $\forall \P_1, \P_2, \TV(\P_1, \P_2) = \TV(\P_2, \P_1)$
* The $\TV$ is **positive**: $\forall \P_1, \P_2, 0 \leq \TV(\P_1, \P_2) \leq 1$
* The $\TV$ is **definite**: $\forall \P_1, \P_2, \TV(\P_1, \P_2) = 0 \implies \P_1 = \P_2$ p.p.
* The $\TV$ verifies the **triangle inequality**: $\forall \P_1, \P_2, \P_3, \TV(\P_1, \P_3) \leq \TV(\P_1, \P_2) + \TV(\P_2, \P_3)$

Therefore, $\TV$ is a distance between probability distributions.

Problem 1: $\TV$ cannot compare discrete and continuous distributions, which means we cannot trust it to give a reliable estimation of distance between unrelated distributions.

Problem 2: We cannot build an estimator $\theta \mapsto \widehat{\TV}(\P_\theta, \P_\tth)$ as we do not know $\tth$.

Hence we need another "distance" between distributions. The **Kullback-Leibler (*KL*) divergence** between two probability distributions $\P_1, \P_2$ is defined as:

$$\tag{3.4} \KL(\P_1, \P_2)= \begin{cases}\displaystyle\sum_{x\in E}p_1(x)\ln\frac{p_1(x)}{p_2(x)} & \text{if }E\text{ is discrete} \\ \displaystyle \int_Ef_1(x)\ln\frac{f_1(x)}{f_2(x)}dx & \text{if }E\text{ is continuous}\end{cases}$$

Proprieties of the $\KL$ divergence:

* In general, $\KL(\P_1, \P_2) \neq \KL(\P_2, \P_1)$
* The $\KL$ is **positive**: $\forall \P_1, \P_2, \KL(\P_1, \P_2) \geq 0$
* The $\KL$ is **definite**: $\forall \P_1, \P_2, \KL(\P_1, \P_2) = 0 \implies \P_1 = \P_2$
* In general, $\KL(\P_1, \P_3) \not\leq \KL(\P_1, \P_2) + \KL(\P_2, \P_3)$

$\KL$ is not a distance, it's a divergence. But we still have that $\tth$ is the only minimizer of $\theta \mapsto \KL(\P_\tth, \P_\theta)$.

### Lecture 9: Maximum Likelihood Estimation

$$\begin{aligned}\KL(\P_\tth, \P_\theta) &= \E_\tth\left[\ln\frac{f_\tth(X)}{f_\theta(X)}\right] \\ &= \E_\tth[\ln f_\tth(X)] - \E_\tth[\ln f_\theta(X)]\end{aligned}$$

Therefore, by the LLN the $\KL$ divergence can be estimated by taking the average of $\ln f_\theta(X_i)$:

$$\widehat{\KL}(\P_\tth, \P_\theta) = \text{constant} - \frac1n\sum_{i=1}^n\ln f_\theta(X_i)$$

Therefore:

$$\begin{aligned}
    \argmin{\theta \in \Theta}\widehat{\KL}(\P_\tth, \P_\theta) &= \argmin{\theta \in \Theta} -\frac1n\sum_{i=1}^n\ln f_\theta(X_i) \\
    &= \argmax{\theta \in \Theta}\ln\prod_{i=1}^nf_\theta(X_i) \\
    &= \argmax{\theta \in \Theta}\prod_{i=1}^nf_\theta(X_i)
\end{aligned}$$

This is the **maximum likelihood principle**.

Let $\statmodel$ be a statistical model associated with  a sample of i.i.d r.v. $X_1, \ldots, X_n$. The **likelihood** of the model is the function:

$$\tag{3.5}\fun{L_n}{E^n \times \Theta}{[0, 1]}{(x_1, \ldots, x_n, \theta)}{\begin{cases}
    \displaystyle \prod_{i=1}^np_\theta(x_i) & \text{if }E\text{ is discrete} \\
    \displaystyle \prod_{i=1}^nf_\theta(x_i) & \text{if }E\text{ is continous}
\end{cases}}$$

The **maximum likelihood estimator** of $\theta$ is defined as:

$$\tag{3.6} \thn^{MLE} = \argmax{\theta \in \Theta} L_n(X_1, \ldots, X_n, \theta)$$

In practice, we often use the **log-likelihood** $\ell_n(x_1, \ldots, x_n, \theta) = \ln L_n(x_1, \ldots, x_n, \theta)$.

A function twice differentiable $h : \Theta \subseteq \R \rightarrow \R$ is **concave** (resp. **strictly**) if

$$\tag{3.7} \forall \theta \in \Theta, h''(\theta) \leq 0 \quad\text{(resp. }< 0\text{)}$$

$h$ is **convex** (resp. **strictly**) if $-h$ is concave (resp. strictly).

A con-**v**-ex function is shaped like a **v**.

Let $h: \Theta \subseteq \R^d \rightarrow \R$ be a multivariate function. The **gradient** vector of $h$ is:

$$\tag{3.8} \nabla h(\ttheta) = \begin{pmatrix}\displaystyle \diff{h}{\theta_1}(\ttheta) \\ \displaystyle \vdots \\ \displaystyle \diff{h}{\theta_d}(\ttheta)\end{pmatrix}$$

The **Hessian matrix** of $h$ is:

$$\tag{3.9} \HH h(\ttheta) = \begin{pmatrix}
    \displaystyle \diffk{h}{\theta_1}{2}(\ttheta) & \cdots & \displaystyle \diffd{h}{\theta_1}{\theta_d}(\ttheta) \\
    \vdots & \ddots & \vdots \\
    \displaystyle \diffd{h}{\theta_d}{\theta_1}(\ttheta) & \cdots & \displaystyle \diffk{h}{\theta_d}{2}(\ttheta) \\
\end{pmatrix} \in \R^{d\times d}$$

Therefore, $h$ is **concave** (resp. **strictly**) if

$$\tag{3.10} \forall \xx \in \R^d \setminus \{\zz\}, \ttheta \in \Theta, \xx^T\HH h(\ttheta)\xx \leq 0 \quad\text{(resp. }< 0\text{)}$$

If $h$ is strictly concave and has a maximum, then this maximum is unique and is reached for

$$\tag{3.11} \begin{cases}
    h'(\theta) = 0 & \text{if }\Theta \subseteq \R \\
    \nabla h(\ttheta) = \zz & \text{if }\Theta \subseteq \R^d
\end{cases}$$

### Lecture 10: Consistency of MLE, Covariance Matrices, and Multivariate Statistics

Under mild regularity conditions, we have

$$\tag{3.12} \thn^{MLE} \convp \tth$$

Notably if the model is well specified.

In general, when $\theta \in \R^d$, the coordinates are not independent. The **covariance** between two random variables $X$ and $Y$ is defined as:

$$\tag{3.13} \begin{aligned}
    \Cov(X, Y) &= \E[(X - \E[X])(Y - \E[Y])] \\
    &= \E[XY] - \E[X]\E[Y] \\
    &= \E[X(Y - \E[Y])]
\end{aligned}$$

Properties:

$$\tag{3.14} \Cov(X, X) = \V(X)$$
$$\tag{3.15} \Cov(X, Y) = \Cov(Y, X)$$

If $X$ and $Y$ are independent, then $\Cov(X, Y) = 0$ and $\E[XY] = \E[X]\E[Y]$.

**Attention**: the converse is not true; if $\Cov(X, Y) = 0$, $X$ and $Y$ may be dependent.

If $\XX = (X^{(1)}, \ldots, X^{(d)})^T \in \R^d$, then the covariance matrix of the vector is given by:

$$\tag{3.16} \SSigma = \CCov(\XX) = \E[(\XX - \E[\XX])(\XX - \E[\XX])^T] \in \R^{d \times d}$$

Its terms are:

$$\SSigma_{i,j} = \Cov(X^{(i)}, X^{(j)})$$

Proprieties:

$$\tag{3.17} \CCov(A\XX + B) = \CCov(A\XX) = A\CCov(\XX)A^T$$

The **Multivariate Central Limit Theorem**, let $\XX_1, \ldots, \XX_n \in \R^d$ i.i.d copies of $\XX$ such that $\E[\XX] = \mmu$ and $\CCov(\XX) = \SSigma$. Then:

$$\tag{3.18} \sqrt{n}(\bar{\XX}_n - \mmu) \convd \Norm_d(\zz, \SSigma)$$

Equivalently,

$$\sqrt{n}\SSigma^{-1/2}(\bar{\XX}_n - \mmu) \convd \Norm_d(\zz, I_d)$$

The **Multivariate Delta Method**, let $(\TT_n)_n \in (\R^d)^\N$ be a sequence of random vectors such that

$$\sqrt{n}(\TT_n - \ttheta) \convd \Norm_d(\zz, \SSigma)$$

with $\ttheta \in \R^d, \SSigma \in \R^{d\times d}$. Let $\bm{g}: \R^d \rightarrow \R^k$ be continuously differentiable in $\ttheta$. Then:

$$\tag{3.19} \sqrt{n}(\bm{g}(\TT_n) - \bm{g}(\ttheta)) \convd \Norm_k(\zz, \nabla \bm{g}(\ttheta)^T\SSigma\nabla \bm{g}(\ttheta))$$

where

$$\nabla \bm{g}(\ttheta) = \left(\diff{g_j}{\theta_i}\right)_{i, j} = \left(\nabla g_1(\ttheta), \cdots,\nabla g_k(\ttheta)\right)$$

### Lecture 11: Fisher Information, Asymptotic Normality of MLE; Methods of Moments

Let $\statmodel$ be a statistical model with $\Theta \subseteq \R^d$, and $\XX$ an associated random variable. With

$$\ell(\ttheta) = \ln L_1(\XX, \ttheta)$$

The **Fisher Information** of the model is defined as:

$$\tag{3.20} I : \ttheta \mapsto \E[\nabla\ell(\ttheta)\nabla\ell(\ttheta)^T] - \E[\nabla\ell(\ttheta)]\E[\nabla\ell(\ttheta)]^T$$

Under some regularity conditions, we have that:

$$\tag{3.21} I(\ttheta) = - \E[\HH\ell(\ttheta)]$$\

If $\Theta \subseteq \R$, we have:

$$\tag{3.22} I(\theta) = \V[\ell'(\theta)] = -\E[\ell''(\theta)]$$

**Asymptotic normality of the MLE**. Let $\ttth \in \Theta$ be the true parameter. If we have the following:

* $\ttth$ is identifiable,
* $\forall \ttheta \in \Theta$, the support of $\P_\theta$ does not depend on $\theta$,
* $\ttth$ is not on the boundary of $\Theta$,
* $I(\ttheta)$ is invertible in a neighborhood of $\ttth$,
* A few more technical conditions,

then $\tthn^{MLE}$ satisfies:

$$\tag{3.23} \tthn^{MLE} \convp \ttth \quad [\P_{\ttth}]$$

$$\tag{3.24} \sqrt{n}(\tthn^{MLE} - \ttth) \convd \Norm_d(0, I(\ttth)^{-1}) \quad [\P_{\ttth}]$$

Let $\XX_1, …, \XX_n$ be an i.i.d sample associated with a statistical model $\statmodel$, with $E \subseteq \R$ and $\Theta \subseteq \R^d$. The **population moments** are:

$$\tag{3.25} \forall 1 \leq k \leq d, m_k(\ttheta) = \E_\ttheta[X^k]$$

And the **empirical moments are**:

$$\tag{3.26} \forall 1 \leq k \leq d, \hat{m}_k = \overline{X^k_n} = \frac1n\sum_{i=1}^nX_i^k$$

From the LLN, we have

$$\tag{3.27} (\hat{m}_1, \ldots, \hat{m}_d) \conv{\P / a.s.} (m_1(\ttheta), \ldots, m_d(\ttheta))$$

Let

$$\fun{M}{\Theta}{\R^d}{\ttheta}{M(\ttheta) = (m_1(\ttheta), \ldots, n_d(\ttheta))}$$

Then if $M$ is injective, $\ttheta = M^{-1}(m_1(\ttheta), \ldots, m_d(\ttheta))$

The **moment estimator of $\ttheta$**, if it exists, is defined as

$$\tag{3.28} \tthn^{MM} = M^{-1}(\hat m_1, \ldots, \hat m_d)$$

Let $\hat M = (\hat m_1, \ldots, \hat m_d)$. Let $\SSigma(\ttheta)  = \CCov_\ttheta(X, X^2, \ldots, X^d)$. Let us assume $M^{-1}$ is continuously differentiable at $M(\ttheta)$.

We can **generalize** the method of moments to any set of functions $g_1, \ldots, g_d : R \rightarrow \R$ well chosen, by defining $m_k(\ttheta) = \E_\ttheta[g_k(X)]$ and $\SSigma(\ttheta) = \CCov_\ttheta(g_1(X), \ldots, g_k(X))$.

The **generalized method of moments** yields, by applying the CLT and the Delta method:

$$\tag{3.29} \sqrt{n}(\tthn^{MM} - \theta) \convd\Norm_d(\zz, \Gamma(\ttheta))$$

where

$$\Gamma(\ttheta) = \left[\nabla M^{-1}(M(\ttheta))\right]^T\SSigma(\ttheta)\left[\nabla M^{-1}(M(\ttheta))\right]$$

The $MLE$ is more accurate than the $MM$, the $MLE$ still gives good results if the model is mis-specified, however the $MM$ is easier to compute and the $MLE$ can be intractable sometimes.

### Lecture 12: M-Estimation

Let $X_1, \ldots, X_n \iid \P$ on a sample space $E \subseteq \R^d$.

The goal is to estimate some parameter $\true\mmu$ associated with $\P$. We find a function $\rho : E \times \mathcal{M} \rightarrow \R$, where $\mathcal{M}$ is the parameter set for $\true\mmu$, such that:

$$\mathcal{Q}(\mmu) = \E[\rho(\XX, \mmu)]$$

verifies

$$\tag{3.30} \true\mu = \argmin{\mu \in \mathcal{M}}\mathcal{Q}(\mu)$$

For example:

* if $\rho(\xx, \mmu) = \|\xx - \mmu\|_2^2$, then $\true\mmu = \E[\XX]$.
* if $\rho(x, \mu) = |x - \mu|$, then $\true\mu = \Med{}(X)$.

Let $\alpha \in (0, 1)$. We define the **check functions**:

$$\fun{C_\alpha}{\R}{\R}{x}{\begin{cases}
    -(1-\alpha)x & \text{if } x < 0 \\
    \alpha x & \text{if } x \geq 0
\end{cases}}$$

If $\statmodel$ is a statistical model associated with the data, $\mathcal{M} = \Theta$, and $\rho = -\ell_1$ the negative log-likelihood, then

$$\tag{3.31}\true\mmu = \ttth$$

where $\P = \P_\ttth$. As such, the $MLE$ estimator is an M-estimator.

Let

$$J(\mmu) = \HH Q(\mmu)$$

under some regularity conditions,

$$J(\mmu) = \E\left[\diffd{\rho}{\mmu}{\mmu^T}(X, \mmu)\right]$$

Let

$$K(\mmu) = \CCov\left[\nabla_\mmu\rho(X, \mmu)\right]$$

In the case of the log-likelihood, we have $J(\ttheta) = K(\ttheta) = I(\ttheta)$.

Let $\true\mmu \in \mathcal{M}$ be the true parameter. If we have:

* $\true\mmu$ is the only minimizer of $\mathcal{Q}$,
* $J(\mmu)$ is invertible for all $\mmu \in \mathcal{M}$,
* A few more technical conditions,

then $\est{\mmu}$ satisfies:

$$\tag{3.32} \est{\mmu} \convp \true\mmu$$

$$\tag{3.33} \sqrt{n}(\est{\mmu} - \true\mmu) \convd \Norm_d(\zz, J^{-1}(\true\mmu)^TK(\true\mmu)J^{-1}(\true\mmu))$$

## Addendum A: Frequent distributions

### Part 1: Normal distribution

### Description [A.1]

Notation: $\mathcal{N}(\mu, \sigma^2)$

Parameters: $\mu \in \R, \sigma^2 > 0$

Support: $E = \R$

Probability density function (*PDF*):

$$\tag{A.1.1} f_{\mu, \sigma^2}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

The cumulative density function (*CDF*) is noted $\Phi(x)$.

### Properties [A.1]

Mean: $\E[Z] = \mu$

Variance: $\V[Z] = \sigma^2$

*Quantiles*:

$$\tag{A.1.2} q_\alpha = \Phi^{-1}(1 - \alpha)$$

$$\tag{A.1.3} \P[Z \leq q_\alpha] = 1 - \alpha$$

$$\tag{A.1.4} \P[|Z| \leq q_{\alpha/2}] = 1 - \alpha$$

$q_{0.1} \approx 1.281713$, $q_{0.05} \approx 1.64467$, $q_{0.025} \approx 1.95905$,
$q_{0.01} \approx 2.32409$, $q_{0.005} \approx 2.572466$.

*Fischer information*:

$$\tag{A.1.5} I(\mu, \sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}$$

*Log-likelihood*:

$$\tag{A.1.6} \ell_d(x_1, \ldots, x_d, \mu, \sigma^2) = -\frac{d}2\ln2\pi -\frac{d}2\ln\sigma^2 - \frac1{2\sigma^2}\sum_{i=1}^d(x_i - \mu)^2$$

### Moments [A.1]

$\E[Z] = \mu$

$\E[Z^2] = \mu^2 + \sigma^2$

$\E[Z^3] = \mu^3 + 3\mu\sigma^2$

$\E[Z^4] = \mu^4 + 6\mu^2\sigma^2 + 3\sigma^4$

$\E[Z^5] = \mu^5 + 10\mu^3\sigma^2 + 15\mu\sigma^4$

$\E[Z^6] = \mu^6 + 15\mu^4\sigma^2 + 45\mu^2\sigma^4 + 15\sigma^6$

## Part 2: Bernoulli distribution

### Description [A.2]

Notation: $\Ber(p)$

Parameters: $p \in (0, 1)$

Support: $E = \{0, 1\}$

Probability mass function (*PMF*):

$$\tag{A.2.1} f_p(k) = p^k(1-p)^{1-k} = \begin{cases} \displaystyle p & \text{if}\quad k = 1 \\ \displaystyle 1-p & \text{if}\quad k = 0\end{cases}$$

### Properties [A.2]

Mean: $\E[X] = p$

Variance: $\V[X] = p(1-p)$

Moments: $\E[X^k] = p$

Fischer information: $I(p) = \frac{1}{p(1-p)}$

*Log-likelihood*:

$$\tag{A.2.2} \ell_d(x_1, \ldots, x_d, p) = \ln p\sum_{i=1}^dx_i + \ln(1-p)\left(d - \sum_{i=1}^dx_i\right)$$

## Part 3: Binomial distribution

### Description [A.3]

Notation: $\mathcal{B}(n, p)$

Parameters: $n \in \N, p \in (0, 1)$

Support: $E = [\![0, n]\!]$

Probability mass function (*PMF*):

$$\tag{A.3.1} f_{n, p}(k) = \binom{n}kp^k(1-p)^{n-k}$$

### Properties [A.3]

Mean: $\E[X] = np$

Variance: $\V[X] = np(1-p)$

Fischer information: $I_n(p) = \frac{n}{p(1-p)}$ (for fixed $n$)

*Log-likelihood*:

$$\tag{A.3.2} \ell_d(x_1, \ldots, x_d, n, p) = \sum_{i=1}^d\ln\binom{n}{x_i} + \ln p \sum_{i=1}^dx_i + \ln(1-p)\left(nd - \sum_{i=1}^dx_i\right)$$

## Part 4: Poisson distribution

### Description [A.4]

Notation: $\mathcal{Poi}(\lambda)$

Parameters: $\lambda > 0$

Support: $E = \N$

Probability mass function (*PMF*):

$$\tag{A.4.1} f_\lambda(k) = \frac{\lambda^ke^{-\lambda}}{k!}$$

### Properties [A.4]

Mean: $\E[X] = \lambda$

Variance: $\V[X] = \lambda$

Fischer information: $I(\lambda) = \frac1\lambda$

*Log-likelihood*:

$$\tag{A.4.2} \ell_d(x_1, \ldots, x_d, \lambda) = \ln\lambda\sum_{i=1}^dx_i - d\lambda - \sum_{i=1}^d\ln(x_i!)$$

## Part 5: Uniform distribution

### Description [A.5]

Notation: $\mathcal{U}(a, b)$

Parameters: $a, b \in \R, a < b$ (usually, $a = 0$)

Support: $E = [a, b] \subset \R$

Probability density function (*PDF*):

$$\tag{A.5.1} f_{a, b}(x) = \frac1{b-a}\one\{a \leq x \leq b\}$$

Cumulative density function (*CDF*):

$$\tag{A.5.2} F_{a, b}(x) = \begin{cases}\displaystyle 0 & \text{if}\quad x < a \\ \displaystyle \frac{x-a}{b-a} & \text{if}\quad a \leq x \leq b \\ \displaystyle 1 & \text{if}\quad x > b \end{cases}$$

### Properties [A.5]

Mean: $\E[X] = \frac{a+b}2$

Variance: $\V[X] = \frac1{12}(b-a)^2$

Moments: $\E[X^k] = \frac{1}{n+1}\sum_{i=0}^ka^ib^{k-i}$

*Log-likelihood*:

$$\tag{A.5.3} \ell_d(x_1, \ldots, x_d, a, b) = d\ln\frac1{b-a} + \ln\one\left\{a \leq \min_{1 \leq i \leq d}x_i\right\} + \ln\one\left\{\max_{1 \leq i \leq d}x_i \leq b\right\}$$

## Part 6: Exponential distribution

### Description [A.6]

Notation: $\mathcal{Exp}(\lambda)$

Parameters: $\lambda > 0$

Support: $E = [0, +\infty)$

Probability density function (*PDF*):

$$\tag{A.6.1} f_\lambda(x) = \lambda \exp(-\lambda x)\one\{x \geq 0\}$$

Cumulative density function (*CDF*):

$$\tag{A.6.2} F_\lambda(x) = (1 - \exp(-\lambda x))\one\{x \geq 0\}$$

### Properties [A.6]

Mean: $\E[X] = \frac1\lambda$

Variance: $\V[X] = \frac1{\lambda^2}$

Moments: $\E[X^k] = \frac{k!}{\lambda^k}$

Fischer Information: $I(\lambda) = \frac1{\lambda^2}$

*Log-likelihood*:

$$\tag{A.6.3} \ell_d(x_1, \ldots, x_d, \lambda) = d\ln\lambda -\lambda\sum_{i=1}^dx_i+\ln\one\left\{0 \leq \min_{1\leq i \leq d}x_i\right\}$$

Memorylessness: $\P[X > s+t | X > s] = \P[X > t]$

## Part 7: Multivariate normal distribution

### Description [A.7]

Notation: $\mathcal{N}_d(\mmu, \SSigma)$

Parameters: $\mmu \in \R^d, \SSigma\in \R^{d\times d}, \SSigma \succ 0$

Support: $E = \R^d$

Probability density function (*PDF*):

$$\tag{A.7.1} f_{\mmu, \SSigma}(\xx) = \frac{1}{\sqrt{(2\pi)^d|\SSigma|}}\exp\left(-\frac12(\xx - \mmu)^T\SSigma^{-1}(\xx - \mmu)\right)$$

### Properties [A.7]

Mean: $\E[\XX] = \mmu$

Variance: $\V[\XX] = \SSigma$

## Part 8: Cauchy

### Description [A.8]

Notation: $\mathcal{Cau}(x_0, \gamma)$

Parameters: $x_0 \in \R, \gamma > 0$

Support: $E = \R$

Probability density function (*PDF*):

$$\tag{A.8.1} f_{x_0, \gamma}(x) = \frac{1}{\pi\gamma}\frac{\gamma^2}{\gamma^2 + (x-x_0)^2}$$

Cumulative density function (*CDF*):

$$\tag{A.8.2} F_{x_0, \gamma}(x) = \frac12 + \frac1{\pi}\arctan\frac{x-x_0}{\gamma}$$

### Properties [A.8]

Mean: undefined

Variance: undefined

Moments: undefined

## Part 9: Laplace

### Description [A.9]

Notation: $\mathcal{Lapl}(\mu, b)$

Parameters: $\mu \in \R, b > 0$

Support: $E = \R$

Probability density function (*PDF*):

$$\tag{A.9.1} f_{\mu, b}(x) = \frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)$$

Cumulative density function (*CDF*):

$$\tag{A.9.2} F_{\mu, b}(x) = \begin{cases} \displaystyle \frac12\exp\left(\frac{x-\mu}{b}\right) & \text{if}\quad x \leq \mu \\ \displaystyle 1 - \frac12\exp\left(\frac{\mu - x}{b}\right) & \text{if}\quad x > \mu \end{cases}$$

### Properties [A.9]

Mean: $\E[X] = \mu$

Variance: $\V[X] = 2b^2$

*Log-likelihood*:

$$\tag{A.9.3} \ell_d(x_1, \ldots, x_d, \mu, b) = -d\ln2 -d\ln b - \frac1b\sum_{i=1}^d|x_i - \mu|$$

*Maximum Likelihood Estimators*:

$$\hat{\mu} = \Med{1\leq i \leq d}X_i, \hat{b} = \frac{1}d\sum_{i=1}^d|X_i - \hat{\mu}|$$