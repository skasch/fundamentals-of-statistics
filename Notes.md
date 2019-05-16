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

Let $\Xton$ be **independent and identically distributed** (*i.i.d*)
**random variables** (*r.v.*). Let $\mu = \E[X]$ the expectation and
$\sigma^2 = \V[X]$ the variance of these random variables.

We define the **sample average** of our $X_i$ as:

$$\bar{X}_n = \frac1n \sum_{i=1}^nX_i$$

**Law of Large Numbers** (*LLN*):

$$\tag{1.1} \bar{X}_n \conv{\P, a.s.}\mu$$

Where $\P$ designs convergence in probability and $a.s.$ convergence
almost surely.

**Central Limit Theorem** (*CLT*):

$$\tag{1.2} \sqrt{n}\frac{\bar{X}_n - \mu}{\sigma} \convd \Norm(0, 1)$$

Equivalently, we also have:

$$\sqrt{n}(\bar{X}_n - \mu) \convd \Norm(0, \sigma^2)$$

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

$$T_n \conv{a.s./\P} T, \quad U_n \conv{a.s/\P} U$$

then:

$$\tag{1.9} T_n + U_n \conv{a.s/\P} T + U, \quad T_nU_n \conv{a.s/\P} TU$$

if in addition, $U \neq 0$ almost surely, then

$$\tag{1.10} \frac{T_n}{U_n} \conv{a.s/\P} \frac{T}U$$

**Slutsky's Theorem**. Let $u \in \R$. If

$$T_n \convd T, \quad U_n \convp u$$

then:

$$\tag{1.11} T_n + U_n \convd T + u, \quad T_nU_n \convd uT_n$$

if in addition, $u \neq 0$, then

$$\tag{1.12} \frac{T_n}{U_n} \convd \frac1uT$$

**Continuous mapping theorem**. Let $f \in \mathcal{C}(\R)$. If

$$T_n \conv{a.s./\P/(d)}T$$

then:

$$\tag{1.13} f(T_n) \conv{a.s./\P/(d)} f(T)$$

## Unit 2: Foundation of Inference

### Lecture 3: Parametric Statistical Models

Trinity of statistical inference: *estimation*, *confidence intervals* and *hypothesis testing*.

Given $\Xton \iid \P$, the goal is to learn the distribution $\P$.

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

An estimator $\ethn$ of $\theta$ is **weakly (resp. strongly) consistent** if:

$$\tag{2.4} \ethn \conv{\P\;(resp.\;a.s.)} \theta \quad [\P_\theta]$$

An estimator $\ethn$ of $\theta$ is **asymptotically normal** if:

$$\tag{2.5} \sqrt{n}(\ethn - \theta) \convd \Norm(0, \sigma^2)$$

The quantity $\sigma^2$ is called the **asymptotic variance** of $\ethn$, noted

$$\tag{2.6} V(\ethn) = \sigma^2$$

The **bias** of an estimator $\ethn$ of $\theta$ is defined as:

$$\tag{2.7} \Bias(\ethn) = \E[\ethn] - \theta$$

If $\Bias(\ethn) = 0$, then we say that $\ethn$ is **unbiased**.

The **quadratic risk** (or **risk**) of an estimator $\ethn$ is

$$\tag{2.8} R(\ethn) = \E\left[(\ethn - \theta)^2\right] = \V[\ethn] + \Bias(\ethn)^2$$

Let $\statmodel$ be a statistical model with $\Theta \subseteq \R$ associated with observations $\Xton$, and $\alpha \in (0, 1)$.

A **confidence interval (*C.I.*) of level $1 - \alpha$** is any random interval $\mathcal{I}$ independent from $\theta$ such as:

$$\tag{2.9} \forall \theta \in \Theta, \P_\theta[\mathcal{I} \ni \theta] \geq 1 - \alpha$$

A **C.I. of asymptotic level $1 - \alpha$** is any random interval $\mathcal{I}_n$ independent from $\theta$ such as:

$$\tag{2.10} \forall \theta \in \Theta, \limn\P_\theta[\mathcal{I}_n \ni \theta] \geq 1 - \alpha$$

*Example*: $\Xton \iid \Ber(p)$. Our estimator is $\estn{p} = \bar{X}_n$. From the CLT:

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

**Plug-in bounds**; as $\estn{p} \convas p$, we can apply Slutsky's theorem:

$$\sqrt{n}\frac{\bar{X}_n-p}{\sqrt{\estn{p}(1 -\estn{p})}} \convd \Norm(0, 1) \implies \mathcal{I}_{plug} = \bar{X}_n + q_{\alpha/2}\left[-\sqrt{\frac{\estn{p}(1-\estn{p})}{n}}, \sqrt{\frac{\estn{p}(1-\estn{p})}{n}}\right]$$

All three intervals verify:

$$\limn \P[\mathcal{I} \ni p] \geq 1 - \alpha$$

If $\mathcal{I}$ is a confidence interval of level $\alpha$, it is also a confidence interval of level $\beta$ for all $\beta \geq \alpha$.

A confidence interval of level $\alpha$ means that if we repeat the experience multiple times, then the real parameter will be in the confidence interval with frequency $\alpha$. It **does not mean** that the real parameter is in the interval with a certain confidence (as the real parameter is deterministic).

### Lecture 5: Delta Method and Confidence Intervals

The **Delta Method**. Let $(X_n)_n$ be a sequence of r.v. such as

$$\sqrt{n}(X_n - \theta) \convd \Norm(0, \sigma^2)$$

for some $\theta \in \R, \sigma^2 > 0$. Let $g: \R \rightarrow \R$ continuously differentiable at $\theta$. Then $(g(X_n))_n$ is asymptotically normal around $g(\theta)$ and:

$$\tag{2.11}\sqrt{n}(g(X_n) - g(\theta)) \convd \Norm(0, g'(\theta)^2\sigma^2)$$

### Lecture 6: Introduction to Hypothesis Testing, and Type 1 and Type 2 Errors

Let us consider a sample $\Xton$ of i.i.d r.v. and a statistical model $\statmodel$. Let us consider $\Theta_0 \amalg \Theta_1 = \Theta$.

* $H_0: \theta \in \Theta_0$ is called the **null hypothesis**,
* $H_1: \theta \in \Theta_1$ is called the **alternative hypothesis**.

We are testing $H_0$ against $H_1$ to decide whether to **reject $H_0$** by looking for evidence against $H_0$.

$H_0$ and $H_1$ do not play symmetric roles in hypothesis testing: $H_0$ is the default hypothesis when we lack evidence.

A **test** is a statistic $\psi : E^n \rightarrow \{0, 1\}$ such as:

* If $\psi(\Xton) = 0$ then $H_0$ is not rejected,
* If $\psi(\Xton) = 1$ then $H_0$ is rejected and $H_1$ is accepted.

The **rejection region** of a test $\psi$ is:

$$\tag{2.12} R_\psi = \{(x_1, \ldots, x_n) \in E^n, \psi(x_1, \ldots, x_n) = 1\}$$

The **type 1** error of a test $\psi$ corresponds to rejecting $H_0$ when it is actually true:

$$\tag{2.13} \fun{\alpha_\psi}{\Theta_0}{[0, 1]}{\theta}{\P_\theta[\psi(\Xton) = 1]}$$

The **type 2** error of a test $\psi$ corresponds to not rejecting $H_0$ when $H_1$ is actually true:

$$\tag{2.14} \fun{\beta_\psi}{\Theta_1}{[0, 1]}{\theta}{\P_\theta[\psi(\Xton) = 0]}$$

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

Example: Let $X_1, \ldots, X_n \iid \mathcal{D}(\mu)$ where $\mu$ and $\sigma^2$ are the expectation and variance of $\mathcal{D}(\mu)$. Let's consider $\mu_0$.

We want to test $H_0: \mu = \mu_0$ against $H_1: \mu \neq \mu_0$ with asymptotic level $\alpha$.

$$\tag{2.18} T_n = \sqrt{n}\frac{\estn{\mu} - \mu_0}{\sigma} \convd \Norm(0, 1)$$

Therefore, if $H_0$ is true, then

$$\P[|T_n| > q_{\alpha/2}] \conv{} \alpha$$

Let $\psi_\alpha = \one\{|T_n| > q_{\alpha/2}\}$.

Now, we want to test $H_0: \mu \leq \mu_0$ and $H_1: \mu > \mu_0$ with asymptotic level $\alpha$. Which value of $\mu \in \Theta_0$ should we consider?

Type 1 error is the function $\mu \mapsto \P_\mu[\psi(\Xton) = 1]$. To control the level, we need the $\mu$ that maximises this expression over $\Theta_0$: clearly, this happens for $\mu = \mu_0$. Therefore, if $H_0$ is true, then

$$\P_{\mu_0}[T_n > q_\alpha] \conv{} \alpha$$

If $H_1: \mu < \mu_0$, then we want $\P_{\mu_0}[T_n < -q_\alpha]$.

The **(asymptotic) p-value** of a test $\psi_\alpha$ is the smallest (asymptotic) level $\alpha$ at which $\psi_\alpha$ rejects $H_0$. The p-value is random and depends on the sample.

The *golden rule*: $H_0$ is rejected by $\psi_\alpha$ at any asymptotic level $\alpha \geq \pval(\Xton)$.

## Unit 3: Methods of estimation

### Lecture 8: Distance measures between distributions

Let $\statmodel$ be a statistical model, associated with a sample of i.i.d r.v. $\Xton$. We assume the model is well specified, i.e. $\exists \trth \in \Theta, X \sim \P_\trth$. $\trth$ is called the **true parameter**.

We want to find an estimator $\ethn$ such that $\P_{\ethn}$ is close to $\P_\trth$.

The **total variation distance** between two probability measures $\P_1$ and $\P_2$ is defined as:

$$\tag{3.1} \TV(\P_1, \P_2) = \max_{A \subset E}|\P_1(A) - \P_2(A)|$$

If $E$ is discrete, and $p_1, p_2$ are the PMF associated with $\P_1$ and $\P_2$:

$$\tag{3.2} \TV(\P_1, \P_2) = \frac12\sum_{x\in E}|p_1(x) - p_2(x)|$$

If $E$ is continuous, and $f_1, f_2$ are the PDF associated with $\P_1$ and $\P_2$:

$$\tag{3.3} \TV(\P_1, \P_2) = \frac12\int_E|f_1(x) - f_2(x)|dx$$

Proprieties of the total variation distance:

* The $\TV$ is **symmetric**: $\forall \P_1, \P_2, \TV(\P_1, \P_2) = \TV(\P_2, \P_1)$,
* The $\TV$ is **positive**: $\forall \P_1, \P_2, 0 \leq \TV(\P_1, \P_2) \leq 1$,
* The $\TV$ is **definite**: $\forall \P_1, \P_2, \TV(\P_1, \P_2) = 0 \implies \P_1 = \P_2$ almost everywhere,
* The $\TV$ verifies the **triangle inequality**: $\forall \P_1, \P_2, \P_3, \TV(\P_1, \P_3) \leq \TV(\P_1, \P_2) + \TV(\P_2, \P_3)$.

Therefore, $\TV$ is a distance between probability distributions.

Problem 1: $\TV$ cannot compare discrete and continuous distributions, which means we cannot trust it to give a reliable estimation of distance between unrelated distributions.

Problem 2: We cannot build an estimator $\theta \mapsto \widehat{\TV}(\P_\theta, \P_\trth)$ as we do not know $\trth$.

Hence we need another "distance" between distributions. The **Kullback-Leibler (*KL*) divergence** between two probability distributions $\P_1, \P_2$ is defined as:

$$\tag{3.4} \KL(\P_1, \P_2)= \begin{cases}\displaystyle\sum_{x\in E}p_1(x)\ln\frac{p_1(x)}{p_2(x)} & \text{if }E\text{ is discrete} \\ \displaystyle \int_Ef_1(x)\ln\frac{f_1(x)}{f_2(x)}dx & \text{if }E\text{ is continuous}\end{cases}$$

Proprieties of the $\KL$ divergence:

* In general, $\KL(\P_1, \P_2) \neq \KL(\P_2, \P_1)$
* The $\KL$ is **positive**: $\forall \P_1, \P_2, \KL(\P_1, \P_2) \geq 0$
* The $\KL$ is **definite**: $\forall \P_1, \P_2, \KL(\P_1, \P_2) = 0 \implies \P_1 = \P_2$
* In general, $\KL(\P_1, \P_3) \not\leq \KL(\P_1, \P_2) + \KL(\P_2, \P_3)$

$\KL$ is not a distance, it's a divergence. But we still have that $\trth$ is the only minimizer of $\theta \mapsto \KL(\P_\trth, \P_\theta)$.

### Lecture 9: Maximum Likelihood Estimation

$$\begin{aligned}\KL(\P_\trth, \P_\theta) &= \E_\trth\left[\ln\frac{f_\trth(X)}{f_\theta(X)}\right] \\ &= \E_\trth[\ln f_\trth(X)] - \E_\trth[\ln f_\theta(X)]\end{aligned}$$

Therefore, by the LLN the $\KL$ divergence can be estimated by taking the average of $\ln f_\theta(X_i)$:

$$\widehat{\KL}(\P_\trth, \P_\theta) = \text{constant} - \frac1n\sum_{i=1}^n\ln f_\theta(X_i)$$

Therefore:

$$\begin{aligned}
    \argmin{\theta \in \Theta}\widehat{\KL}(\P_\trth, \P_\theta) &= \argmin{\theta \in \Theta} -\frac1n\sum_{i=1}^n\ln f_\theta(X_i) \\
    &= \argmax{\theta \in \Theta}\ln\prod_{i=1}^nf_\theta(X_i) \\
    &= \argmax{\theta \in \Theta}\prod_{i=1}^nf_\theta(X_i)
\end{aligned}$$

This is the **maximum likelihood principle**.

Let $\statmodel$ be a statistical model associated with  a sample of i.i.d r.v. $\Xton$. The **likelihood** of the model is the function:

$$\tag{3.5}\fun{L_n}{E^n \times \Theta}{[0, 1]}{(\Xton, \theta)}{\begin{cases}
    \displaystyle \prod_{i=1}^np_\theta(x_i) & \text{if }E\text{ is discrete} \\
    \displaystyle \prod_{i=1}^nf_\theta(x_i) & \text{if }E\text{ is continous}
\end{cases}}$$

The **maximum likelihood estimator** of $\theta$ is defined as:

$$\tag{3.6} \ethn^{MLE} = \argmax{\theta \in \Theta} L_n(\Xton, \theta)$$

In practice, we often use the **log-likelihood** $\ell_n(\Xton, \theta) = \ln L_n(\Xton, \theta)$.

A function twice differentiable $h : \Theta \subseteq \R \rightarrow \R$ is **concave** (resp. **strictly**) if

$$\tag{3.7} \forall \theta \in \Theta, h''(\theta) \leq 0 \quad\text{(resp. }< 0\text{)}$$

$h$ is **convex** (resp. **strictly**) if $-h$ is concave (resp. strictly).

A con-**v**-ex function is shaped like a **v**.

Let $h: \Theta \subseteq \R^d \rightarrow \R$ be a multivariate function. The **gradient** vector of $h$ is:

$$\tag{3.8} \nabla h(\tth) = \begin{pmatrix}\displaystyle \diff{h}{\theta_1}(\tth) \\ \displaystyle \vdots \\ \displaystyle \diff{h}{\theta_d}(\tth)\end{pmatrix}$$

The **Hessian matrix** of $h$ is:

$$\tag{3.9} \HH h(\tth) = \begin{pmatrix}
    \displaystyle \diffk{h}{\theta_1}{2}(\tth) & \cdots & \displaystyle \diffd{h}{\theta_1}{\theta_d}(\tth) \\
    \vdots & \ddots & \vdots \\
    \displaystyle \diffd{h}{\theta_d}{\theta_1}(\tth) & \cdots & \displaystyle \diffk{h}{\theta_d}{2}(\tth) \\
\end{pmatrix} \in \R^{d\times d}$$

Therefore, $h$ is **concave** (resp. **strictly**) if

$$\tag{3.10} \forall \xx \in \R^d \setminus \{\zz\}, \tth \in \Theta, \xx^T\HH h(\tth)\xx \leq 0 \quad\text{(resp. }< 0\text{)}$$

If $h$ is strictly concave and has a maximum, then this maximum is unique and is reached for

$$\tag{3.11} \begin{cases}
    h'(\theta) = 0 & \text{if }\Theta \subseteq \R \\
    \nabla h(\tth) = \zz & \text{if }\Theta \subseteq \R^d
\end{cases}$$

### Lecture 10: Consistency of MLE, Covariance Matrices, and Multivariate Statistics

Under mild regularity conditions, we have

$$\tag{3.12} \ethn^{MLE} \convp \trth$$

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

If $\XX = (X^{(1)}, \ldots, X^{(d)})^\top \in \R^d$, then the covariance matrix of the vector is given by:

$$\tag{3.16} \SSigma = \CCov(\XX) = \E[(\XX - \E[\XX])(\XX - \E[\XX])^\top] \in \R^{d \times d}$$

Its terms are:

$$\SSigma_{i,j} = \Cov(X^{(i)}, X^{(j)})$$

Proprieties:

$$\tag{3.17} \CCov(A\XX + B) = \CCov(A\XX) = A\CCov(\XX)A^\top$$

The **Multivariate Central Limit Theorem**, let $\XX_1, \ldots, \XX_n \in \R^d$ i.i.d copies of $\XX$ such that $\E[\XX] = \mmu$ and $\CCov(\XX) = \SSigma$. Then:

$$\tag{3.18} \sqrt{n}(\bar{\XX}_n - \mmu) \convd \Norm_d(\zz, \SSigma)$$

Equivalently,

$$\sqrt{n}\SSigma^{-1/2}(\bar{\XX}_n - \mmu) \convd \Norm_d(\zz, I_d)$$

The **Multivariate Delta Method**, let $(\TT_n)_n \in (\R^d)^\N$ be a sequence of random vectors such that

$$\sqrt{n}(\TT_n - \tth) \convd \Norm_d(\zz, \SSigma)$$

with $\tth \in \R^d, \SSigma \in \R^{d\times d}$. Let $\gg: \R^d \rightarrow \R^k$ be continuously differentiable in $\tth$. Then:

$$\tag{3.19} \sqrt{n}(\gg(\TT_n) - \gg(\tth)) \convd \Norm_k(\zz, \nabla \gg(\tth)^\top\SSigma\nabla \gg(\tth))$$

where

$$\nabla \gg(\tth) = \left(\diff{g_j}{\theta_i}\right)_{i, j} = \left(\nabla g_1(\tth), \cdots,\nabla g_k(\tth)\right) = \bf{J}_\gg^\top(\tth)$$

where $\bf{J}_\gg$ is the Jacobian of $\gg$.

### Lecture 11: Fisher Information, Asymptotic Normality of MLE; Methods of Moments

Let $\statmodel$ be a statistical model with $\Theta \subseteq \R^d$, and $\XX$ an associated random variable, and

$$\ell(\tth) = \ln L_1(\XX, \tth)$$

The **Fisher Information** of the model is defined as:

$$\tag{3.20} I : \tth \mapsto \E[\nabla\ell(\tth)\nabla\ell(\tth)^\top] - \E[\nabla\ell(\tth)]\E[\nabla\ell(\tth)]^\top$$

Under some regularity conditions, we have that:

$$\tag{3.21} I(\tth) = - \E[\HH\ell(\tth)]$$\

If $\Theta \subseteq \R$, we have:

$$\tag{3.22} I(\theta) = \V[\ell'(\theta)] = -\E[\ell''(\theta)]$$

**Asymptotic normality of the MLE**. Let $\trtth \in \Theta$ be the true parameter. If we have the following:

* $\trtth$ is identifiable,
* $\forall \tth \in \Theta$, the support of $\P_\theta$ does not depend on $\theta$,
* $\trtth$ is not on the boundary of $\Theta$,
* $I(\tth)$ is invertible in a neighborhood of $\trtth$,
* A few more technical conditions,

then $\etthn^{MLE}$ satisfies, with regards to $\P_{\trtth}$:

$$\tag{3.23} \etthn^{MLE} \convp \trtth, \quad \sqrt{n}(\etthn^{MLE} - \trtth) \convd \Norm_d(0, I(\trtth)^{-1})$$

Let $\Xton$ be an i.i.d sample associated with a statistical model $\statmodel$, with $E \subseteq \R$ and $\Theta \subseteq \R^d$. The **population moments** are:

$$\tag{3.24} \forall 1 \leq k \leq d, m_k(\tth) = \E_\tth[X^k]$$

And the **empirical moments are**:

$$\tag{3.25} \forall 1 \leq k \leq d, \hat{m}_k = \overline{X^k_n} = \frac1n\sum_{i=1}^nX_i^k$$

From the LLN, we have

$$\tag{3.26} (\hat{m}_1, \ldots, \hat{m}_d) \conv{\P / a.s.} (m_1(\tth), \ldots, m_d(\tth))$$

Let

$$\fun{M}{\Theta}{\R^d}{\tth}{M(\tth) = (m_1(\tth), \ldots, m_d(\tth))}$$

Then if $M$ is injective, $\tth = M^{-1}(m_1(\tth), \ldots, m_d(\tth))$

The **moment estimator of $\tth$**, if it exists, is defined as

$$\tag{3.27} \etthn^{MM} = M^{-1}(\hat m_1, \ldots, \hat m_d)$$

Let $\hat M = (\hat m_1, \ldots, \hat m_d)$. Let $\SSigma(\tth)  = \CCov_\tth(X, X^2, \ldots, X^d)$. Let us assume $M^{-1}$ is continuously differentiable at $M(\tth)$.

We can **generalize** the method of moments to any set of functions $g_1, \ldots, g_d : \R \rightarrow \R$ well chosen, by defining $m_k(\tth) = \E_\tth[g_k(X)]$ and $\SSigma(\tth) = \CCov_\tth(g_1(X), \ldots, g_k(X))$.

The **generalized method of moments** yields, by applying the CLT and the Delta method:

$$\tag{3.28} \sqrt{n}(\etthn^{MM} - \tth) \convd\Norm_d(\zz, \Gamma(\tth))$$

where

$$\Gamma(\tth) = \left[\nabla M^{-1}(M(\tth))\right]^\top\SSigma(\tth)\left[\nabla M^{-1}(M(\tth))\right]$$

The $MLE$ is more accurate than the $MM$, the $MLE$ still gives good results if the model is mis-specified, however the $MM$ is easier to compute and the $MLE$ can be intractable sometimes.

### Lecture 12: M-Estimation

Let $\Xton \iid \P$ on a sample space $E \subseteq \R^d$.

The goal is to estimate some parameter $\true\mmu$ associated with $\P$. We find a function $\rho : E \times \mathcal{M} \rightarrow \R$, where $\mathcal{M}$ is the parameter set for $\true\mmu$, such that:

$$\mathcal{Q}(\mmu) = \E[\rho(\XX, \mmu)]$$

verifies

$$\tag{3.29} \true\mmu = \argmin{\mmu \in \mathcal{M}}\mathcal{Q}(\mmu)$$

For example:

* if $\rho(\xx, \mmu) = \|\xx - \mmu\|_2^2$, then $\true\mmu = \E[\XX]$.
* if $\rho(x, \mu) = |x - \mu|$, then $\true\mu = \Med{}(X)$.

Let $\alpha \in (0, 1)$. We define the **check functions**:

$$\fun{C_\alpha}{\R}{\R}{x}{\begin{cases}
    -(1-\alpha)x & \text{if } x < 0 \\
    \alpha x & \text{if } x \geq 0
\end{cases}}$$

If $\statmodel$ is a statistical model associated with the data, $\mathcal{M} = \Theta$, and $\rho = -\ell_1$ the negative log-likelihood, then

$$\tag{3.30}\true\mmu = \trtth$$

where $\P = \P_\trtth$. As such, the $MLE$ estimator is an M-estimator.

Let

$$J(\mmu) = \HH \mathcal{Q}(\mmu)$$

under some regularity conditions,

$$J(\mmu) = \E\left[\diffd{\rho}{\mmu}{\mmu^\top}(X, \mmu)\right]$$

Let

$$K(\mmu) = \CCov\left[\nabla_\mmu\rho(X, \mmu)\right]$$

In the case of the log-likelihood, we have $J(\tth) = K(\tth) = I(\tth)$.

Let $\true\mmu \in \mathcal{M}$ be the true parameter. If we have:

* $\true\mmu$ is the only minimizer of $\mathcal{Q}$,
* $J(\mmu)$ is invertible for all $\mmu \in \mathcal{M}$,
* A few more technical conditions,

then if we define:

$$\mathcal{Q}_n(\mmu) = \frac1n\sum_{i=1}^n\rho(\XX_i, \mmu), \quad \estn{\mmu} = \argmin{\mmu \in \mathcal{M}}\mathcal{Q}_n(\mmu)$$

the sample minimizer of $\mathcal{Q}$, then $\estn{\mmu}$ satisfies:

$$\tag{3.31} \estn{\mmu} \convp \true\mmu, \quad \sqrt{n}(\estn{\mmu} - \true\mmu) \convd \Norm_d(\zz, J^{-1}(\true\mmu)^\top K(\true\mmu)J^{-1}(\true\mmu))$$

## Unit 4: Hypothesis testing

### Lecture 13: Chi Squared Distribution, T-Test

The **chi-squared ($\chi^2$) distribution**: let $d \in \N^*$. the $\chi^2_d$ distribution with *$d$ degrees of freedom* is the law of the random variable

$$\tag{4.1} Z_1^2 + \ldots + Z_d^2 \sim \chi^2_d$$

where $Z_1, \ldots, Z_d \iid \Norm(0, 1)$.

* If $Z \sim \Norm_d(0, I_d)$, then $\|Z\|_2^2 \sim \chi^2_d$.
* $\chi^2_2 = \Exp(1/2)$.

*Proprieties*: let $V \sim \chi^2_d$.

$$\tag{4.2} \E[V] = d, \quad \V[V] = 2d$$

**Cochran's Theorem**: let $\Xton \iid \Norm(\mu, \sigma^2)$. Let

$$S_n = \frac1n \sum_{i=1}^n(X_i - \bar{X}_n)^2 = \frac1n \sum_{i=1}^nX_i^2 - \bar{X}_n^2$$

be the sample variance. Then:

$$\tag{4.3}\forall n \in \N, \bar{X}_n \perp S_n \quad \text{and} \quad \frac{nS_n}{\sigma^2} \sim \chi^2_{n-1}$$

We often prefer the unbiased sample variance:

$$\tilde{S}_n = \frac1{n-1}\sum_{i=1}^n(X_i - \bar{X}_n)^2 = \frac{n}{n-1}S_n$$

which verifies $\E[\tilde{S}_n] = \sigma^2$. In this case, we have instead $\frac{(n-1)\tilde{S}_n}{\sigma^2} \sim \chi^2_{n-1}$.

**Student's T Distribution**: let $d \in \N^*$. The Student's T Distribution $t_d$ with $d$ degrees of freedom is the law of the random variable

$$\tag{4.4} \frac{Z}{\displaystyle \sqrt{\frac{V}d}} \sim t_d$$

where $Z \sim \Norm(0, 1), V \sim \chi^2_d$ and $Z \perp V$.

**Student's T test (one sample, two-sided)**: let $\Xton \iid \Norm(\mu, \sigma^2)$. We want to test $H_0: \mu = \mu_0 = 0$ against $H_1: \mu \neq 0$. We define the test statistic as:

$$\tag{4.5}T_n = \frac{\bar{X}_n}{\displaystyle \sqrt{\frac{\tilde{S}_n}n}}
= \frac{\displaystyle \sqrt{n}\frac{\bar{X}_n - \mu_0}{\sigma}}{\displaystyle \sqrt{\frac{\tilde{S}_n}{\sigma^2}}}$$

Since $\sqrt{n}\frac{\bar{X}_n - \mu_0}{\sigma} \sim \Norm(0, 1)$ under $H_0$ and $\frac{\tilde{S}_n}{\sigma^2} \sim \frac1{n-1}\chi^2_{n-1}$ are independent by Cochran's Theorem,

$$T_n \sim t_{n-1}$$

The *non-asymptotic* Student's test is therefore written, at level $\alpha$:

$$\psi_\alpha = \one\{|T_n| > q_{\alpha/2}(t_{n-1})\}$$

**Student's T test (one sample, one-sided)**: if instead we have $H_0: \mu = \mu_0 = 0$ and $H_1: \mu > 0$, the test is written, at level $\alpha$:

$$\psi_\alpha = \one\{T_n > q_\alpha(t_{n-1})\}$$

**Student's T test (two samples, two-sided)**: let $\Xton \iid \Norm(\mu_X, \sigma^2_X)$ and $Y_1, \ldots, Y_m \iid \Norm(\mu_Y, \sigma^2_Y)$. We want to test $H_0: \mu_X = \mu_Y$ against $H_1: \mu_X \neq \mu_Y$. The test statistic is written:

$$\tag{4.6} T_{n, m} = \frac{\bar{X}_n - \bar{Y}_m}{\displaystyle \sqrt{\frac{\est{\sigma^2}_X}n + \frac{\est{\sigma^2}_Y}m}}$$

**Welch-Satterthwaite formula**: $T_{n, m} \sim t_N$ with

$$\tag{4.7} N = \frac{\displaystyle \left(\frac{\est{\sigma^2}_X}n + \frac{\est{\sigma^2}_Y}m\right)^2}{\displaystyle \frac{\est{\sigma^2}^2_X}{n^2(n-1)} + \frac{\est{\sigma^2}^2_Y}{m^2(m-1)}} \geq \min(n, m)$$

$N$ should always be rounded down.

**Pros and cons of the T test**:

* Non-asymptotic and can be run on small samples, and can also be seemlessly applied for large samples.
* The samples must be Gaussian.

### Lecture 14: Wald's Test, Likelihood Ratio Test, and Implicit Hypothesis Testing

**Wald's Test**: let $\Xton$ i.i.d samples with a statistical model $\statmodel$, where $\Theta \subseteq \R^d, d \geq 1$. Let $\trtth$ be the true parameter and $\tth_0 \in \Theta$. We want to test $H_0: \trtth = \tth_0$ against $H_1: \trtth \neq \tth_0$. Let $\etthn^{MLE}$ be the maximum likelihood estimator, assuming the conditions are satisfied. Under $H_0$, using Slutsky:

$$\left.\begin{array}{r} \sqrt{n}I(\tth_0)^{1/2} \\ \sqrt{n}I(\trtth)^{1/2} \\ \sqrt{n}I(\etthn^{MLE})^{1/2}\end{array}\right\} \times \left(\etthn^{MLE} - \tth_0\right) \convd \Norm_d(0, I_d)$$

Hence, by taking the 2-norm, we have the *Wald's test statistic*:

$$\tag{4.8}T_n = n\left(\etthn^{MLE} - \tth_0\right)^\top I\left(\etthn^{MLE}\right)\left(\etthn^{MLE} - \tth_0\right) \convd \chi^2_d$$

The corresponding test with level $\alpha$ is:

$$\psi_\alpha = \one\{T_n > q_\alpha(\chi^2_d)\}$$

Wald's test is also valid for a one-sided test, *but is less powerful*.

**Likelihood ratio test**: let $r \in \lb 0, d \rb$. Let $\tth_0 = (\theta^0_{r+1}, \ldots, \theta^0_d)^\top \in \R^{d-r}$. Let us suppose the null hypothesis is given by:

$$H_0: (\theta_{r+1}, \ldots, \theta_d)^\top = \tth_{r+1\ldots d} = \tth_0$$

Let $\etthn^{MLE}$ be the maximum likelihood estimator, and

$$\tag{4.9}\etthn^c = \argmax{\theta \in \Theta_0}\ell_n(\theta)$$

be the constrained maximum likelihood estimator under the null hypothesis. The *likelihood ratio test statistic* is:

$$\tag{4.10}T_n = 2\left(\ell_n(\etthn^{MLE}) - \ell_n(\etthn^c)\right)$$

**Wilk's Theorem**: under $H_0$, if the MLE conditions are satisfied:

$$\tag{4.11}T_n \convd \chi^2_{d-r}$$

The corresponding test with level $\alpha$ is:

$$\psi_\alpha = \one\{T_n > q_\alpha(\chi^2_{d-r})\}$$

**Implicit testing**: let $\gg: \R^d \rightarrow \R^k$ be continuously differentiable, with $k \leq d$. We want to test $H_0: \gg(\tth) = \zz$ against $H_1: \gg(\tth) \neq \zz$. By applying the Delta Method:

$$\sqrt{n}\left(\gg(\etthn) - \gg(\tth)\right) \convd \Norm_k(\zz_k, \underbrace{\nabla \gg(\tth)^\top\SSigma(\tth)\nabla \gg(\tth)}_{\Gamma(\tth)})$$

If $\nabla \gg(\tth)$ has rank $k$ and $\Gamma(\tth)$ is invertible and continuous, we can apply the Wald's Test method, under $H_0: \gg(\tth) = \zz$:

$$\tag{4.12}T_n = n\gg(\etthn)^\top\Gamma^{-1}(\etthn)\gg(\etthn) \convd \chi^2_k$$

The corresponding test with level $\alpha$ is:

$$\psi_\alpha = \one\{T_n > q_\alpha(\chi^2_k)\}$$

### Lecture 15: Goodness of Fit Test for Discrete Distributions

**Categorical distribution**: let $E = \{a_1, \ldots, a_K\}$ be a finite space and $(\P_\pp)_{\pp \in \Delta_K}$ be the family of all distributions over $E$:

$$\tag{4.13}\Delta_K = \left\{\pp \in (0, 1)^K, \sum_{k=1}^Kp_k = 1\right\}$$

The distribution of $X \sim \P_\pp$ is defined by

$$\tag{4.14}\forall k \in \lb 1, K \rb, \P_\pp(X = a_k) = p_k$$

**Goodness of fit test**: let us consider $\pp, \pp^0 \in \Delta_K$, and $\Xton \iid \P_\pp$. We want to test $H_0: \pp = \pp^0$ against $H_1: \pp \neq \pp^0$.

For example, we can test against the uniform distribution $\pp^0 = (1/K, \ldots, 1/K)^\top$.

We cannot apply Wald's test directly because of the constraint $\sum_kp_k = 1$. Under this constraint, the MLE is:

$$\tag{4.15}\forall k \in \lb 1, K \rb, \hat{p}_k = \frac{N_k}{n}$$

where $N_k = |\{X_i = a_k, i \in \lb 1, n \rb\}|$ is the number of occurences of the $k$-th element in the data.

**Theorem**: under $H_0$,

$$\tag{4.16}T_n = n\sum_{k=1}^K\frac{(\hat{p}_k - p^0_k)^2}{p^0_k} \convd \chi^2_{K-1}$$

The associated test therefore is:

$$\psi_\alpha = \one\{T_n > q_\alpha(\chi^2_{K-1})\}$$

If, instead of testing against a single parameter $\pp^0$, we want to test against a family of distributions, i.e. if we have the following problem:

$$H_0: \P_\pp \in \{\P_\tth\}_{\tth \in \Theta}, \quad H_1: \P_\pp \not\in \{\P_\tth\}_{\tth \in \Theta}$$

with $\Theta \subseteq \R^d, d \in \N^*$, then the test statistic is

$$\tag{4.17}T_n = n\sum_{k=1}^K\frac{(\hat{p}_k - \P_{\etth}[X=a_k])^2}{\P_{\etth}[X=a_k]} \convd \chi^2_{K-1-d}$$

where $\etth$ is the MLE estimator of $\tth$ given the data under $H_0$.

For example, let us test against any binomial distribution $\mathcal{Binom}(N, p)$ where $p$ is unknown. The support is of cardinal $N+1$, hence $K = N+1$, and the dimension of $\Theta$ is $d=1$; hence, the asymptotic distribution is $\chi^2_{K-1-d} = \chi^2_{N-1}$.

### Lecture 16: Goodness of Fit Test Continued: Kolmogorov-Smirnov test, Kolmogorov-Lilliefors test, Quantile-Quantile Plots

Let $\Xton$ be i.i.d random variables. The CDF of a random variable $X$ is defined as:

$$\forall t \in \R, F(t) = \P[X \leq t]$$

The **empirical CDF** of the sample $\Xton$ is defined as:

$$\tag{4.18}F_n(t) = \frac1n\sum_{i=1}^n\one\{X_i \leq t\} = \frac{|\{X_i \leq t, i \in \lb 1, n \rb\}|}n$$

The **Glivenko-Cantelli Theorem** (Fundamental theorem of statistics) gives us:

$$\tag{4.19}\sup_{t\in\R}|F_n(t) - F(t)| \convas 0$$

By the central limit theorem, as $\P[X \leq t] = F(t)$ corresponds to a Bernoulli distribution, we know that:

$$\tag{4.20}\forall t \in \R, \sqrt{n}(F_n(t) - F(t)) \convd \Norm(0, F(t)(1 - F(t)))$$

**Donsker's Theorem**: if $F$ is continuous, then

$$\tag{4.21}\sqrt{n}\sup_{t\in\R}|F_n(t) - F(t)| \convd \sup_{t \in [0, 1]} |\B(t)|$$

where $\B$ is the Brownian bridge distribution over $[0, 1]$, and more importantly, a *pivot distribution*.

**Kolmogorov-Smirnov test**: let $\Xton$ be i.i.d random variables with unknown CDF $F$. Let $F^0$ be a *continuous* CDF. We want to test $H_0: F = F^0$ against $H_1: F \neq F^0$. Let $F_n$ be the empirical CDF of the sample. Then, under $H_0$:

$$T_n = \sqrt{n}\sup_{t \in \R}|F_n(t) - F^0(t)| \convd Z \sim \sup_{t \in [0, 1]}|\B(t)|$$

The associated test is the *Kolmogorov-Smirnov* test, with level $\alpha$:

$$\tag{4.22}\delta^{KS}_\alpha = \one\{T_n > q_\alpha(Z)\}$$

**Please be careful**: some tables gives the values for $\frac{T_n}{\sqrt{n}}$, instead of $T_n$.

We can compute $T_n$ with a formula, using the property that $F^0$ is increasing and $F_n$ piecewise constant. Let us reorder our samples $X_{(1)} \leq X_{(2)} \leq \ldots \leq X_{(n)}$. Then:

$$\tag{4.23}T_n = \sqrt{n}\max_{i \in \lb 1, n \rb}\left(\max\left(\left|\frac{i-1}n - F^0(X_{(i)})\right|, \left|\frac{i}n - F^0(X_{(i)})\right|\right)\right)$$

**Pivotal distribution**: let us consider $U_i = F^0(X_i)$, with associated empirical CDF $G_n$. Under $H_0$, $U_1, \ldots, U_n \iid \Unif(0, 1)$ and

$$T_n = \sqrt{n}\sup_{x \in [0, 1]}|G_n(x) - x|$$

Which justifies that $T_n$ is indeed a *pivotal statistic*.

To estimate the quantiles numerically, as long as we can generate random values along a uniform distribution, we can proceed as follows:

* With $M$ large, simulate $M$ copies of $T_n$, $T_n^1, \ldots, T_n^M$,
* Estimate the $q_\alpha(T_n)$ quantile with $\hat{q}_\alpha^M(T_n)$ by finding the $1-\alpha$ sample cutoff among the $T_n^m$,
* Apply the Kolmogorov-Smirnov test with $\delta_\alpha = \one\{T_n > \hat{q}_\alpha^M(T_n)\}$.

The $p$-value is then given by:

$$\pval \approx \frac{|\{T_n^m > T_n, m \in \lb 1, M \rb \}|}M$$

Other distances can measure the difference between two functions than the $\sup$. For example, the Cramér-Von Mises and the Anderson-Darling distances:

$$d^{CVM}(F_n, F) = \int_\R(F_n(t) - F(t))^2dF(t) = \E_F[(F_n(X) - F(X))^2]$$

$$d^{AD}(F_n, F) = \int_\R\frac{(F_n(t) - F(t))^2}{F(t)(1-F(t))}dF(t) = \E_F\left[\frac{(F_n(X) - F(X))^2}{F(X)(1-F(X))}\right]$$

**The Kolmogorov-Smirnov test is not valid against a family of distributions**. For example, if we want to test if $X$ has any Gaussian distribution, we cannot simply plugin the estimators $\hat\mu, \est{\sigma^2}$ into the Kolmogorov-Smirnov estimator.

**Kolmogorov-Lilliefors test**. However, for a Gaussian distribution,

$$T_n = \sup_{t\in \R}|F_n(t) - \Phi_{\hat\mu, \est{\sigma^2}}(t)| \sim Z_n$$

does *not* depend on any parameter; as such, it is a pivotal statistic, and gives us the Kolmogorov-Lilliefors test:

$$\tag{4.24}\delta^{KL}_\alpha = \one\{T_n > q_\alpha(Z_n)\}$$

**Quantile-quantile (*QQ*) plots**: informal visual cues to decide whether it's likely a distribution is close to another one. Given a sample CDF $F_n$ and a target CDF $F$, we plot:

$$\tag{4.25}\left(F^{-1}\left(\frac1n\right), F_n^{-1}\left(\frac1n\right)\right), \left(F^{-1}\left(\frac2n\right), F_n^{-1}\left(\frac2n\right)\right), \ldots, \left(F^{-1}\left(\frac{n-1}n\right), F_n^{-1}\left(\frac{n-1}n\right)\right)$$

If the plot is aligned along the $y = x$ axis, they are likely close to each other. There are four patterns of differences between distributions:

* *Heavier tails*: below > above the diagonal.
* *Lighter tails*: above > below the diagonal.
* *Right-skewed*: above > below > above the diagonal.
* *Left-skewed*: below > above > below the diagonal.

## Unit 5: Bayesian Statistics

### Lecture 17: Introduction to Bayesian Statistics

Bayesian inference conceptually amounts to weighting the likelihood $L_n(\tth)$ by a prior knowledge we might have on $\tth$.

Given a statistical model $\sstatmodel$, we technically model our parameter $\tth$ as if it were a random variable. We therefore define the **prior distribution** (PDF):

$$\pi(\tth)$$

Let $\Xton$. We note $L_n(\Xton|\tth)$ the joint probability distribution of $\Xton$ conditioned on $\tth$ where $\tth \sim \pi$. This is exactly the likelihood from the frequentist approach.

**Bayes' formula**. The **posterior distribution** verifies:

$$\tag{5.1}\forall \tth \in \Theta, \pi(\tth|\Xton) \propto \pi(\tth)L_n(\Xton | \tth)$$

The constant is the normalization factor to ensure the result is a proper distribution, and does not depend on $\tth$:

$$\pi(\tth|\Xton) = \frac{\pi(\tth)L_n(\Xton | \tth)}{\int_\Theta\pi(\tth)L_n(\Xton | \tth)d\tth}$$

We can often use an **improper prior**, i.e. a prior that is not a proper probability distribution (whose integral diverges), and still get a proper posterior. For example, the improper prior $\pi(\tth) = 1$ on $\Theta$ gives the likelihood as a posterior.

### Lecture 18: Jeffrey's Prior and Bayesian Confidence Interval

**Jeffreys Prior** is defined as:

$$\tag{5.2} \pi_J(\tth) \propto \sqrt{\det I(\tth)}$$

where $I(\tth)$ is the Fisher information. This prior is **invariant by reparameterization**, which means that if we have $\eeta = \phi(\tth)$, then the same prior gives us a probability distribution for $\eeta$ verifying:

$$\tag{5.3}\tilde\pi_J(\eeta) \propto \sqrt{\det \tilde I(\eeta)}$$

The change of parameter follows the following formula:

$$\tag{5.4}\tilde\pi_J(\eeta) = \det(\nabla \phi^{-1}(\eeta)) \pi_J(\phi^{-1}(\eeta))$$

**Bayesian confidence region**. Let $\alpha \in (0, 1)$. A *Bayesian confidence region with level $\alpha$* is a random subset $\mathcal{R} \subset \Theta$ depending on $\Xton$ (and the prior $\pi$) such that:

$$\tag{5.5}\P[\tth \in \mathcal{R} | \Xton] \geq 1 - \alpha$$

*Bayesian confidence region and confidence interval are **distinct** notions*.

The Bayesian framework can be used to estimate the true underlying parameter. In that case, it is used to build a new class of estimators, based on the posterior distribution.

The **Bayes estimator** (*posterior mean*) is defined as:

$$\tag{5.6}\etth_{(\pi)} = \int_\Theta\tth\pi(\tth | \Xton)d\tth$$

The **Maximum a posteriori (*MAP*) estimator** is defined as:

$$\tag{5.7}\etth^{MAP}_{(\pi)} = \argmax{\tth\in\Theta}\pi(\tth | \Xton)$$

The MAP is equivalent to the MLE, if the prior is uniform.

## Unit 6: Linear Regression

### Lecture 19: Linear Regression 1

Given two random variables $\XX$ and $Y$, how can we predict the values of $Y$ given $\XX$?

Let us consider $(X_1, Y_1), \ldots, (X_n, Y_n) \iid \P$ where $\P$ is an unknown joint distribution. $\P$ can be described entirely by:

$$g(X) = \int f(X, y)dy, \quad h(Y|X=x) = \frac{f(x, Y)}{g(x)}$$

where $f$ is the joint PDF, $g$ the marginal density of $X$ and $h$ the conditional density. What we are interested in is $h(Y|X)$.

**Regression function**: For a partial description, we can consider instead the conditional expection of $Y$ given $X=x$:

$$\tag{6.1}x \mapsto f(x) = \E[Y | X=x] = \int yh(y|x)dy$$

We can also consider different descriptions of the distribution, like the median, quantiles or the variance.

**Linear regression**: trying to fit any function to $\E[Y | X=x]$ is a nonparametric problem; therefore, we restrict the problem to the tractable one of linear function:

$$f: x \mapsto a + bx$$

**Theoretical linear regression**: let $X, Y$ be two random variables with two moments such as $\V[X] > 0$. The theoretical linear regression of $Y$ on $X$ is the line $\true{a} + \true{b}x$ where

$$\tag{6.2}(\true a, \true b) = \argmin{(a, b) \in \R^2}\E\left[(Y - a - bX)^2\right]$$

Which gives:

$$\tag{6.3}\true b = \frac{\Cov(X, Y)}{\V[X]}, \quad \true a = \E[Y] - b^* \E[X]$$

**Noise**: we model the noise of $Y$ around the regression line by a random variable $\varepsilon = Y - \true a - \true b X$, such as:

$$\E[\varepsilon] = 0, \quad \Cov(X, \varepsilon) = 0$$

We have to estimate $\true a$ and $\true b$ from the data. We have $n$ random pairs $(X_1, Y_1), \ldots, (X_n, Y_n) \iid (X, Y)$ such as:

$$\tag{6.4} Y_i = \true a + \true b X_i + \varepsilon_i$$

The **Least Squares Estimator (*LSE*)** of $(\true a, \true b)$ is the minimizer of the squared sum:

$$\tag{6.5} (\estn a, \estn b) = \argmin{(a, b) \in \R^2}\sum_{i=1}^n(Y_i - a - bX_i)^2$$

The estimators are given by:

$$\tag{6.6} \estn b = \frac{\overline{XY} - \bar{X}\bar{Y}}{\overline{X^2} - \bar{X}^2}, \quad \estn a = \bar{Y} - \estn b \bar{X}$$

The **Multivariate Regression** is given by:

$$\tag{6.7}Y_i = \sum_{j=1}^pX_i^{(j)}\true\beta_j + \varepsilon_i= \underbrace{\XX_i^\top}_{1 \times p}\underbrace{\true\bbeta}_{p \times 1} + \varepsilon_i$$

We can assuming that the $X_i^{(1)}$ are 1 for the intercept. We call:

* the *explanatory variables* or *covariates* the $\XX_i \in \R^p$,
* the *response*, *dependent* or *explained variable* the $Y_i$,
* if $\true\bbeta = (\true a, \true\bb^\top)^\top$, $\true\beta_1 = \true a$ is the *intercept*.
* the $\varepsilon_i$ is the *noise*, satisfying $\CCov(\XX_i, \varepsilon_i) = \zz$.

The **Multivariate Least Squares Estimator (*LSE*)** of $\true\bbeta$ is the minimizer of the sum of square errors:

$$\tag{6.8}\estn \bbeta = \argmin{\bbeta \in \R^p}\sum_{i=1}^n(Y_i - \XX_i^\top\bbeta)^2$$

**Matrix form**: we can rewrite these expressions. Let $\YY = (Y_1, \ldots, Y_n)^\top \in \R^n$, and $\eepsilon = (\varepsilon_1, \ldots, \varepsilon_n)^\top$. Let

$$\tag{6.9}\X = \begin{pmatrix} \XX_1^\top \\ \vdots \\ \XX_n^\top \end{pmatrix} \in \R^{n \times p}$$

$\X$ is called the **design matrix**. The regression is given by:

$$\tag{6.10}\YY = \X\true\bbeta + \eepsilon$$

and the LSE is given by:

$$\tag{6.11} \estn \bbeta = \argmin{\bbeta \in \R^p} \|\YY - \X\bbeta\|^2_2$$

### Lecture 20: Linear Regression 2

Let us suppose $n \geq p$ and $\rank(\X) = p$. If we write:

$$F(\bbeta)  = \|\YY - \X\bbeta\|^2_2 = (\YY - \X\bbeta)^\top(\YY - \X\bbeta)$$

Then:

$$\nabla F(\bbeta) = 2 \X^\top(\YY - \X\bbeta)$$

**Least squares estimator**: setting $\nabla F(\bbeta) = \zz$ gives us the expression of $\est \bbeta$:

$$\tag{6.12} \est \bbeta = (\X^\top\X)^{-1}\X^\top\YY$$

**Geometric interpretation**: $\X\est\bbeta$ is the orthogonal projection of $\YY$ onto the subspace spanned by the columns of $\X$:

$$\tag{6.13} \X\est\bbeta = P\YY$$

where $P = \X(\X^\top\X)^{-1}\X^\top$ is the expression of the projector.

**Statistic inference**: let us suppose that:

* The design matrix $\X$ is deterministic and $\rank(\X) = p$.
* The model is **homoscedastic**: $\varepsilon_1, \ldots, \varepsilon_n$ are i.i.d.
* The noise is Gaussian: $\eepsilon \sim \Norm_n(\zz, \sigma^2I_n)$.

We therefore have:

$$\tag{6.14}\YY \sim \Norm_n(\X\true\bbeta, \sigma^2I_n)$$

Properties of the LSE:

$$\tag{6.15}\est\bbeta \sim \Norm_p(\true\bbeta, \sigma^2(\X^\top\X)^{-1})$$

The quadratic risk of $\est\bbeta$ is given by:

$$\tag{6.16}\E\left[\|\est\bbeta - \true\bbeta\|^2_2\right] = \sigma^2\Tr\left((\X^\top\X)^{-1}\right)$$

The prediction error is given by:

$$\tag{6.17}\E\left[\|\YY - \X\est\bbeta\|^2_2\right] = \sigma^2(n - p)$$

The unbiased estimator of $\sigma^2$ is:

$$\tag{6.18}\est{\sigma^2} = \frac{1}{n-p}\|\YY - \X\est\bbeta\|^2_2 = \frac1{n-p}\sum_{i=1}^n\hat{\varepsilon}_i^2$$

By **Cochran's Theorem**:

$$\tag{6.19} (n-p)\frac{\est{\sigma^2}}{\sigma^2} \sim \chi^2_{n-p}, \quad \hat\bbeta \perp \est{\sigma^2}$$

**Significance test**: let us test $H_0: \beta_j = 0$ against $H_1: \beta_j \neq 0$. Let us call

$$\gamma_j = \left((\X^\top\X)^{-1}\right)_{jj} > 0$$

then:

$$\tag{6.20}\frac{\est{\beta}_j- \beta_j}{\sqrt{\est{\sigma^2}\gamma_j}} \sim t_{n-p}$$

We can define the test statistic for our test:

$$\tag{6.21}T_n^{(j)} = \frac{\est{\beta}_j}{\sqrt{\est{\sigma^2}\gamma_j}}$$

The test with non-asymptotic level $\alpha$ is given by:

$$\tag{6.22}\psi_\alpha^{(j)} = \one\{|T_n^{(j)}| > q_{\alpha/2}(t_{n-p})\}$$

**Bonferroni's test**: if we want to test the significance level of multiple tests at the same time, we cannot use the same level $\alpha$ for each of them. We must use a stricter test for each of them. Let us consider $S \subseteq \{1, \ldots, p\}$. Let us consider

$$H_0: \forall j \in S, \beta_j = 0, \quad H_1: \exists j \in S, \beta_j \neq 0$$

The *Bonferroni's test* with significance level $\alpha$ is given by:

$$\tag{6.22}\psi_\alpha^{(S)} = \max_{j \in S}\psi_{\alpha/K}^{(j)}$$

where $K = |S|$. The rejection region therefore is the union of all rejection regions:

$$\tag{6.23}R_\alpha^{(S)} = \bigcup_{j \in S}R_{\alpha/K}^{(j)}$$

This test has nonasymptotic level at most $\alpha$:

$$\P_{H_0}\left[R_\alpha^{(S)}\right] \leq \sum_{j\in S}\P_{H_0}\left[R_{\alpha/K}^{(j)}\right] = \alpha$$

This test also works for implicit testing (for example, $\beta_1 \geq \beta_2$).

## Unit 7: Generalized Linear Models

### Lecture 21: Introduction to Generalized Linear Models; Exponential Families

The assumptions of a linear regression are:

* The **noise is Gaussian**: $Y | \XX = \xx \sim \Norm_d(\mu(\xx), \sigma^2)$,
* The **regression function is linear**: $\mu(\xx) = \xx^\top \bbeta$.

We want to relax both these assumptions for Generalized Linear Models, because some response random variables cannot fit in this framework (for example, a binary answer $Y \in \{0, 1\}$). Instead, we assume:

* **Some distribution** for the noise: $Y | \XX = \xx \sim \mathcal{D}$,
* A **link function** $g$ before the mean: $g(\mu(\xx)) = \xx^\top \bbeta$.

**Exponential family**: a family of distributions $\{\P_\tth, \tth \in \Theta\}, \Theta \subseteq \R^k$ is said to be a **$k$-parameter exponential family** on $\R^d$, if there exist:

* $\eta_1, \ldots, \eta_k, B: \Theta \rightarrow \R$,
* $T_1, \ldots, T_k, h: \R^d \rightarrow \R$,

such that the PMF/PDF of $\P_\tth$ can be written as:

$$\tag{7.1} f_\tth(\yy) = h(\yy)\exp\left(\sum_{i=1}^k\eta_i(\tth)T_i(\yy) - B(\tth)\right) = h(\yy)\exp\left(\eeta(\tth)^\top \TT(\yy) - B(\tth)\right)$$

For $k=1$ and $y \in \R$, the **canonical exponential family** form is a special form of the 1-parameter exponential family:

$$\tag{7.2} f_\theta(y) = \exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y, \phi)\right)$$

We will always assume that $\phi$ is known in this class; $\phi$ is called the **dispersion parameter**.

Let $\ell(\theta) = \ln f_\theta(Y)$ be the log-likelihood of this distribution. The log-likelihood has the following properties:

$$\E\left[\diff{\ell}{\theta}\right] = 0, \quad \E\left[\diffk{\ell}{\theta}{2}\right] + \E\left[\diff{\ell}{\theta}\right]^2 = 0$$

When the distribution is written in the canonical form, we have:

$$\tag{7.3}\ell(\theta) = \frac{Y\theta - b(\theta)}{\phi} + c(Y, \phi)$$

Combined with the previous expression, we get the following expressions:

$$\tag{7.4}\E[Y] = b'(\theta), \quad \V[Y] = \phi b''(\theta)$$

### Lecture 22: GLM: Link Functions and the Canonical Link Function

We need to link our model back to the parameter of interest, $\bbeta$. For that, we need a **link function** between our linear predictor and the mean parameter $\mu$:

$$\tag{7.5}\XX^\top \bbeta = g(\mu(\XX))$$

We require $g$ to be monotone increasing and differentiable.

$g$ maps the domain of the parameter $\mu$ of the distribution to the entire real line (the range of $\XX^\top\bbeta$). For example:

* For a linear model, $g$ is the identity.
* For a Poisson distribution or an Exponential distribution ($\mu > 0$), we can use $g = \ln: (0, \infty) \rightarrow \R$.
* For a Bernoulli distribution, we can use:
    - The logit function $g: \mu \mapsto \ln(\frac{\mu}{1 - \mu})$,
    - The probit function $g: \mu \mapsto \Psi^{-1}(\mu)$ ($\Psi$ being the normal CDF).

The logit is the natural choice; such as model is called a **logistic regression**.

The link $g$ mapping $\mu$ to the parameter $\theta$ is called the **canonical link**:

$$\tag{7.6} g_c(\mu) = \theta$$

As $\mu = b'(\theta)$,

$$\tag{7.7} g_c(\mu) = (b')^{-1}(\mu)$$

If $\phi > 0$, as $b''(\theta) = \phi\V[Y] > 0$, $b'$ is strictly increasing and $g_c$ is also **strictly increasing**.

**Back to $\bbeta$**: let us consider $(\XX_1, Y_1), \ldots, (\XX_n, Y_n) \in \R^{p+1}$ i.i.d, such that the PDF of $Y_i | \XX_i = \xx_i$ has density in the canonical exponential family:

$$\tag{7.8} f_{\theta_i}(y_i) = \exp\left(\frac{y_i\theta_i - b(\theta_i)}{\phi} + c(y_i, \phi)\right)$$

Using the matrix notation $\YY = (Y_1, \ldots, Y_n)^\top$, $\X = (\XX_1, \ldots, \XX_n)^\top \in \R^{n\times p}$, the parameters $\theta_i$ are linked to $\beta$ via the following relations:

$$\E[Y_i|X_i] = \mu_i = b'(\theta_i), \quad g(\mu_i) = \XX_i^\top \bbeta$$

Therefore, given a link $g$:

$$\tag{7.9}\theta_i = (b')^{-1}(g^{-1}(\XX_i^\top \bbeta)) = (g \circ b')^{-1}(\XX_i^\top \bbeta) = h(\XX_i^\top \bbeta)$$

where $h = (g \circ b')^{-1}$. If $g$ is the **canonical link**, $h = \Id$.

The log-likelihood is given by:

$$\tag{7.10} \ell_n(\YY, \X, \bbeta) = \sum_{i=1}^n\frac{Y_ih(\XX_i^\top\bbeta) - b(h(\XX_i^\top\bbeta))}{\phi} + K$$

If we use the **canonical link** $g_c$, then:

$$\tag{7.11} \ell_n(\YY, \X, \bbeta) = \sum_{i=1}^n\frac{Y_i\XX_i^\top\bbeta - b(\XX_i^\top\bbeta)}{\phi} + K$$

The Hessian of this expression is given by:

$$\tag{7.12} \HH_\bbeta\ell_n(\bbeta) = -\frac1{\phi}\sum_{i=1}^nb''(\XX_i^\top\bbeta)\XX_i\XX_i^\top \prec 0$$

Which means the log-likelihood is **strictly concave** when using the canonical link and $\phi > 0$. Therefore, the MLE is unique (if it exists).

In general, there is no closed form for the MLE, but we can approximate it using optimization algorithms, such as the gradient descent.

The MLE is also asymptotically normal:

$$\tag{7.13}\sqrt{n}(\est\bbeta^{MLE} - \bbeta) \convd \Norm_p(\zz, I^{-1}(\bbeta))$$

We can therefore apply our statistical tests (Wald's, likelihood ratio, …) to test hypotheses about our parameter (for example, the significance of some $\beta_i$).

## Addendum A: Frequent distributions

### Part 1: Normal distribution

### Description [A.1]

Notation: $\Norm(\mu, \sigma^2)$

Parameters: $\mu \in \R, \sigma^2 > 0$

Support: $E = \R$

Probability density function (*PDF*):

$$\tag{A.1.1} f_{\mu, \sigma^2}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac1{2\sigma^2}(x-\mu)^2\right)$$

The cumulative density function (*CDF*) for the standard normal distribution $Z \sim \Norm(0, 1)$ is noted $\Phi(x)$:

$$\tag{A.1.2}\P[Z \leq x] = \Phi(x)$$

### Properties [A.1]

Mean: $\E[X] = \mu$

Variance: $\V[X] = \sigma^2$

To *normalize* a normal random variable:

$$\tag{A.1.3}Z = \frac{X - \mu}{\sigma} \sim \Norm(0, 1)$$

*Quantiles*:

$$\tag{A.1.4} q_\alpha = \Phi^{-1}(1 - \alpha)$$

$$\tag{A.1.5} \P[Z \leq q_\alpha] = 1 - \alpha$$

$$\tag{A.1.6} \P[|Z| \leq q_{\alpha/2}] = 1 - \alpha$$

$q_{0.1} \approx 1.28155$, $q_{0.05} \approx 1.64485$, $q_{0.025} \approx 1.95996$,
$q_{0.01} \approx 2.32635$, $q_{0.005} \approx 2.57583$.

*Fischer information*:

$$\tag{A.1.7} I(\mu, \sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}$$

*Likelihood*:

$$\tag{A.1.8} L_n(\Xton, \mu, \sigma^2) = \frac1{(2\pi\sigma^2)^{\frac{n}2}}\exp\left(-\frac1{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2\right)$$

*Log-likelihood*:

$$\tag{A.1.9} \ell_n(\Xton, \mu, \sigma^2) = -\frac{n}2\ln2\pi -\frac{n}2\ln\sigma^2 - \frac1{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2$$

*Maximum likelihood estimator*:

$$\tag{A.1.10} \hat\mu_n = \bar{X}_n, \quad \est{\sigma^2}_n = \frac1n\sum_{i=1}^n(X_i - \bar{X}_n)^2$$

*Exponential family form*, $\tth = (\mu, \sigma^2)^\top$:

$$\tag{A.1.11} f_{\mu, \sigma^2}(x) = \overbrace{\frac{1}{\sqrt{2\pi}}}^{h(x)}\exp\Big(\overbrace{\frac{\mu}{\sigma^2}x - \frac{1}{2\sigma^2}x^2}^{\eeta(\tth)^\top\TT(x)} - \overbrace{\frac{\mu^2}{2\sigma^2} - \frac12\ln(\sigma^2)}^{B(\tth)}\Big)$$

*Canonical exponential form*, $\sigma^2$ known:

$$\tag{A.1.12} f_{\theta}(x) = \exp\Bigg(\frac{x\theta - \overbrace{\frac{\theta^2}{2}}^{b(\theta)}}{\phi} \overbrace{- \frac{x^2}{2\phi} - \frac12\ln(2\pi\phi)}^{c(x, \phi)}\Bigg), \quad \theta = \mu, \phi = \sigma^2$$

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

*Likelihood*:

$$\tag{A.2.2} L_n(\Xton, p) = p^{\discr \sum_{i=1}^nX_i}(1-p)^{\discr n-\sum_{i=1}^nX_i}$$

*Log-likelihood*:

$$\tag{A.2.3} \ell_n(\Xton, p) = \ln p\sum_{i=1}^nX_i + \ln(1-p)\left(n - \sum_{i=1}^nX_i\right)$$

*Maximum likelihood estimator*:

$$\tag{A.2.4} \hat{p}_n = \frac{\sum_iX_i}n$$

*Canonical exponential form*:

$$\tag{A.2.5} f_{\theta}(x) = \exp\big(x\theta - \overbrace{\ln(1 + e^\theta)}^{b(\theta)} + \overbrace{0}^{c(x, \phi)}\big), \quad \theta = \ln\left(\frac{p}{1-p}\right), \phi = 1$$

## Part 3: Binomial distribution

### Description [A.3]

Notation: $\Binom(N, p)$

Parameters: $N \in \N^*, p \in (0, 1)$

Support: $E = \lb 0, N \rb$

Probability mass function (*PMF*):

$$\tag{A.3.1} f_{N, p}(k) = \binom{N}kp^k(1-p)^{N-k}$$

### Properties [A.3]

Mean: $\E[X] = Np$

Variance: $\V[X] = Np(1-p)$

Fischer information: $I_N(p) = \frac{N}{p(1-p)}$ (for fixed $N$)

*Likelihood*:

$$\tag{A.3.2} L_n(\Xton, N, p) = \prod_{i=1}^n\binom{N}{X_i} p^{\discr \sum_{i=1}^nX_i}(1-p)^{\discr nN - \sum_{i=1}^nX_i}$$

*Log-likelihood*:

$$\tag{A.3.3} \ell_n(\Xton, N, p) = \sum_{i=1}^n\ln\binom{N}{X_i} + \ln p \sum_{i=1}^nX_i + \ln(1-p)\left(nN - \sum_{i=1}^nX_i\right)$$

*Maximum likelihood estimator* (for fixed $N$):

$$\tag{A.3.4} \hat{p}_n = \frac{\sum_iX_i}{nN}$$

*Canonical exponential form*, $N$ known:

$$\tag{A.3.5} f_{\theta}(x) = \exp\bigg(x\theta - \overbrace{N\ln(1 + e^\theta)}^{b(\theta)} + \overbrace{\ln\binom{N}{x}}^{c(x, \phi)}\bigg), \quad \theta = \ln\left(\frac{p}{1-p}\right) = \ln\left(\frac{\mu}{N-\mu}\right), \phi = 1$$

## Part 4: Categorical distribution

### Description [A.4]

Notation: $\mathcal{Cat}(\pp)$

Parameters: $\pp \in \Delta_K = \{\pp \in (0, 1)^K, \sum_kp_k = 1\}$

Support: $E = \{a_1, \ldots, a_K\} \sim \lb 1, K \rb$

Probability mass function (*PMF*):

$$\tag{A.4.1} f_\pp(a_k) = p_k$$

### Properties [A.4]

Given a sample $\Xton$, we define the number of occurences of each outcome:

$$\forall k \in \lb 1, K \rb, N_k = \sum_{i=1}^n\one\{X_i = a_k\} = |\{i \in \lb 1, n \rb, X_i = a_k\}|$$

*Likelihood*:

$$\tag{A.4.1} L_n(\Xton, \pp) = \prod_{k=1}^Kp_k^{N_k}$$

*Log-likelihood*:

$$\tag{A.4.2} \ell_n(\Xton, \pp) = \sum_{k=1}^KN_k\ln p_k$$

*Maximum likelihood estimator* (under the constraint $\sum_kp_k = 1$):

$$\tag{A.4.3} \forall k \in \lb 1, K \rb, \hat{\pp}_n = \left(\frac{N_1}n, \ldots, \frac{N_K}n\right)$$

*Exponential family form*, $\tth = \pp$:

$$\tag{A.4.4} f_{\pp}(x) = \overbrace{1}^{h(x)}\times \exp\bigg(\sum_{k=1}^K\overbrace{\ln(p_k)}^{\eta_k(\tth)}\overbrace{\one\{x = a_k\}}^{T_k(x)} - \overbrace{0}^{B(\tth)}\bigg)$$

## Part 5: Poisson distribution

### Description [A.5]

Notation: $\Poiss(\lambda)$

Parameters: $\lambda > 0$

Support: $E = \N$

Probability mass function (*PMF*):

$$\tag{A.5.1} f_\lambda(k) = \frac{\lambda^ke^{-\lambda}}{k!}$$

### Properties [A.5]

Mean: $\E[X] = \lambda$

Variance: $\V[X] = \lambda$

Fischer information: $I(\lambda) = \frac1\lambda$

*Likelihood*:

$$\tag{A.5.2} L_n(\Xton, \lambda) = \frac{1}{\discr \prod_{i=1}^nX_i!} \lambda^{\discr \sum_{i=1}^nX_i} e^{-n\lambda}$$

*Log-likelihood*:

$$\tag{A.5.3} \ell_n(\Xton, \lambda) = \ln\lambda\sum_{i=1}^nX_i - n\lambda - \sum_{i=1}^n\ln(X_i!)$$

*Maximum likelihood estimator*:

$$\tag{A.5.4} \hat{\lambda}_n = \bar{X}_n = \frac1n\sum_{i=1}^nX_i$$

*Canonical exponential form*:

$$\tag{A.5.5} f_{\theta}(x) = \exp\big(x\theta - \overbrace{e^\theta}^{b(\theta)} \overbrace{- \ln x!}^{c(x, \phi)}\big), \quad \theta = \ln \lambda, \phi = 1$$

## Part 6: Uniform distribution

### Description [A.6]

Notation: $\Unif(a, b)$

Parameters: $a, b \in \R, a < b$ (usually, $a = 0$)

Support: $E = [a, b] \subset \R$

Probability density function (*PDF*):

$$\tag{A.6.1} f_{a, b}(x) = \frac1{b-a}\one\{a \leq x \leq b\}$$

Cumulative density function (*CDF*):

$$\tag{A.6.2} F_{a, b}(x) = \begin{cases}\displaystyle 0 & \text{if}\quad x < a \\ \displaystyle \frac{x-a}{b-a} & \text{if}\quad a \leq x \leq b \\ \displaystyle 1 & \text{if}\quad x > b \end{cases}$$

### Properties [A.6]

Mean: $\E[X] = \frac{a+b}2$

Variance: $\V[X] = \frac1{12}(b-a)^2$

Moments: $\E[X^k] = \frac{1}{n+1}\sum_{i=0}^ka^ib^{k-i}$

*Likelihood*:

$$\tag{A.6.3} L_n(\Xton, a, b) = \frac1{(b - a)^n}\one\left\{\min_{i \in \lb 1, n \rb}X_i \geq a\right\}\one\left\{\max_{i \in \lb 1, n \rb}X_i \leq b\right\}$$

*Log-likelihood*:

$$\tag{A.6.4} \ell_n(\Xton, a, b) = n\ln\frac1{b-a} + \ln\one\left\{\min_{i \in \lb 1, n \rb}X_i \geq a\right\} + \ln\one\left\{\max_{i \in \lb 1, n \rb}X_i \leq b\right\}$$

*Maximum likelihood estimator* (cannot differentiate the log-likelihood):

$$\tag{A.6.5} \hat{a}_n = \min_{i \in \lb 1, n \rb}X_i, \quad \hat{b}_n = \max_{i \in \lb 1, n \rb}X_i$$

## Part 7: Exponential distribution

### Description [A.7]

Notation: $\Exp(\lambda)$

Parameters: $\lambda > 0$

Support: $E = [0, +\infty)$

Probability density function (*PDF*):

$$\tag{A.7.1} f_\lambda(x) = \lambda e^{-\lambda x}\one\{x \geq 0\}$$

Cumulative density function (*CDF*):

$$\tag{A.7.2} F_\lambda(x) = (1 - e^{-\lambda x})\one\{x \geq 0\}$$

### Properties [A.7]

Mean: $\E[X] = \frac1\lambda$

Variance: $\V[X] = \frac1{\lambda^2}$

Moments: $\E[X^k] = \frac{k!}{\lambda^k}$

Fischer Information: $I(\lambda) = \frac1{\lambda^2}$

*Likelihood*:

$$\tag{A.7.3} L_n(\Xton, \lambda) = \lambda^n \exp\left(-\lambda \sum_{i=1}^nX_i\right)\one\left\{\min_{i \in \lb 1, n \rb}X_i \geq 0\right\}$$

*Log-likelihood*:

$$\tag{A.7.4} \ell_n(\Xton, \lambda) = n\ln\lambda - \lambda\sum_{i=1}^nX_i+\ln\one\left\{\min_{i \in \lb 1, n \rb}X_i \geq 0\right\}$$

*Maximum likelihood estimator*:

$$\tag{A.7.5} \hat{\lambda}_n = \frac{n}{\sum_iX_i} = \frac1{\bar{X}_n}$$

Memorylessness: $\P[X > s+t | X > s] = \P[X > t]$

*Canonical exponential form*:

$$\tag{A.7.6} f_{\theta}(x) = \exp\big(x\theta - \overbrace{(-\ln(-\theta))}^{b(\theta)} + \overbrace{0}^{c(x, \phi)}\big), \quad \theta = -\lambda = -\frac1{\mu}, \phi = 1$$

## Part 8: Multivariate normal distribution

### Description [A.8]

Notation: $\Norm_d(\mmu, \SSigma)$

Parameters: $\mmu \in \R^d, \SSigma\in \R^{d\times d}, \SSigma \succ 0$

Support: $E = \R^d$

Probability density function (*PDF*):

$$\tag{A.8.1} f_{\mmu, \SSigma}(\xx) = \frac{1}{\sqrt{(2\pi)^d|\SSigma|}}\exp\left(-\frac12(\xx - \mmu)^\top\SSigma^{-1}(\xx - \mmu)\right)$$

### Properties [A.8]

Mean: $\E[\XX] = \mmu$

Variance: $\V[\XX] = \SSigma$

## Part 9: Cauchy distribution

### Description [A.9]

Notation: $\mathcal{Cau}(x_0, \gamma)$

Parameters: $x_0 \in \R, \gamma > 0$

Support: $E = \R$

Probability density function (*PDF*):

$$\tag{A.9.1} f_{x_0, \gamma}(x) = \frac{1}{\pi\gamma}\frac{\gamma^2}{\gamma^2 + (x-x_0)^2}$$

Cumulative density function (*CDF*):

$$\tag{A.9.2} F_{x_0, \gamma}(x) = \frac12 + \frac1{\pi}\arctan\frac{x-x_0}{\gamma}$$

### Properties [A.9]

Mean: undefined

Variance: undefined

Moments: undefined

## Part 10: Laplace distribution

### Description [A.10]

Notation: $\mathcal{Lapl}(\mu, b)$

Parameters: $\mu \in \R, b > 0$

Support: $E = \R$

Probability density function (*PDF*):

$$\tag{A.10.1} f_{\mu, b}(x) = \frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)$$

Cumulative density function (*CDF*):

$$\tag{A.10.2} F_{\mu, b}(x) = \begin{cases} \displaystyle \frac12\exp\left(\frac{x-\mu}{b}\right) & \text{if}\quad x \leq \mu \\ \displaystyle 1 - \frac12\exp\left(\frac{\mu - x}{b}\right) & \text{if}\quad x > \mu \end{cases}$$

### Properties [A.10]

Mean: $\E[X] = \mu$

Variance: $\V[X] = 2b^2$

*Likelihood*:

$$\tag{A.10.3} L_n(\Xton, \mu, b) = \frac1{(2b)^n}\exp\left(-\frac1b\sum_{i=1}^n|X_i - \mu|\right)$$

*Log-likelihood*:

$$\tag{A.10.4} \ell_n(\Xton, \mu, b) = -n\ln2 -n\ln b - \frac1b\sum_{i=1}^n|X_i - \mu|$$

*Maximum Likelihood Estimators*:

$$\hat{\mu}_n = \Med{i \in \lb 1, n \rb}X_i, \quad \hat{b}_n = \frac{1}n\sum_{i=1}^n|X_i - \hat{\mu}|$$

## Part 11: Chi-squared distribution

### Description [A.11]

Notation: $\chi^2_d$ ($d$ fixed)

Support: $E = [0, +\infty)$

Definition: $Z_1^2 + \ldots Z_d^2 \sim \chi^2_d$, where $Z_1, \ldots, Z_d \iid \Norm(0, 1)$.

Probability density function (*PDF*):

$$\tag{A.11.1} f_d(x) = \frac{1}{2^{d/2}\Gamma(d/2)}x^{d/2-1}e^{-x/2}$$

### Properties [A.11]

Mean: $\E[X] = d$

Mode: $\Mode(X) = \max(d-2, 0)$

Variance: $\V[X] = 2d$

Useful for many test statistics as a pivot distribution.

## Part 12: Beta distribution

### Description [A.12]

Notation: $\mathcal{Beta}(a, b)$

Parameters: $a, b \in (0, +\infty)$

Support: $E = [0, 1]$ or $(0, 1)$

Probability density function (*PDF*):

$$\tag{A.12.1} f_{a, b}(x) = \frac{x^{a-1}(1-x)^{b-1}}{B(a, b)}$$

where $B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$.

### Properties [A.12]

Mean: $\E[X] = \frac{a}{a+b}$

Mode: $\Mode(X) = \frac{a - 1}{a + b - 2}$

Variance: $\V[X] = \frac{ab}{(a+b)^2(a+b+1)}$

Useful as the *Bayesian conjugate for Bernoulli distributions*.

*Exponential family form*, $\tth = (a, b)^\top$:

$$\tag{A.12.2} f_{a, b}(x) = \overbrace{1}^{h(x)}\times \exp\big(\overbrace{(a-1)\ln(x) + (b-1)\ln(1-x)}^{\eeta(\tth)^\top\TT(x)} - \overbrace{\ln(B(a, b))}^{B(\tth)}\big)$$

## Part 13: Gamma distribution

### Description [A.13]

Notation: $\mathcal{Gamma}(\alpha, \beta), \mathcal{Gamma}(k, \tau)$

Parameters: $\alpha, \beta; k, \tau \in (0, +\infty)$

Support: $E = (0, +\infty)$

Probability density function (*PDF*):

$$\tag{A.13.1} f_{\alpha, \beta}(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}\exp(-\beta x), \quad f_{k, \tau}(x) = \frac{1}{\Gamma(k)\tau^k}x^{k-1}\exp\left(-\frac{x}\tau\right)$$

### Properties [A.13]

Mean: $\E[X] = \frac{\alpha}{\beta} = k\tau$

Mode: $\Mode(X) = \frac{\alpha - 1}{\beta} = (k-1)\tau$

Variance: $\V[X] = \frac{\alpha}{\beta^2} = k\tau^2$

Useful as the *Bayesian conjugate for exponential distributions*.

*Exponential family form*, $\tth = (\alpha, \beta)^\top$:

$$\tag{A.13.2} f_{\alpha, \beta}(x) = \overbrace{1}^{h(x)}\times \exp\big(\overbrace{(\alpha-1)\ln(x) -\beta x}^{\eeta(\tth)^\top\TT(x)} - \overbrace{(-\alpha \ln\beta  + \ln\Gamma(\alpha))}^{B(\tth)}\big)$$

*Exponential family form*, $\tth = (k, \tau)^\top$:

$$\tag{A.13.3} f_{k, \tau}(x) = \overbrace{1}^{h(x)}\times \exp\bigg(\overbrace{(k-1)\ln(x) -\frac1\tau x}^{\eeta(\tth)^\top\TT(x)} - \overbrace{(k \ln\tau  + \ln\Gamma(k))}^{B(\tth)}\bigg)$$

*Canonical exponential form*, $\alpha$ known:

$$\tag{A.13.4} f_{\theta}(x) = \exp\big(x\theta - \overbrace{(-\alpha\ln(-\theta))}^{b(\theta)} + \overbrace{(\alpha - 1)\ln x - \ln \Gamma(\alpha)}^{c(x, \phi)}\big), \quad \theta = -\beta = -\frac\alpha\mu, \phi = 1$$

# Addendum B: Notable relations

## Statistics

### Probability, density and expectation

A probability space is a triplet $(\Omega, \mathcal{F}, P)$ where $\Omega$ is the set of possible outcomes, $\mathcal{F}$ a set of subsets of $\Omega$ such as $(\Omega, \mathcal{F})$ is measurable and $P: \mathcal{F} \rightarrow [0, 1]$ is the probability function such as it is (countably) additive and $P(\Omega) = 1$.

A **random variable** $X$ is a function:

$$\tag{B.1.1}X: \Omega \rightarrow E$$

Where $E$ is the value-space of $X$; usually, $E \subseteq \R$. For any $S \subseteq E$, the **probability** of $X \in S$ is given by:

$$\tag{B.1.2}\P[X \in S] = P(\{\omega \in \Omega | X(\omega) \in S\})$$

If $E$ is discrete, we can define the **Probability Mass Function (*PMF*)** $p_X: E \rightarrow [0,1]$ of $X$:

$$\tag{B.1.3}\forall x \in E, p_X(x) = \P[X = x] = P(X^{-1}(\{x\}))$$

If $E \subseteq \R$, we can defin the **Cumulative Density Function (*CDF*)** $F_X: E \rightarrow [0,1]$ of $X$:

$$\tag{B.1.4}\forall x \in E, F_X(x) = \P[X \leq x] = P(X^{-1}((-\infty, x])))$$

If $F_X$ is differentiable, the **Probability Density Function (*PDF*)** $f_X: E \rightarrow [0,+\infty)$ is defined by:

$$\tag{B.1.5}f_X(x) = F_X'(x)$$

Which leads to the following relation:

$$\tag{B.1.6}\int_a^bf_X(x)dx = F_X(b) - F_X(a) = \P[X \leq b] - \P[X \leq a] = \P[x \in [a,b]] = P(X^{-1}([a,b]))$$

The **expectation** of a random variable $X$ is defined as:

$$\tag{B.1.7} \E[X] = \int_\Omega X(\omega)dP(\omega) = \int_0^1xdF_X(x) = \int_Exf_X(x)dx$$

Given a function $g : E \rightarrow F \subseteq \R$ bijective and continuously differentiable, and $Y = g(X)$, then

$$\tag{B.1.8}F_Y(y) = \P[Y \leq y] = \P[g(X) \leq g(x)] = \begin{cases}
    \P[X \leq x] = F_X(x) & \text{ if }g\text{ is increasing} \\
    \P[X \geq x] = 1 - F_X(x) & \text{ if }g\text{ is decreasing}
\end{cases}$$

Therefore:

$$f_Y(y) = \diff{F_Y}{y}(y) =\begin{cases}
    \displaystyle \diff{F_X}{y}(y) = \diff{x}{y}(y)\diff{F_X}{x}(x) = (g^{-1})'(y)f_X(x) & \text{ if }g\text{ is increasing} \\
    \displaystyle \diff{1-F_X}{y}(y) = \diff{x}{y}(y)\diff{1-F_X}{x}(x) = -(g^{-1})'(y)f_X(x) & \text{ if }g\text{ is decreasing}
\end{cases}$$

Which means:

$$\tag{B.1.9}f_Y(y) = |(g^{-1})'(y)|f_X(x)$$

Combining this result with the change of variable $y = g(x)$ yields the **Law of the Unconscious Statistician**:

$$\tag{B.1.10}\E[g(X)] = \int_0^1ydF_Y(y) = \int_Eg(x)f_X(x)dx$$

**Expectation properties**: given $X, Y$ random variables and $a, b \in \R$:

$$\tag{B.1.11} \begin{cases}
    X \overset{a.s.}{=} Y \implies \E[X] = \E[Y] \\
    X \overset{a.s.}{=} c \in E \implies \E[X] = c \\
    \E[\E[X]] = \E[X] \\
    \E[aX + bY] = a\E[X] + b\E[Y]
\end{cases}$$

**Jensen's Inequality**: given a function $\phi$ convex:

$$\tag{B.1.12}\varphi(\E[X]) \leq \E[\varphi(X)]$$

### Variance and Independence

The **variance** of a random variable $X$ is defined by:

$$\tag{B.1.13}\begin{aligned}\V[X] &= \E[(X - \E[X])^2] \\&= \E[X^2 - 2X\E[X] + \E[X]^2] \\&= \E[X^2] - 2\E[X]\E[X] + \E[X]^2 \\&= \E[X^2] - \E[X]^2\end{aligned}$$

**Variance properties**: with $X$ a random variable and $a \in \R$:

$$\tag{B.1.14}\begin{cases}
    \V[X + a] = \V[X] \\
    \V[aX] = a^2\V[X]
\end{cases}$$

The **covariance** of two random variables $X$ and $Y$ is defined by:

$$\tag{B.1.15}\begin{aligned}\Cov(X, Y) &= \E[(X - \E[X])(Y - \E[Y])] \\&= \E[XY] - \E[X]\E[Y] \\&= \E[X(Y - \E[Y])] \\&= \E[(X - \E[X])Y]\end{aligned}$$

In particular,

$$\tag{B.1.16}\Cov(X, X) = \V[X]$$

**Covariance properties**: with $X, Y, Z$ random variables, and $a, b, c \in \R$:

$$\tag{B.1.17}\begin{cases}
    \Cov(X, a) = 0 \\
    \Cov(X, Y) = \Cov(Y, X) \\
    \Cov(aX + bY, cZ) = ac\Cov(X, Z) + bc\Cov(Y, Z)
\end{cases}$$

In particular,

$$\tag{B.1.18}\V[X\pm Y] = \V[X] + \V[Y] \pm 2\Cov(X, Y)$$

The **covariance matrix** of a vector random variable $\XX \in \R^d$ is defined as:

$$\tag{B.1.19}\CCov(\XX) = \E\left[(\XX - \E[\XX])(\XX - \E[\XX])^\top\right] \in \R^{d\times d}$$

**Covariance matrix properties**: with $A \in \R^{k\times d}, B \in \R^k$:

$$\tag{B.1.20} \CCov(A\XX + B) = \CCov(A\XX) = A\CCov(\XX)A^\top \in \R^{k\times k}$$

Two random variables $X, Y$ are **independent** ($X \perp Y$) if

$$\tag{B.1.21}X \perp Y \iff F_{X,Y}(x, y) = F_X(x)F_Y(y) \iff f_{X,Y}(x,y) = f_X(x)f_Y(y)$$

Therefore, if $X \perp Y$, then

$$\tag{B.1.22} \E[XY] = \E[X]\E[Y]$$

But **the converse isn't true in general**. It follows that:

$$\tag{B.1.23}\begin{cases}
    \Cov(X, Y) = 0 \\
    \V[X+Y] = \V[X] + \V[Y]
\end{cases}$$

### Tests

The *p-value* is formally defined as:

$$\tag{B.1.24} \pval = \P[\mathcal{D} \geq T_n | T_n]$$

Where $T_n$ is the test statistics and $\mathcal{D}$ is the test distribution of $T_n$.

## Probability

### Conditional Probability

Given two events $A, B \subset \Omega$, the **conditional probabilty** of $A$ knowing $B$ is defined as:

$$\tag{B.2.1}P(A|B) = \frac{P(A \cap B)}{P(B)} \leq 1$$

The **Bayes Formula** gives us an expression of this probability:

$$\tag{B.2.2}P(A|B) = \frac{P(A)P(B|A)}{P(B)}$$

$A$ and $B$ are **independent** if

$$\tag{B.2.3}P(A\cap B) = P(A)P(B)$$

Which leads to:

$$\tag{B.2.4}P(A|B) = P(A), P(B|A) = P(B)$$
