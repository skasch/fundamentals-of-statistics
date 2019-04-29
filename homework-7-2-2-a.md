# Student's T Test

## Deriving the Student's T Test from the Likelihood Ratio

Let $X_1, \ldots, X_n \overset{\text{i.i.d}}\sim \mathcal{N}(\mu_1, \sigma_1^2)$. Because the test is one-sided, I will consider the two alternatives:

$$\begin{array}{ccl} H_0 & : & \mu_1 = \mu_0 = 5 \\ H_1 & : & \mu_1 > 5\end{array}$$

$\mu_1$ and $\sigma^2_1$ are both unknown, therefore

$$\Theta_0 = \{(\mu, \sigma^2) \in \R \times (0, +\infty), \mu = \mu_0\}$$

The likelihood ratio test is defined as:

$$\tag{1}T_n' = 2(\ell_n(\hat\mu^{MLE}_n, \widehat{\sigma^2}^{MLE}_n) - \ell_n(\hat\mu^{0}_n, \widehat{\sigma^2}^0_n))$$

Where

$$(\hat\mu^0_n, \widehat{\sigma^2}^0_n) = \underset{(\mu, \sigma^2) \in \Theta_0}{\text{argmax}}\ell_n(\mu, \sigma^2)$$

We know that

$$\tag{2}\ell_n(\mu, \sigma^2) = -\frac{n}2\ln(2\pi) - \frac{n}2\ln(\sigma^2) - \frac1{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2$$

We are trying to maximize $\ell_n(\mu, \sigma^2)$ under the constraint $\mu = \mu_0$. Let us define $g(\mu, \sigma^2) = \mu - \mu_0$. Our constraint therefore is $g(\mu, \sigma^2) = 0$. Using the Lagrange multiplier:

$$\begin{aligned}\mathcal{L}(\mu, \sigma^2, \lambda) &= \ell_n(\mu, \sigma^2) + \lambda g(\mu, \sigma^2)
\\&= -\frac{n}2\ln(2\pi) - \frac{n}2\ln(\sigma^2) - \frac1{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2 + \lambda(\mu - \mu_0)\end{aligned}$$

We want to find the solution of $\nabla\mathcal{L}(\mu, \sigma^2, \lambda) = 0$.

$$\frac{\partial \mathcal{L}}{\partial \mu}(\mu, \sigma^2, \lambda) = \frac{1}{\sigma^2}\sum_{i=1}^n(X_i - \mu) + \lambda$$

$$\frac{\partial \mathcal{L}}{\partial \sigma^2}(\mu, \sigma^2, \lambda) = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n(X_i - \mu)^2$$

$$\frac{\partial \mathcal{L}}{\partial \lambda}(\mu, \sigma^2, \lambda) = \mu - \mu_0$$

We are only interested in finding $\mu$ and $\sigma^2$; using the last two equations, and equating them with $0$, leads to:

$$\tag{3}\hat\mu^0_n = \mu_0$$

And for $\widehat{\sigma^2}^0_n$:

$$-\frac{n}{2\widehat{\sigma^2}^0_n} + \frac{1}{2\left(\widehat{\sigma^2}^0_n\right)^2}\sum_{i=1}^n(X_i - \mu_0)^2 = 0$$

$$-\frac{n}{2\left(\widehat{\sigma^2}^0_n\right)^2}\left(\widehat{\sigma^2}^0_n - \frac1n\sum_{i=1}^n(X_i-\mu_0)^2\right) = 0$$

$$\tag{4}\widehat{\sigma^2}^0_n = \frac1n\sum_{i=1}^n(X_i-\mu_0)^2$$

We also know the forms of the maximum likelihood estimators:

$$\tag{5}\hat\mu_n^{MLE} = \bar{X}_n$$

$$\tag{6}\widehat{\sigma^2}_n^{MLE} = \frac1n\sum_{i=1}^n(X_i - \bar{X}_n)^2$$

Injecting $(5)$ and $(6)$ into $(2)$ gives us:

$$\ell_n(\hat\mu_n^{MLE}, \widehat{\sigma^2}^{MLE}_n) = -\frac n2\ln(2\pi) - \frac{n}2\ln(\widehat{\sigma^2}^{MLE}_n) - \frac12\frac{1}{\displaystyle \frac1n\sum_{i=1}^n(X_i - \bar{X}_n)^2}\sum_{i=1}^n(X_i - \bar{X}_n)^2$$

$$\tag{7}\ell_n(\hat\mu_n^{MLE}, \widehat{\sigma^2}^{MLE}_n) = -\frac n2\ln(2\pi) - \frac{n}2\ln(\widehat{\sigma^2}^{MLE}_n) - \frac{n}2$$

Similarly, by injecting $(3)$ and $(4)$ into $(2)$:

$$\ell_n(\hat\mu_n^0, \widehat{\sigma^2}^0_n) = -\frac n2\ln(2\pi) - \frac{n}2\ln(\widehat{\sigma^2}^0_n) - \frac12\frac{1}{\displaystyle \frac1n\sum_{i=1}^n(X_i - \mu_0)^2}\sum_{i=1}^n(X_i - \mu_0)^2$$

$$\tag{8}\ell_n(\hat\mu_n^0, \widehat{\sigma^2}^0_n) = = -\frac{n}2\ln(2\pi) - \frac{n}2\ln(\widehat{\sigma^2}^0) - \frac{n}2$$

Therefore, our test is, using $(1)$:

$$\begin{aligned}T_n' &= 2(\ell_n(\hat\mu^{MLE}_n, \widehat{\sigma^2}^{MLE}_n) - \ell_n(\hat\mu^{0}_n, \widehat{\sigma^2}^0_n))\\
&= 2\left(-\frac n2\ln(2\pi) - \frac{n}2\ln(\widehat{\sigma^2}^{MLE}_n) - \frac{n}2 - \left(-\frac{n}2\ln(2\pi) - \frac{n}2\ln(\widehat{\sigma^2}^0) - \frac{n}2\right)\right)\\
&= n\left(-\ln(\widehat{\sigma^2}^{MLE}_n) +\ln(\widehat{\sigma^2}^0_n) \right)\end{aligned}$$


$$\tag{9} T_n' = n\ln\left(\frac{\widehat{\sigma^2}^0_n}{\widehat{\sigma^2}^{MLE}_n}\right)$$

We can simplify further this expression, by using the definition of $\widehat{\sigma^2}^0_n$:

$$\begin{aligned}\widehat{\sigma^2}^0_n &= \frac1n\sum_{i=1}^n(X_i - \mu_0)^2\\
&= \frac1n\sum_{i=1}^n(X_i - \bar{X}_n + \bar{X}_n - \mu_0)^2\\
&= \frac1n\sum_{i=1}^n\left((X_i - \bar{X}_n)^2 + 2(X_i - \bar{X}_n)(\bar{X}_n - \mu_0) + (\bar{X}_n - \mu_0)^2\right)\\
&= \frac1n\sum_{i=1}^n(X_i - \bar{X}_n)^2 + \frac{2}n(\bar{X}_n - \mu_0)\left(\sum_{i=1}^nX_i - n\bar{X}_n\right) + \frac{n}n(\bar{X}_n - \mu_0)^2\\
&= \widehat{\sigma^2}^{MLE}_n + \frac2{n}(\bar{X}_n-\mu_0)(n\bar{X}_n - n\bar{X}_n) + (\bar{X}_n - \mu_0)^2\end{aligned}$$

$$\tag{10}\widehat{\sigma^2}^0_n = \widehat{\sigma^2}^{MLE}_n + (\bar{X}_n - \mu_0)^2$$

Which gives us the final expression of our test:

$$\tag{11}T_n' =n\ln\left(1 + \frac{(\bar{X}_n - \mu_0)^2}{\widehat{\sigma^2}^{MLE}_n}\right)$$

## What I believe to be the wrong answer

If, instead of finding the optimal $\widehat{\sigma^2}^0_n$ in $\Theta_0$, we directly inject $\widehat{\sigma^2}^{MLE}_n$, we have instead the following expression:

$$\ell_n(\hat\mu_n^0, \widehat{\sigma^2}^{MLE}_n) = -\frac n2\ln(2\pi) - \frac{n}2\ln(\widehat{\sigma^2}^{MLE}_n) - \frac{n}2\frac{\widehat{\sigma^2}^0_n}{\widehat{\sigma^2}^{MLE}_n}$$

Which leads to the alternative expression for $T_n'$:

$$\begin{aligned}T_n' &= 2\left(- \frac{n}2\ln(\widehat{\sigma^2}^{MLE}_n) - \frac{n}2 + \frac{n}2\ln(\widehat{\sigma^2}^{MLE}_n) + \frac{n}2\frac{\widehat{\sigma^2}^0_n}{\widehat{\sigma^2}^{MLE}_n}\right)\\
&= n\left(\frac{\widehat{\sigma^2}^{0}_n}{\widehat{\sigma^2}^{MLE}_n} - 1\right)\\
&= n\frac{\widehat{\sigma^2}^{MLE}_n + (\bar{X}_n - \mu_0)^2 - \widehat{\sigma^2}^{MLE}_n}{\widehat{\sigma^2}^{MLE}_n}\\
&= n\frac{(\bar{X}_n - \mu_0)^2}{\widehat{\sigma^2}^{MLE}_n}\end{aligned}$$

And we find exactly what I think you are expecting; **but** this is, I think, an incorrect application of the likelihood ratio test, as we cannot consider that the maximizer $\widehat{\sigma^2}_n^0$ of $\ell_n$ restricted to $\Theta_0$ is $\widehat{\sigma^2}^{MLE}_n$; this case corresponds to the scenario where $\sigma^2_1$ is **known** and equal to $\widehat{\sigma^2}^{MLE}_n$.

It is of course possible that my reasoning is wrong, but I don't see where I might have missed something.