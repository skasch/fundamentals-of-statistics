import math
from typing import Callable

import numpy as np

size = 10 ** 3
n = 500
theta = 1
mu, tau = 0.1, 0.1


def inverse_phi(theta):
    def fun(t):
        return t ** (1 / theta)

    return fun


def theta_est(sample, size):
    return -size / (np.log(sample).sum(axis=1))


def random_sample(inv_pdf, size):
    return inv_pdf(np.random.uniform(0, 1, size=size))


def asymptotic(
    sample: np.ndarray,
    estimator: Callable,
    true: float,
    size: int,
    transform: Callable = lambda x: x,
):
    xs = transform(sample)
    xns = estimator(xs, size)
    return (math.sqrt(size) * (xns - true)).var()


def y(x):
    return (x <= 0.5).astype(int)


def thetay_est(sample, size):
    return -np.log(sample.mean(axis=1)) / np.log(2)


def iy(theta):
    return np.log(2) ** 2 / (np.power(2, theta) - 1)


inv_pdf = inverse_phi(theta)
sample = random_sample(inv_pdf, (size, size))
# print(1 - (y(random_sample(distribution, (size, n))).mean(axis=1) <= 0.49).mean())


def mu_est(sample, size=None):
    return sample.mean(axis=1)


def tau_est(sample, size=None):
    return (np.power(sample, 2).mean(axis=1)) - np.power(
        sample.mean(axis=1), 2
    )


def g_est(sample, size=None):
    return 2 * np.power(mu_est(sample), 2) + tau_est(sample)


sample = np.random.normal(mu, np.sqrt(tau), size=(size, size))
print(asymptotic(sample, g_est, 2 * mu ** 2 + tau, size))
print(16 * tau * mu ** 2 + 2 * tau ** 2)
