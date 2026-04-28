from typing import Callable
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from scipy.stats import norm, chi2, bootstrap

SEED = 6
GENERATOR = RandomState(MT19937(SeedSequence(SEED)))

BETA = 0.95
ALPHA = 1 - BETA
Z = norm.ppf(1 - ALPHA/2)

THETA_TRUE = 3.0
N = 100
B = 10000


class ParetoModel:
    _theta: np.float64
    _generator: Callable

    def __init__(self, theta: np.float64) -> None:
        self._theta = theta
        self._generator = lambda size: (1 - GENERATOR.uniform(0, 1, size)) ** (-1/(theta - 1))

    def get_sample(self, size: int) -> np.ndarray:
        return self._generator(size)


def mle_theta(sample):
    return 1 + len(sample) / np.sum(np.log(sample))


pareto_model = ParetoModel(theta=np.float64(THETA_TRUE))
sample = pareto_model.get_sample(N)

theta_hat = mle_theta(sample)

intervals = dict()

# асимптотический интервал для theta
intervals["asymptotic"] = (
    theta_hat - Z * (theta_hat - 1) / np.sqrt(N),
    theta_hat + Z * (theta_hat - 1) / np.sqrt(N)
)

# точный интервал для theta через хи-квадрат
sum_log = np.sum(np.log(sample))
intervals["exact"] = (
    1 + chi2.ppf(ALPHA/2, 2*N) / (2 * sum_log),
    1 + chi2.ppf(1 - ALPHA/2, 2*N) / (2 * sum_log)
)

# непараметрический бутстрап
bootstrap_result = bootstrap(
    sample.reshape(1, -1),
    lambda x: 1 + len(x) / np.sum(np.log(x)),
    n_resamples=B,
    confidence_level=BETA,
    method='basic',
    rng=GENERATOR
)

intervals["nonparametric_bootstrap"] = (
    bootstrap_result.confidence_interval.low,
    bootstrap_result.confidence_interval.high
)

# параметрический бутстрап
bootstrap_theta = []
for _ in range(B):
    u = GENERATOR.uniform(0, 1, N)
    boot_sample = (1 - u) ** (-1/(theta_hat - 1))
    bootstrap_theta.append(mle_theta(boot_sample))

bootstrap_theta = np.array(bootstrap_theta)
intervals["parametric_bootstrap"] = (
    np.percentile(bootstrap_theta, 100 * ALPHA/2),
    np.percentile(bootstrap_theta, 100 * (1 - ALPHA/2))
)

print("=" * 70)
print(f"{'Метод':<30} {'Нижняя':<12} {'Верхняя':<12} {'Длина':<10}")
print("-" * 70)
for name, (low, high) in intervals.items():
    length = high - low
    print(f"{name:<30} {low:<12.6f} {high:<12.6f} {length:<10.6f}")
print("=" * 70)