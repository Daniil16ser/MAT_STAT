from typing import Callable
import numpy as np
from numpy.random import RandomState
from scipy.stats import moment, bootstrap, norm

# Initialize random generator
_RANDOM_SEED = 10
_RNG = RandomState(_RANDOM_SEED)

# Confidence parameter
_CONFIDENCE = 0.95

class DataGenerator:
    _param: np.float64
    _source: Callable

    def __init__(
        self,
        param: np.float64,
    ) -> None:
        self._param = param
        self._source = lambda size : _RNG.uniform(param, 2 * param, size)

    def fetch(
        self,
        size: int,
    ) -> np.ndarray:
        return self._source(size)

# Generate observations
generator = DataGenerator(param=np.float64(6))
observations = generator.fetch(100)

# Store computed bounds
bounds = dict()

# Method 1: exact interval
bounds["exact"] = (
    observations.max() / (1 + np.pow((1 + _CONFIDENCE) / 2, 0.01)),
    observations.max() / (1 + np.pow((1 - _CONFIDENCE) / 2, 0.01))
)

# Method 2: large-sample approximation
bounds["approx"] = (
    2/3 * (observations.mean() - norm.ppf((1 + 0.95) / 2) * moment(observations, order=2) / 10),
    2/3 * (observations.mean() - norm.ppf((1 - 0.95) / 2) * moment(observations, order=2) / 10)
)

# Method 3: resampling approach
resample_result = bootstrap(
    observations.reshape(1, -1),
    lambda x : 2/3 * np.mean(x),
    n_resamples=1000,
    confidence_level=0.95,
    method='basic',
    rng=_RNG
)

bounds["resample"] = (
    resample_result.confidence_interval.low,
    resample_result.confidence_interval.high
)

print("\n" + "="*60)
print(" CONFIDENCE INTERVAL ESTIMATES ".center(60, "="))
print("="*60)

for method_name, (lower_val, upper_val) in bounds.items():
    print(f"\n[{method_name.upper()}]")
    print(f"  Lower boundary: {lower_val:.8f}")
    print(f"  Upper boundary: {upper_val:.8f}")
    print(f"  Interval width: {upper_val - lower_val:.8f}")

print("\n" + "="*60)