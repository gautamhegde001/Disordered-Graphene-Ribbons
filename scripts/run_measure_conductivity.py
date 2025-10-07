import logging
from pkg.measure_conductivity import measure_conductivity_once

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    W = 2 # Setting width
    params = {
        'V': 0.1,
        'L': 30,
        'W': W,
        't_a': [1.0] * W,
        't_b': [1.0] * W,
        't_c': [1.0] * (W-1),
        'E': 0.5,
        'rng_seed': 42,
        }

    result = measure_conductivity_once(params)
    print("Hello")
    print("log(T) =", result)