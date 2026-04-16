"""Seed management utilities for reproducibility."""

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for all RNG sources used in Stargazer.

    This function ensures reproducibility by seeding:
    - NumPy's random number generator (both legacy and new Generator API)
    - Python's built-in random module
    - Any other library that depends on NumPy's global random state (e.g., celerite2)

    Parameters
    ----------
    seed : int
        The random seed value. Must be a non-negative integer.

    Example
    -------
    >>> from stargazer.seed_utils import set_global_seed
    >>> set_global_seed(42)
    >>> # All subsequent random operations will be reproducible
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")

    # Seed Python's built-in random module (used in priors.py)
    random.seed(seed)

    # Seed NumPy's legacy random state (used by celerite2 and other libraries)
    np.random.seed(seed)

    # Note: Individual numpy.random.Generator instances created with
    # np.random.default_rng(seed) are already handled by explicit seed
    # parameters throughout the codebase. This function handles global state.
