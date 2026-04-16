"""Tests for reproducibility and seed functionality."""

import numpy as np
import pytest

from stargazer.seed_utils import set_global_seed
from stargazer.task_factory import TaskFactory
from stargazer.env import RvEnv
from stargazer.priors import sample_periods_with_resonances, sample_planets
from stargazer.config import StarParams


def test_set_global_seed():
    """Test that set_global_seed sets all RNG states."""
    set_global_seed(42)

    # Test Python's random module
    import random
    val1 = random.random()

    set_global_seed(42)
    val2 = random.random()
    assert val1 == val2, "Python random module should be seeded"

    # Test NumPy's legacy random
    set_global_seed(42)
    arr1 = np.random.random(5)

    set_global_seed(42)
    arr2 = np.random.random(5)
    np.testing.assert_array_equal(arr1, arr2, err_msg="NumPy legacy random should be seeded")


def test_task_factory_reproducibility():
    """Test that TaskFactory produces identical tasks with same seed."""
    factory = TaskFactory()

    # Generate two tasks with the same seed
    task1 = factory.sample(seed=42)
    task2 = factory.sample(seed=42)

    # Check that key properties are identical
    assert task1.task_id == task2.task_id
    assert len(task1.config.planets) == len(task2.config.planets)

    # Check planet parameters
    for p1, p2 in zip(task1.config.planets, task2.config.planets):
        assert p1.P_days == p2.P_days
        assert p1.m_sin_i_mjup == p2.m_sin_i_mjup
        assert p1.e == p2.e
        assert p1.inc_rad == p2.inc_rad
        assert p1.Omega_rad == p2.Omega_rad
        assert p1.omega_rad == p2.omega_rad
        assert p1.l_rad == p2.l_rad

    # Check observations
    np.testing.assert_array_equal(task1.observations.times_days, task2.observations.times_days)
    np.testing.assert_array_equal(task1.observations.rvs_ms, task2.observations.rvs_ms)
    np.testing.assert_array_equal(task1.observations.sigmas_ms, task2.observations.sigmas_ms)

    # Check noise parameters
    assert task1.config.noise.sigma_white_ms == task2.config.noise.sigma_white_ms
    assert task1.config.noise.sigma_jitter_ms == task2.config.noise.sigma_jitter_ms

    # Check GP parameters if present
    if task1.config.noise.gp.use_gp and task2.config.noise.gp.use_gp:
        assert task1.config.noise.gp.sigma_ms == task2.config.noise.gp.sigma_ms
        assert task1.config.noise.gp.period_days == task2.config.noise.gp.period_days
        assert task1.config.noise.gp.Q0 == task2.config.noise.gp.Q0
        assert task1.config.noise.gp.dQ == task2.config.noise.gp.dQ
        assert task1.config.noise.gp.f == task2.config.noise.gp.f


def test_task_factory_different_seeds():
    """Test that TaskFactory produces different tasks with different seeds."""
    factory = TaskFactory()

    task1 = factory.sample(seed=42)
    task2 = factory.sample(seed=123)

    # Tasks should be different
    # At least one planet parameter should differ
    planets_differ = False
    if len(task1.config.planets) == len(task2.config.planets):
        for p1, p2 in zip(task1.config.planets, task2.config.planets):
            if (p1.P_days != p2.P_days or
                p1.m_sin_i_mjup != p2.m_sin_i_mjup or
                p1.e != p2.e):
                planets_differ = True
                break
    else:
        planets_differ = True

    assert planets_differ, "Different seeds should produce different tasks"


def test_sample_periods_with_resonances_reproducibility():
    """Test that period sampling with resonances is reproducible."""
    rng1 = np.random.default_rng(42)
    periods1 = sample_periods_with_resonances(3, 2.0, 300.0, rng=rng1)

    rng2 = np.random.default_rng(42)
    periods2 = sample_periods_with_resonances(3, 2.0, 300.0, rng=rng2)

    np.testing.assert_array_equal(periods1, periods2)


def test_sample_planets_reproducibility():
    """Test that planet sampling is reproducible."""
    star = StarParams(M_star_sun=1.0, gamma_ms=0.0)

    rng1 = np.random.default_rng(42)
    planets1 = sample_planets(2, star, rng=rng1)

    rng2 = np.random.default_rng(42)
    planets2 = sample_planets(2, star, rng=rng2)

    assert len(planets1) == len(planets2)
    for p1, p2 in zip(planets1, planets2):
        assert p1.P_days == p2.P_days
        assert p1.m_sin_i_mjup == p2.m_sin_i_mjup
        assert p1.e == p2.e
        assert p1.inc_rad == p2.inc_rad


def test_env_reset_with_seed():
    """Test that RvEnv.reset() properly uses seed parameter."""
    factory = TaskFactory()

    # Create environment with task sampler
    env = RvEnv(task_sampler=factory.sample)

    # Reset with seed
    obs1, info1 = env.reset(seed=42)
    task_id1 = info1["task_id"]

    # Reset with same seed should give same task
    obs2, info2 = env.reset(seed=42)
    task_id2 = info2["task_id"]

    assert task_id1 == task_id2
    np.testing.assert_array_equal(obs1["times_days"], obs2["times_days"])
    np.testing.assert_array_equal(obs1["rvs_ms"], obs2["rvs_ms"])

    # Reset with different seed should give different task
    obs3, info3 = env.reset(seed=123)
    task_id3 = info3["task_id"]

    # Task IDs should be different (or at least RVs should differ)
    if task_id1 == task_id3:
        # If IDs happen to match (unlikely), check that RVs differ
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(obs1["rvs_ms"], obs3["rvs_ms"])


def test_global_seed_with_task_generation():
    """Test that set_global_seed works with task generation."""
    set_global_seed(42)
    factory1 = TaskFactory()
    task1 = factory1.sample(seed=100)

    set_global_seed(42)
    factory2 = TaskFactory()
    task2 = factory2.sample(seed=100)

    # With global seed set, tasks should be identical
    assert len(task1.config.planets) == len(task2.config.planets)
    np.testing.assert_array_equal(task1.observations.times_days, task2.observations.times_days)
    np.testing.assert_array_equal(task1.observations.rvs_ms, task2.observations.rvs_ms)


def test_seed_stored_in_task_metadata():
    """Test that seed is stored in task metadata."""
    factory = TaskFactory()
    task = factory.sample(seed=42)

    assert "seed" in task.meta
    assert task.meta["seed"] == 42


if __name__ == "__main__":
    # Run tests manually for debugging
    test_set_global_seed()
    print("✓ test_set_global_seed passed")

    test_task_factory_reproducibility()
    print("✓ test_task_factory_reproducibility passed")

    test_task_factory_different_seeds()
    print("✓ test_task_factory_different_seeds passed")

    test_sample_periods_with_resonances_reproducibility()
    print("✓ test_sample_periods_with_resonances_reproducibility passed")

    test_sample_planets_reproducibility()
    print("✓ test_sample_planets_reproducibility passed")

    test_env_reset_with_seed()
    print("✓ test_env_reset_with_seed passed")

    test_global_seed_with_task_generation()
    print("✓ test_global_seed_with_task_generation passed")

    test_seed_stored_in_task_metadata()
    print("✓ test_seed_stored_in_task_metadata passed")

    print("\nAll tests passed!")
