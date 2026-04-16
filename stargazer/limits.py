"""Shared benchmark limits used across runners and agents."""

SYNTHETIC_MAX_PLANETS = 4
REAL_MAX_PLANETS = 7

# The published dataset includes a 7-planet real task, so submission defaults
# must not silently cap agents below that value.
DEFAULT_SUBMISSION_MAX_PLANETS = REAL_MAX_PLANETS
