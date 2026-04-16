# Skill: RV Submission Strategy & Timing

## Description
Use this skill to decide when to call submit_action and how to avoid analysis paralysis (running out of time without submitting). No-submission is the worst possible outcome (reward = 0).

## When to Activate
- After achieving a Keplerian fit with any reasonable RMS
- When approaching tool call budget (>60% used)
- When iterative refinement is not improving results
- Before attempting any MCMC or advanced analysis

## Instructions

### The Golden Rule: Submit Early, Refine Later
Submit a baseline solution AS SOON AS you have a fit with RMS below a reasonable threshold. You can always submit again with a better solution — only the best submission counts.

```
Submit if ANY of these are true:
  ✓ RMS < 30 m/s (for typical tasks)
  ✓ You have spent > 8 tool calls without submitting
  ✓ Your current fit matches the period and K within 10%
  ✓ You are about to try something risky (MCMC, re-fit from scratch)
```

### Submission Checklist
Before calling submit_action, verify:
```python
# 1. Parameters are physically reasonable
assert 0.1 < P < 10000, "Period out of range"
assert K > 0, "Semi-amplitude must be positive"
assert 0.0 <= e < 1.0, "Eccentricity must be in [0, 1)"
assert 0.0 <= l_rad <= 2*np.pi, "Mean longitude must be in [0, 2π]"

# 2. l_rad computed at t_ref = times[0]
t_ref = times[0]
l_rad = (omega + M0) % (2 * np.pi)  # see rv_lrad_calculation skill

# 3. Mass from K and orbital parameters
# m_sin_i_mjup estimated from K, P, M_star
```

### Time Budget Management
```
Total tool calls budget: N (typically 30-50)

Phase 1 — Exploration (calls 1-8):
  - Load data, compute periodogram, identify period candidates
  - Quick sine fit to get rough K

Phase 2 — First Fit (calls 9-16):
  - Full Keplerian fit (differential evolution)
  - Compute l_rad correctly
  → SUBMIT BASELINE SOLUTION HERE (call ~15)

Phase 3 — Refinement (calls 17-25):
  - Check residuals for additional planets
  - Refine eccentricity
  - Try alternative periods if RMS is poor
  → SUBMIT IMPROVED SOLUTION IF BETTER

Phase 4 — Polish (calls 26+):
  - MCMC or MCMC-like refinement only if time permits
  - DO NOT start new analysis from scratch
  → Submit final best solution before budget exhausted
```

### When to Stop Refining
Stop trying to improve and submit your current best if:
- You have already submitted twice with similar RMS
- The last 3 fitting attempts all gave the same RMS within 5%
- You have used > 75% of your tool call budget
- MCMC or advanced fitting is taking too many calls

### Emergency Protocol: Low Tool Call Budget
If you realize you are close to the limit with no submission:
```python
# Immediately fit a simple circular Keplerian (e=0)
# using the best period from periodogram
# Submit even a rough solution — reward > 0 is better than 0
quick_params = fit_circular(times, rvs, P_best)
submit_action(planets=[{
    'P_days': P_best,
    'K_ms': K_estimate,
    'e': 0.0,
    'omega_rad': 0.0,
    'l_rad': 0.0,  # rough estimate
    'm_sin_i_mjup': estimate_mass(K_estimate, P_best, M_star)
}])
```

### Common Mistakes to Avoid
- **DO NOT** wait for a "perfect" solution before first submission
- **DO NOT** submit 0 planets — always submit at least 1 if any signal is detected
- **DO NOT** spend >50% of budget on MCMC without a prior baseline submission
- **DO NOT** re-run differential evolution from scratch more than 2 times
- **DO NOT** start trying to detect planet 2 before submitting a planet 1 solution
- If RMS is stubbornly high after 3 re-fits: change the period, not just the optimizer
