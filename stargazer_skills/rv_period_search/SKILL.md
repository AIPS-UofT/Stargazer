# Skill: RV Period Search & Alias Detection

## Description
Use this skill when searching for planetary orbital periods in radial velocity (RV) data, especially to avoid picking up 1-day aliases or harmonics instead of the true period.

## When to Activate
- Running a periodogram (GLS, Lomb-Scargle) on RV data
- Choosing between multiple peaks in a periodogram
- Unsure whether a period candidate is real or an alias
- After fitting a planet and residual RMS is unexpectedly high

## Instructions

### Step 1: Always detrend first
Before running any periodogram, remove linear (or polynomial) RV drift:
```python
from numpy.polynomial import polynomial as P
trend_coeffs = np.polyfit(times, rvs, 1)
rvs_detrended = rvs - np.polyval(trend_coeffs, times)
```
Failure to detrend causes the periodogram to show spurious long-period peaks that mask the real signal.

### Step 2: Check if candidate periods exceed the observation baseline
If any periodogram peak has period > (max_time - min_time), it is almost certainly a trend alias, not a real planet. Reject it immediately.

### Step 3: Identify the 1-day alias family
For any candidate period P_cand, compute its aliases:
```
alias_1 = 1 / (1/P_cand - 1)   # +1 day alias
alias_2 = 1 / (1/P_cand + 1)   # -1 day alias
alias_half = P_cand / 2         # sub-harmonic
alias_double = P_cand * 2       # harmonic
```
If a strong secondary peak lies at one of these aliases, the two peaks are related — pick the one with higher GLS power AND physically shorter period when both have similar power.

### Step 4: Validate with phase-folding
For the top 3 candidate periods, phase-fold the detrended RVs and visually confirm the signal is coherent:
```python
phases = (times % P_cand) / P_cand
# Plot phases vs rvs_detrended — should show a smooth sinusoidal pattern
```
Random scatter = wrong period. Smooth curve = correct period.

### Step 5: Narrow refinement
After identifying the rough period, refine with a fine grid search around ±5% of the candidate:
```python
P_fine = np.linspace(P_cand * 0.95, P_cand * 1.05, 1000)
# Re-run GLS on fine grid and take the maximum
```

### Common Mistakes to Avoid
- **DO NOT** pick a period > observation baseline as the primary signal
- **DO NOT** assume the highest GLS peak is correct without checking aliases
- **DO NOT** use T0 = 0 as reference epoch — always use times[0]
- After detrending, re-run the periodogram — the best peak may shift

## Example: Correct alias rejection
```
Observation baseline: 30 days
Periodogram peaks: 45d (power 0.82), 1.52d (power 0.71), 3.04d (power 0.65)

Analysis:
- 45d > 30d baseline → reject as trend alias
- 3.04d ≈ 2 × 1.52d → harmonic relationship
- Choose 1.52d as true period (shorter, physically more common for hot Jupiters)
- Phase-fold at 1.52d → confirm smooth RV curve
```
