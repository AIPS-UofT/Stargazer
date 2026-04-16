# Skill: Multi-Planet Detection via Iterative Residual Analysis

## Description
Use this skill to determine the correct number of planets and detect additional planets after fitting the first. Multi-planet tasks are the most common cause of planet count mismatch (ok_count failure).

## When to Activate
- After fitting a first planet and computing residuals
- When task difficulty is ≥ 4 (higher probability of multiple planets)
- When residual RMS after first planet fit is still high (> 2× the noise floor)
- When ok_count fails in evaluation

## Instructions

### Step 1: After fitting planet 1, always check residuals
```python
# Subtract planet 1 model from data
residuals_1 = rvs - planet1_model(times, *params1)
rms_1 = np.sqrt(np.mean(residuals_1**2))

# Compare to noise floor (median sigma)
sigma_median = np.median(sigmas)
print(f"Residual RMS: {rms_1:.2f} m/s, noise floor: {sigma_median:.2f} m/s")
```

If `rms_1 > 2 × sigma_median`, a second planet likely exists. Proceed to Step 2.
If `rms_1 ≤ 1.5 × sigma_median`, the system is likely single-planet.

### Step 2: Run periodogram on residuals
```python
from astropy.timeseries import LombScargle
ls = LombScargle(times, residuals_1, sigmas)
freq, power = ls.autopower(minimum_frequency=1/1000, maximum_frequency=1/0.5)
periods = 1/freq

# Find peak
best_P2 = periods[np.argmax(power)]
```
Repeat alias checks from rv_period_search skill for the second period.

### Step 3: Use BIC to decide 1 vs 2 planets
```python
from scipy.optimize import minimize

def bic(n_params, n_data, rms):
    """Bayesian Information Criterion."""
    chi2 = n_data * (rms / sigma_median)**2
    return chi2 + n_params * np.log(n_data)

# 1-planet model: 5 free params per planet + 1 gamma
bic_1 = bic(6, len(times), rms_1)

# Fit 2-planet model
# ... fit params2, compute rms_2 ...
bic_2 = bic(11, len(times), rms_2)

delta_bic = bic_1 - bic_2
if delta_bic > 10:
    print("Strong evidence for 2nd planet (ΔBIC > 10)")
elif delta_bic > 6:
    print("Moderate evidence for 2nd planet")
else:
    print("No strong evidence for 2nd planet — submit 1 planet")
```

### Step 4: Fit 2-planet model simultaneously
After finding approximate P2 from residuals, fit both planets together:
```python
# Joint 2-planet RV model: sum of two Keplerian signals + gamma
def two_planet_rv(t, P1, K1, e1, w1, M01, P2, K2, e2, w2, M02, gamma):
    return kepler_rv(t, P1, K1, e1, w1, M01, 0) + \
           kepler_rv(t, P2, K2, e2, w2, M02, 0) + gamma
```
Use differential_evolution with bounds for both planets.

### Step 5: Check for a 3rd planet in 2-planet residuals
Repeat Steps 1-3 on residuals after 2-planet subtraction. Stop adding planets when:
- Residual RMS ≈ noise floor, OR
- ΔBIC < 6 (no strong evidence for additional planet)

### Decision Rule: When to submit multi-planet vs single-planet
```
residual_rms after N planets:
  > 3× noise: add another planet
  2-3× noise: check ΔBIC, add if > 6
  < 2× noise: stop, submit N planets
```

### Common Mistakes to Avoid
- **DO NOT** assume single-planet just because one planet fits "okay"
- **DO NOT** submit 0 planets — if you detect any signal at all, submit at least 1
- **DO NOT** skip BIC comparison — blindly adding planets inflates ok_count failures
- **DO NOT** fit planets sequentially without re-optimizing jointly — parameters are correlated
- If time is running out, submit the best single-planet solution rather than nothing
