# Skill: Robust Keplerian Orbit Fitting

## Description
Use this skill to fit a full Keplerian orbit model to RV data. Ensures eccentricity is properly optimized, avoids local minima, and produces reliable parameter estimates including K, e, ω, and M0.

## When to Activate
- After identifying a candidate period from a periodogram
- When a sine-fit gives poor RMS or implausible eccentricity
- When submitted result has good match_score on period but fails ok_rms
- When fitting 1 or more planets to RV data

## Instructions

### Step 1: Never use sine-fit as final answer
Sine fitting assumes e=0 (circular orbit). Always follow up with a full Keplerian fit:
```python
from scipy.optimize import minimize, differential_evolution

def keplerian_rv(t, P, K, e, omega, M0, gamma):
    """Full Keplerian RV model."""
    M = (2 * np.pi / P) * (t - t[0]) + M0
    # Solve Kepler's equation iteratively
    E = M.copy()
    for _ in range(50):
        E = M + e * np.sin(E)
    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    return K * (np.cos(nu + omega) + e * np.cos(omega)) + gamma
```

### Step 2: Use global optimization for initial fit
Local optimizers (Nelder-Mead, L-BFGS-B) get stuck in local minima. Use differential evolution for initial parameter search:
```python
bounds = [
    (P_cand * 0.98, P_cand * 1.02),  # P: tight around periodogram peak
    (0.1, 3 * K_sine),                # K: semi-amplitude
    (0.0, 0.8),                        # e: eccentricity
    (0.0, 2*np.pi),                    # omega: argument of periastron
    (0.0, 2*np.pi),                    # M0: mean anomaly at t[0]
    (-50, 50),                          # gamma: systemic velocity offset
]
result = differential_evolution(residual_func, bounds, seed=42, maxiter=300)
```

### Step 3: Polish with local optimizer
After global search, refine with a local optimizer:
```python
from scipy.optimize import minimize
polished = minimize(residual_func, result.x, method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-8})
```

### Step 4: Validate eccentricity
After fitting, check:
- If e > 0.8: suspect numerical artifact, re-fit with e bounded to [0, 0.7]
- If RMS with e=0 (circular) is within 5% of best-fit RMS: submit circular orbit (e=0)
- If e > 0.05 and significantly reduces RMS: keep eccentric solution

### Step 5: Check for per-instrument offsets
If observations come from multiple instruments (different gamma per instrument), always fit independent gamma for each:
```python
# Check if 'instruments' field has multiple unique values
unique_insts = set(instruments)
if len(unique_insts) > 1:
    # Add gamma_i for each instrument to parameter vector
    pass
```

### Step 6: Report RMS and accept/reject
```python
residuals = rvs - model_rvs
rms = np.sqrt(np.mean(residuals**2))
print(f"RMS = {rms:.3f} m/s")
# If RMS > task threshold, try: different period, different e starting point
```

### Common Mistakes to Avoid
- **DO NOT** submit a sine-fit directly — always do full Keplerian
- **DO NOT** fix e=0 unless explicitly justified by data (low-e test)
- **DO NOT** use only local optimization — it will miss the global minimum
- **DO NOT** use t=0 as reference time — always use t_ref = times[0]
- If fitting multiple planets: fit them sequentially, subtracting each before fitting the next
