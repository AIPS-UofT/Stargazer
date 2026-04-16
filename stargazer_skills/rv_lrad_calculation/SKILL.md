# Skill: Mean Longitude (l_rad) Calculation

## Description
Use this skill to correctly compute the mean longitude l_rad at the reference epoch t_ref = times[0], required for the submit_action call. This is one of the most common sources of match_score failure.

## When to Activate
- Preparing to call submit_action
- After fitting orbital parameters P, K, e, omega, M0
- When match_score fails specifically on phase/epoch alignment

## Instructions

### The correct formula
```
l_rad = (Omega_rad + omega_rad + M0_at_tref) mod 2π
```

Where:
- `Omega_rad`: longitude of ascending node (often 0 for RV-only fits)
- `omega_rad`: argument of periastron (ω), in radians
- `M0_at_tref`: mean anomaly at t_ref = times[0]

For RV-only fits (no astrometry), Omega_rad = 0, so:
```python
l_rad = (omega_rad + M0_at_tref) % (2 * np.pi)
```

### Computing M0 at t_ref
If your optimizer gives you M0 at some time t_fit, convert to t_ref = times[0]:
```python
t_ref = times[0]
# If M0 was defined at t_fit:
M0_at_tref = (M0_at_tfit + 2*np.pi/P * (t_ref - t_fit)) % (2*np.pi)
```

If you defined M0 directly at times[0] during fitting (recommended), then:
```python
M0_at_tref = M0  # from optimizer
l_rad = (omega_rad + M0_at_tref) % (2 * np.pi)
```

### Full example
```python
# After fitting: P, K, e, omega, M0 (all defined at t_ref = times[0])
import numpy as np

t_ref = times[0]
Omega_rad = 0.0  # RV-only assumption

# M0 is mean anomaly at t_ref from your Keplerian fit
l_rad = (Omega_rad + omega + M0) % (2 * np.pi)

print(f"l_rad = {l_rad:.6f} rad")
```

### Sanity checks before submitting
1. l_rad must be in [0, 2π]: `assert 0 <= l_rad <= 2*np.pi`
2. Verify by computing the RV at t_ref using your model — it should match the first observed RV closely
3. If you have multiple planets, compute l_rad independently for each

### Critical: Always use times[0] as reference epoch
The benchmark evaluates l_rad at t_ref = times[0]. Using any other epoch (e.g., t=0, JD 2450000, or the midpoint of observations) will cause match_score failure even if all other parameters are correct.

```python
# WRONG:
t_ref = 0.0
t_ref = np.mean(times)
t_ref = 2450000.0

# CORRECT:
t_ref = times[0]  # Always the first observation time
```

### Common Mistakes to Avoid
- **DO NOT** use t=0 or any epoch other than times[0]
- **DO NOT** confuse M0 (mean anomaly) with true anomaly ν
- **DO NOT** forget the mod 2π — l_rad must wrap to [0, 2π]
- **DO NOT** confuse ω (omega, argument of periastron) with Ω (Omega, longitude of ascending node)
- If l_rad from different fitting runs differs by ~π, you likely have a phase ambiguity — check omega sign convention
