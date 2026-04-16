# Stargazer Agent Skills

Skills synthesized from failure analysis of 800 agent traces (8 models × 100 tasks).

## Skills Index

| Skill | Folder | Addresses |
|-------|--------|-----------|
| RV Period Search & Alias Detection | `rv_period_search/` | Period aliasing, trend artifacts, wrong period selection |
| Robust Keplerian Orbit Fitting | `rv_keplerian_fit/` | Local minima, e=0 defaults, incomplete fitting |
| Mean Longitude (l_rad) Calculation | `rv_lrad_calculation/` | Wrong epoch, l_rad formula errors, phase mismatch |
| Multi-Planet Detection | `rv_multiplanet_detection/` | Missing planet 2+, wrong planet count, BIC decision |
| Submission Strategy & Timing | `rv_submit_strategy/` | No-submission failures, analysis paralysis, time management |

## Failure Patterns Covered

From trace analysis (gpt-5.2 and claude-sonnet-4.5 failures):

| Pattern | Frequency | Skill(s) |
|---------|-----------|----------|
| Match score failure (wrong params) | 26+ cases | rv_period_search, rv_keplerian_fit, rv_lrad_calculation |
| RMS minimization failure | 13+ cases | rv_keplerian_fit |
| No submission (analysis paralysis) | 9 cases | rv_submit_strategy |
| Planet count mismatch | 8+ cases | rv_multiplanet_detection |
| Period aliasing | ~6 cases | rv_period_search |
| Eccentricity errors | ~5 cases | rv_keplerian_fit |
| l_rad / phase errors | ~4 cases | rv_lrad_calculation |
| Missed secular trend | ~3 cases | rv_period_search |

## Usage with Anthropic Agent Skills

Place this `stargazer_skills/` folder in your project root. Configure your agent runner to include these skills using the Agent Skills SDK.

Each `SKILL.md` follows the Anthropic Agent Skills format and will be loaded on-demand by Claude when the task context matches the skill's description.

## Experimental Design

These skills are intended for **test-time scaling experiments**:
- Training set: 100 tasks from RUN1 (used to derive these skills)
- Test set: New 30-task held-out sample (different seeds, not in RUN1)
- Comparison: pass rate with vs without skills injection

Use Anthropic's built-in A/B testing (Skill Description Optimization Loop) to measure improvement.
