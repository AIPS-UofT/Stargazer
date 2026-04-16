# Stargazer

A benchmark for evaluating LLM agents on exoplanet detection from radial velocity (RV) data.

## Overview

Stargazer tests whether LLM agents can perform end-to-end scientific reasoning: analyzing time-series RV observations, detecting planetary signals, fitting Keplerian orbital models, and correctly identifying multi-planet systems. The benchmark includes 100 synthetic tasks (graded by difficulty 1--10) and 20 real-world tasks from published archival RV datasets.

Agents interact with the environment through a ReAct-style loop with two tools:
- **PythonREPL**: execute analysis code (periodograms, Keplerian fitting, residual inspection)
- **submit_action**: propose candidate planetary systems and receive per-criterion feedback

Each submission is evaluated against four pass/fail criteria: delta-BIC, RMS, Match Score, and Planet Count. A task is solved only when all four are satisfied simultaneously.

## Installation

```bash
pip install -r requirements.txt
```

To also run the classical and nested-sampling baselines:

```bash
pip install -r requirements-baselines.txt
```

Required: Python 3.10+. Set API keys via environment variables or `.env`:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Quick Start

Run a single task by ID (useful for debugging):
```bash
python run_agent_batch_hard_timeout.py \
    --model gpt-5-mini \
    --task-ids seed22_diff4 \
    --verbose \
    --output-dir results_debug
```

Run a batch of 10 tasks:
```bash
python run_agent_batch_hard_timeout.py \
    --model gpt-5-mini \
    --count 10 \
    --output-dir results_gpt5mini
```

Run all difficulty-1 tasks with verbose output:
```bash
python run_agent_batch_hard_timeout.py \
    --model gpt-5-mini \
    --count 1000 \
    --difficulties 1 \
    --workers 10 \
    --verbose \
    --output-dir results_diff1
```

Run with skills injection (enabled by default in the skills runner):
```bash
python run_agent_batch_skills_timeout.py \
    --model gpt-5-mini \
    --count 10 \
    --output-dir results_gpt5mini_skills
```

Disable skills in the skills runner:
```bash
python run_agent_batch_skills_timeout.py --no-skills ...
```

## Upload To Hugging Face

Export the benchmark into a Hugging Face-ready dataset repository folder:

```bash
python3 prepare_hf_dataset.py --output-dir hf_dataset_export
```

Install the optional upload dependencies:

```bash
pip install -r requirements-hf.txt
```

Push the exported dataset to a Hugging Face dataset repo:

```bash
export HF_TOKEN=...
python3 prepare_hf_dataset.py \
    --output-dir hf_dataset_export \
    --repo-id your-username/Stargazer \
    --license other \
    --push
```

This writes a publishable dataset repo layout under `hf_dataset_export/`:

```text
hf_dataset_export/
  README.md
  data/
    default/train.jsonl
    synthetic/train.jsonl
    real/train.jsonl
```

The `default` config contains all tasks, while `synthetic` and `real` are exposed as separate Hugging Face configs. Replace the placeholder license with the correct one before publishing.

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `gpt-5-mini` | Model name (supports OpenAI, Anthropic, OpenRouter) |
| `--bank-dir` | `stargazer/Stargazer_synthetic_task` | Task bank directory |
| `--task-ids` | — | Run specific task(s) by ID, comma-separated (e.g. `seed22_diff4`) |
| `--count` | 3 | Number of tasks to run |
| `--difficulties` | all | Filter by difficulty, e.g. `1-3` or `7,8,9` |
| `--workers` | 1 | Parallel workers |
| `--hard-timeout` | 1600 | Wall-clock timeout per task (seconds) |
| `--verbose` | off | Print detailed agent traces (tool calls, responses) |
| `--difficulty-budget` | on | Auto-adjust token/step budget by tier |
| `--skills` | on (skills_timeout) | Inject domain-expert skills into system prompt |
| `--resume` | off | Skip already-completed tasks in output dir |

## Project Structure

```
stargazer/                          Core package
  config.py                         Task and system config dataclasses
  task_factory.py                   Synthetic task generation with difficulty scoring
  bank.py                           File-based task bank (load/save/list)
  env.py                            RvEnv: agent-environment interaction loop
  evaluator.py                      4-criterion evaluation (BIC, RMS, Match, Count)
  matching.py                       Hungarian matching for planet-to-planet comparison
  forward_keplerian.py              Analytic Keplerian RV forward model
  engine_rebound.py                 N-body RV simulation via REBOUND
  noise.py                          Gaussian process and white noise injection
  priors.py                         Orbital parameter sampling distributions
  schedule.py                       Observation scheduling
  agents/
    tabular_agent.py                LLM agent (OpenAI, Anthropic, OpenRouter)
    common.py                       Submission parsing and validation
    format_utils.py                 Trace export (JSON, Markdown, HTML)
    openai_utils.py                 OpenAI chat completion wrapper
    tools/                          PythonREPL and submit_action tool definitions
  benchmarks/baselines.py           Null model and single-sine baselines
  tests/                            Unit tests
  Stargazer_synthetic_task/          100 synthetic tasks (10 per difficulty 1--10)
  Stargazer_real_data_task/          20 real-world tasks from archival RV data

stargazer_skills/                   5 domain-expert skills for prompt injection
  rv_period_search/                 Period search and alias detection
  rv_keplerian_fit/                 Robust Keplerian orbit fitting
  rv_lrad_calculation/              Mean longitude computation
  rv_multiplanet_detection/         Iterative residual analysis (difficulty >= 4)
  rv_submit_strategy/               Submission timing and budget management

run_agent_batch.py                  Base batch runner (library, used by timeout scripts)
run_agent_batch_hard_timeout.py     Main experiment runner with subprocess watchdog
run_agent_batch_skills_timeout.py   Skills experiment runner (--skills/--no-skills)
stargazer_submit.py                 Evaluate a single submission against ground truth
generate_synthetic_bank.py          Generate synthetic task bank from seeds
generate_real_tasks.py              Generate real-data tasks from archival RV data
generate_ground_truth.py            Generate ground truth parameters
generate_balanced_tasks_by_difficulty.py   Generate difficulty-stratified task sets
```

## Evaluation Criteria

A task passes only when **all four** criteria are met:

1. **ok_delta_bic**: submitted model is statistically preferred over a flat line
2. **ok_rms**: residual RMS <= 1.5x median measurement uncertainty
3. **ok_match**: Match Score >= 0.8 (planet parameters match ground truth)
4. **ok_count**: correct number of planets detected

## Difficulty Tiers

| Tier | Difficulty | Synthetic Tasks | Token Budget | Time Budget | Max Submissions |
|---|---|---|---|---|---|
| Easy | 1--2 | 20 | 200K | 600s | 3 |
| Medium | 3--6 | 40 | 450K | 900s | 5 |
| Hard | 7--10 | 40 | 900K | 1500s | 10 |
