# CAD Design Benchmark — BracketDesign-v0

**Tasks:** 50 bracket design tasks across 3 difficulty tiers and 9 materials  
**Verifier:** Basic open-source verifier (beam-theory FEA)  
**Metric:** % of tasks where `info["success"] == True` (score ≥ 0.7 AND structural ≥ 0.5)

_Last updated: 2026-04-21_

---

## Results

### Zero-shot (no RL training)

| Model | Easy (15) | Medium (20) | Hard (15) | Overall | Notes |
|-------|-----------|-------------|-----------|---------|-------|
| Qwen2.5-Coder-32B | 47% | 10% | 0% | 14% | Best zero-shot |
| DeepSeek-Coder-V2 | 40% | 10% | 0% | 12% | |
| GPT-4o | 33% | 10% | 0% | 11% | Via API |
| Claude Sonnet 4.6 | 27% | 5% | 0% | 8% | Via API |
| Qwen2.5-Coder-7B | 33% | 5% | 0% | 7% | Colab T4 |

### After RL training

| Model | Easy (15) | Medium (20) | Hard (15) | Overall | Training episodes | Notes |
|-------|-----------|-------------|-----------|---------|-------------------|-------|
| Qwen2.5-Coder-32B | — | — | — | 67%* | ~1000 | *Reported result, full eval pending |

_Submit your results — see below._

---

## Methodology

1. Load each of the 50 tasks by ID (`cad_bracket_001` through `cad_bracket_050`)
2. Prompt the model with `obs["task_description"]` + scalar requirements
3. Execute the generated CadQuery code in `BracketDesignEnv`
4. Record `info["success"]` and `info["dimension_scores"]`
5. Report pass rate per difficulty tier

Full evaluation harness: [`examples/cad/02_run_benchmark.py`](../examples/cad/02_run_benchmark.py)  
Visualization guide: [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md)

### Prompt used for all evaluations

```
You are a mechanical engineer. Write Python code using CadQuery to design a bracket
that satisfies the requirements below.

Task: {obs["task_description"]}
Load: {load_kg} kg
Extension: {extension_mm} mm from mounting surface
Budget: ${max_cost_usd}

Rules:
- Import cadquery as cq at the top
- Define a variable named `result` of type cadquery.Workplane
- No other output needed

CadQuery code:
```

---

## How to submit your results

### Requirements

- Evaluation run against all 50 tasks (`cad_bracket_001` through `cad_bracket_050`)
- Results saved as JSON (format below)
- Model name, parameter count, and whether RL-trained

### Steps

1. Run the evaluation harness:

```bash
python examples/cad/02_run_benchmark.py \
    --model your_model_name \
    --output results/my_model_eval.json
```

2. Fork the repo, create a branch named `leaderboard/<your-model>`.

3. Add your results JSON to `benchmarks/results/<your-model>_<date>.json`.

4. Add one row to the table above in this file.

5. Open a PR with:
   - Title: `Leaderboard: <Model Name> (<pass rate>%)`
   - Body: model card link, hardware used, training details if RL

### Results JSON format

```json
{
  "model": "Qwen2.5-Coder-7B-Instruct",
  "model_params": "7B",
  "rl_trained": false,
  "rl_episodes": null,
  "eval_date": "2026-04-21",
  "evaluator": "github.com/your-handle",
  "hardware": "Colab T4",
  "overall_pass_rate": 0.07,
  "pass_rate_by_difficulty": {
    "easy": 0.33,
    "medium": 0.05,
    "hard": 0.00
  },
  "mean_scores": {
    "structural": 0.32,
    "cost": 0.61,
    "geometry": 0.75
  },
  "task_results": {
    "cad_bracket_001": {
      "passed": true,
      "reward": 0.74,
      "dimension_scores": {"structural": 0.81, "cost": 0.72, "geometry": 1.0}
    }
  }
}
```

### What we verify before merging

- JSON parses and matches the schema above
- Task IDs cover all 50 tasks
- Pass rate in the table matches the JSON
- No obviously gamed results (structural = 1.0 on all hard tasks with cost = 0.0 is a red flag)

---

## Notes on the basic verifier

The basic verifier uses simplified beam-theory FEA. It can be gamed:
a model that learns to make very tall, thin brackets will get high structural scores
regardless of whether the bracket is manufacturable.

If you see structural scores → 1.0 on hard tasks after RL training, run the same
evaluation with `verifier_mode="pro"` and report both numbers.

Results that show significant divergence between basic and Pro verifier scores
are especially valuable — they demonstrate the gaming pattern and help train the Pro verifier.
