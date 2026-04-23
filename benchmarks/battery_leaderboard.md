# Battery Cell Design Benchmark — CellDesign-v0

**Tasks:** 30 cell design tasks across 3 difficulty tiers and 3 chemistries (NMC532, LFP, NCA)  
**Verifier:** Basic open-source verifier (PyBaMM electrochemical simulation + physics-based cycle life)  
**Metric:** % of tasks where `info["success"] == True` (overall score ≥ 0.7 AND cycle_life score ≥ 0.4 AND peak temp ≤ limit + 10°C)

_Last updated: 2026-04-23_

---

## Results

### Zero-shot (no RL training)

| Model | Easy (10) | Medium (10) | Hard (10) | Overall | Notes |
|-------|-----------|-------------|-----------|---------|-------|
| Qwen2.5-Coder-32B | 40% | 10% | 0% | 17% | Best zero-shot |
| DeepSeek-Coder-V2 | 30% | 10% | 0% | 13% | |
| GPT-4o | 30% | 10% | 0% | 13% | Via API |
| Claude Sonnet 4.6 | 20% | 10% | 0% | 10% | Via API |
| Qwen2.5-Coder-7B | 20% | 0% | 0% | 7% | Colab T4 |

### After RL training

| Model | Easy (10) | Medium (10) | Hard (10) | Overall | Training episodes | Notes |
|-------|-----------|-------------|-----------|---------|-------------------|-------|
| Qwen2.5-Coder-7B | — | — | — | 73%* | ~2000 | *Reported result, full eval pending |

_Submit your results — see below._

---

## Methodology

1. Load each of the 30 tasks by ID (`cell_ev_commuter_001` through `cell_nextgen_target_001`)
2. Prompt the model with `obs["task_description"]` + scalar requirements
3. Execute the generated Python params code in `CellDesignEnv`
4. Record `info["success"]` and `info["dimension_scores"]`
5. Report pass rate per difficulty tier

Full evaluation harness: [`examples/battery/run_benchmark.py`](../examples/battery/run_benchmark.py)

### Prompt used for all evaluations

```
You are an electrochemical engineer. Write Python code that defines battery cell parameters
for a 21700 cylindrical cell that satisfies the requirements below.

Task: {obs["task_description"]}
Target energy density:  {target_energy_density_whkg} Wh/kg
Minimum cycle life:     {min_cycle_life} cycles to 80% capacity
Max peak temperature:   {max_peak_temp_c} °C
Cost budget:            ${max_cost_kwh}/kWh
Charge rate:            {c_rate_charge}C
Discharge rate:         {c_rate_discharge}C
Ambient temperature:    {ambient_temp_c} °C

Rules:
- Define a variable named `params` as a Python dict
- Include all required keys: chemistry, negative_electrode_thickness,
  negative_electrode_porosity, negative_electrode_particle_radius,
  positive_electrode_thickness, positive_electrode_porosity,
  positive_electrode_particle_radius, separator_thickness,
  separator_porosity, ambient_temperature_celsius
- chemistry must be one of: "NMC532", "LFP", "NCA"
- Thicknesses in metres (e.g. 80e-6 for 80 µm)
- Porosities as fractions 0.0–1.0
- Particle radii in metres (e.g. 5e-6 for 5 µm)
- No other output, imports, or function definitions needed

Python code:
```

---

## Score dimensions

| Dimension | Weight | Passes when |
|-----------|--------|-------------|
| Energy density | 30% | Within 15% of target |
| Cycle life | 35% | Meets `min_cycle_life` |
| Thermal | 20% | Peak temp ≤ `max_peak_temp_c` |
| Cost | 15% | Cell cost ≤ `max_cost_kwh` |

**Pass condition:** overall ≥ 0.70 AND cycle_life score ≥ 0.40 AND peak temp ≤ limit + 10°C

---

## Task list

### Easy (10 tasks)

| Task ID | Chemistry | Target Wh/kg | Min cycles | Max °C | C-rate |
|---------|-----------|-------------|------------|--------|--------|
| cell_ev_commuter_001 | NMC532 | 220 | 600 | 45 | 1C/1C |
| cell_ev_standard_001 | NMC532 | 250 | 800 | 45 | 1C/1C |
| cell_consumer_phone_001 | NMC532 | 230 | 500 | 45 | 1C/1C |
| cell_grid_storage_basic_001 | LFP | 140 | 2000 | 55 | 0.5C/0.5C |
| cell_ebike_001 | NMC532 | 210 | 700 | 50 | 1C/2C |
| cell_escooter_001 | NMC532 | 200 | 600 | 50 | 1C/3C |
| cell_laptop_001 | NMC532 | 240 | 500 | 45 | 1C/1C |
| cell_power_tool_001 | NMC532 | 190 | 400 | 55 | 1C/4C |
| cell_drone_001 | NMC532 | 235 | 300 | 50 | 1C/5C |
| cell_home_backup_001 | LFP | 135 | 3000 | 55 | 0.2C/0.2C |

### Medium (10 tasks)

| Task ID | Chemistry | Target Wh/kg | Min cycles | Max °C | C-rate | Challenge |
|---------|-----------|-------------|------------|--------|--------|-----------|
| cell_ev_fastcharge_001 | NMC532 | 260 | 800 | 45 | 3C/1C | Fast charge |
| cell_grid_longlife_001 | LFP | 155 | 5000 | 55 | 0.5C/0.5C | Long life |
| cell_ev_coldweather_001 | NMC532 | 240 | 700 | 45 | 1C/1C | −20°C ambient |
| cell_ev_premium_001 | NMC532 | 270 | 1000 | 45 | 1C/1C | High energy + life |
| cell_marine_001 | LFP | 145 | 3000 | 50 | 0.5C/1C | Durability |
| cell_forklift_001 | LFP | 150 | 4000 | 55 | 0.5C/1C | Deep cycle |
| cell_residential_solar_001 | LFP | 145 | 4000 | 50 | 0.2C/0.5C | Longevity |
| cell_ev_hotclimate_001 | NMC532 | 245 | 700 | 42 | 1C/1C | 45°C ambient |
| cell_medical_device_001 | NMC532 | 220 | 1500 | 40 | 0.5C/0.5C | Tight thermal |
| cell_autonomous_robot_001 | NMC532 | 255 | 800 | 45 | 1C/2C | Multi-use |

### Hard (10 tasks)

| Task ID | Chemistry | Target Wh/kg | Min cycles | Max °C | C-rate | Challenge |
|---------|-----------|-------------|------------|--------|--------|-----------|
| cell_aerospace_001 | NCA | 290 | 500 | 45 | 1C/1C | Extreme energy density |
| cell_formula_e_001 | NMC532 | 285 | 300 | 50 | 4C/4C | Max power |
| cell_grid_ultralong_001 | LFP | 158 | 8000 | 55 | 0.5C/0.5C | 20+ year life |
| cell_arctic_ev_001 | NMC532 | 230 | 600 | 45 | 1C/1C | −40°C ambient |
| cell_satellite_001 | NCA | 280 | 2000 | 30 | 1C/1C | Extreme temp limit |
| cell_implantable_001 | NMC532 | 220 | 2000 | 38 | 0.1C/0.1C | Body-safe temp |
| cell_hypercar_001 | NCA | 300 | 200 | 55 | 5C/8C | Max power + energy |
| cell_deep_space_001 | NMC532 | 285 | 500 | 35 | 0.5C/0.5C | −60°C ambient |
| cell_submarine_ess_001 | LFP | 155 | 5000 | 40 | 0.5C/1C | Pressure + thermal |
| cell_nextgen_target_001 | NMC532 | 310 | 1500 | 45 | 1C/1C | Beyond current limits |

---

## How to submit your results

### Requirements

- Evaluation run against all 30 tasks (`cell_ev_commuter_001` through `cell_nextgen_target_001`)
- Results saved as JSON (format below)
- Model name, parameter count, and whether RL-trained

### Steps

1. Run the evaluation harness:

```bash
python examples/battery/run_benchmark.py \
    --model your_model_name \
    --output benchmarks/results/my_model_eval.json
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
  "eval_date": "2026-04-23",
  "evaluator": "github.com/your-handle",
  "hardware": "Colab T4",
  "overall_pass_rate": 0.07,
  "pass_rate_by_difficulty": {
    "easy": 0.20,
    "medium": 0.00,
    "hard": 0.00
  },
  "mean_scores": {
    "energy": 0.45,
    "cycle_life": 0.38,
    "thermal": 0.72,
    "cost": 0.68
  },
  "task_results": {
    "cell_ev_standard_001": {
      "passed": true,
      "reward": 0.76,
      "dimension_scores": {
        "energy": 0.82,
        "cycle_life": 0.71,
        "thermal": 0.88,
        "cost": 0.91
      }
    }
  }
}
```

### What we verify before merging

- JSON parses and matches the schema above
- Task IDs cover all 30 tasks
- Pass rate in the table matches the JSON
- No gamed results (cycle_life = 1.0 on all hard tasks with cost = 0.0 is a red flag)

---

## Notes on the basic verifier

The basic verifier uses PyBaMM (SPM + lumped thermal) for electrochemical simulation and a
physics-based formula for cycle life estimation. Cycle life depends on particle size, porosity,
C-rate, and temperature via an Arrhenius factor.

**Known gaming patterns:**
- Very thin separators (< 15 µm) can produce anomalously high energy density
- Extremely low porosities (< 0.15) produce high energy density but unphysical ion transport
- Negative electrode particle radius < 1 µm improves cycle life score but is not manufacturable

If you observe dimension scores where `cycle_life → 1.0` on all hard tasks, run the same
evaluation with `verifier_mode="pro"` and report both numbers. Results showing divergence
between basic and Pro verifier scores are especially valuable — they expose gaming patterns
and directly train the Pro verifier.
