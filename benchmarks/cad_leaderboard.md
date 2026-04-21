# CAD Design Benchmark — BracketDesign-v0

**Tasks:** 50 bracket design tasks (5 easy, 25 medium, 20 hard)  
**Verifier:** Basic open-source verifier  
**Metric:** % tasks passing all grading dimensions  

_Last updated: 2026-04-20_

## Results

| Model | Zero-shot | After RL | Notes |
|-------|-----------|----------|-------|
| Qwen2.5-Coder-32B | 14% | — | Baseline |
| GPT-4o | 11% | — | Baseline |
| Claude Sonnet 4.6 | 8% | — | Baseline |
| DeepSeek-Coder-V2 | 12% | — | Baseline |

_RL-trained results coming soon._

## Methodology

Each model is prompted with the task description and asked to generate CadQuery Python code.
The generated code is executed, simulated with simplified beam theory FEA, and scored across three dimensions.

Full evaluation harness: [`examples/cad/02_run_benchmark.py`](../examples/cad/02_run_benchmark.py)

## Submit Results

Open a PR adding your model's results with the evaluation notebook.
