# SOLAR — Linear Probe Experiment (Claim 3: Mechanism)

## Purpose

Determine **where** the Composition Gap occurs:

- **Option A (Hypothesis):** Models internally encode enough information to determine
  structural labels, but the output head fails to compose pieces into structure.
  → Probe recovers pieces AND structure from hidden states.
  → The classifier head is the bottleneck.

- **Option B (Alternative):** Models don't encode structural information at all.
  → Probe recovers pieces but NOT structure.
  → Representational failure, not compositional.

## Scripts

| Script | What it does | GPU needed? |
|--------|-------------|-------------|
| `extract_hidden_states.py` | Runs model inference, saves hidden-state vectors per layer | Yes |
| `train_probes.py` | Trains linear classifiers on saved hidden states | No (CPU ok) |
| `analyze_results.py` | Generates tables, plots, and statistical tests | No |
| `run_all.sh` | End-to-end orchestration | Yes |

## Usage

```bash
# Full pipeline
bash linear_probe/run_all.sh

# Step by step
python linear_probe/extract_hidden_states.py --model deepseek-ai/DeepSeek-V3 --layers -1,-4,-8,-16 --batch-size 4
python linear_probe/train_probes.py --hidden-states-dir data/results/linear_probe/hidden_states/
python linear_probe/analyze_results.py --results-dir data/results/linear_probe/
```

## Output

```
data/results/linear_probe/
├── hidden_states/          # Saved activations per layer
│   ├── layer_-1.pt
│   ├── layer_-4.pt
│   └── ...
├── probes/                 # Trained probe weights
│   ├── constraint_type_layer_-1.pkl
│   ├── entity_present_layer_-1.pkl
│   └── ...
├── results/                # Metrics and analysis
│   ├── probe_results.json
│   ├── composition_gap_probe.csv
│   └── figures/
└── logs/
```
