# Self-Improvement via Inversion

**Training language models without human labels, ground truth, or external reward models.**

## The Key Insight

> If a model can reconstruct what it was asked from what it generated, it probably understood the task.

```
Forward:  "Scan port 80 on 192.168.1.1" → nmap -p 80 192.168.1.1
Inverse:  nmap -p 80 192.168.1.1 → "Scan port 80 on host 192.168.1.1"
Score:    similarity(original, reconstructed) = 0.95
```

High reconstruction fidelity = the model understood what it generated = good output.

## Results (10 Cycles)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Intrinsic Score | 0.531 | 0.627 | **+18.1%** |
| Acceptance Rate | 60.7% | 88.0% | **+27.3pp** |
| Training Loss | 5.70 | 4.73 | **-17%** |

**Zero human labels. Zero external reward models. Pure self-improvement.**

📊 [Full results breakdown](RESULTS.md)

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  SELF-IMPROVEMENT LOOP                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   1. GENERATE    Task ──────────► Command               │
│                  "Scan port 80"    "nmap -p 80 ..."     │
│                                                         │
│   2. INVERT      Command ─────────► Task'               │
│                  "nmap -p 80 ..."   "Scan port 80"      │
│                                                         │
│   3. SCORE       similarity(Task, Task')                │
│                  High = understood, Low = confused      │
│                                                         │
│   4. SELECT      Keep high-scoring examples             │
│                                                         │
│   5. TRAIN       Fine-tune on self-generated data       │
│                                                         │
│   6. REPEAT      → Better generation → Better scores    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run self-improvement (3 cycles, quick test)
python intrinsic_self_improvement.py --cycles 3 --tasks 30

# Run extended experiment (10 cycles)
python intrinsic_self_improvement.py --cycles 10 --tasks 50

# Run with custom settings
python intrinsic_self_improvement.py \
  --cycles 20 \
  --tasks 50 \
  --candidates 5 \
  --threshold 0.5 \
  --temperature 0.7 \
  --output ./my_experiment
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cycles` | 10 | Number of self-improvement cycles |
| `--tasks` | 50 | Tasks generated per cycle |
| `--candidates` | 3 | Candidates per task |
| `--threshold` | 0.5 | Minimum score to accept example |
| `--temperature` | 0.7 | Generation temperature |
| `--model` | mistralai/Mistral-7B-Instruct-v0.2 | Base model |
| `--output` | ./output/extended_run | Output directory |

## Output

```
output/
├── {experiment_name}/
│   ├── checkpoint_cycle_1/    # Model after cycle 1
│   ├── checkpoint_cycle_2/    # Model after cycle 2
│   ├── ...
│   ├── progress.json          # Cycle-by-cycle results
│   └── self_improvement_results.json  # Final results
```

## Why This Is Novel

| Method | Human Labels | Ground Truth | External Judge |
|--------|--------------|--------------|----------------|
| STaR (2022) | No | **Yes** | No |
| SPIN (2024) | **Yes** | Yes | No |
| Self-Rewarding (2024) | No | No | **Yes** |
| **Ours** | **No** | **No** | **No** |

We eliminate ALL external supervision by using inversion as an intrinsic signal.

## Citation

```bibtex
@misc{inversion-self-improvement-2025,
  title={Self-Improvement via Inversion: Training Language Models Without External Supervision},
  author={Adam Kruger},
  year={2025},
  url={https://github.com/CINOAdam/inversion-self-improvement}
}
```

## License

MIT
