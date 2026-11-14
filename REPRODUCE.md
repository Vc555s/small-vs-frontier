# Reproducing the T1 results

All commands run from the repository root. Times are approximate and were measured on one shared 24 GB Quadro RTX 6000 (Turing, fp16 only); expect them to vary with hardware and load.

## 1. Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132   # pick the build for your CUDA
pip install -r requirements.txt
```

**GPU memory:** at least 11 GB free on one GPU for training (gradient checkpointing is on by default) and about 8–11 GB for evaluation. Bigger evaluation batches need more. Datasets (`mteb/banking77`, `clinc/clinc_oos`) and the base model (`Qwen/Qwen3-4B-Instruct-2507`, about 8 GB) download from Hugging Face on first use.

## 2. T1: fine-tuned small model

```bash
# untuned model plus LoRA at 100 / 500 / 2,000 labels, three seeds (budget 0 needs only one seed)
for s in 0 1 2; do
  b="100,500,2000"; [ $s -eq 0 ] && b="0,100,500,2000"
  CUDA_VISIBLE_DEVICES=0 python exp1_routing.py --seed $s --budgets $b
done
# full training set (9,993 labels), seed 0
CUDA_VISIBLE_DEVICES=0 python full_data.py && python full_data.py --collect
```

About 1 hour per seed for budgets 0–2,000 (the 2,000-label run takes about 32 minutes to train), and about 3 hours for the full training set. Evaluating on the full 3,076-message test set takes about 6 minutes per run.

## 3. T1: Claude baseline

Requires the **Claude Code CLI** (version 2.1.322 was used) logged in with a Claude subscription. Calls are subject to the subscription's usage and rate limits. The script sends 20 calls of 25 messages each, sequentially, from an empty temporary directory with all tools disabled.

```bash
python baseline_claude_code.py --limit_batches 1   # one batch first, to check the CLI works
python baseline_claude_code.py                     # remaining batches, then scoring
```

## 4. Robustness tests (typos, two-issue messages, out-of-scope messages)

```bash
bash run_stress.sh all      # builds the frozen test sets, runs GPU and Claude stages, writes the report
```

Resumable: a re-run skips finished stages. About 2.5 hours of GPU time, most of it the three retrained out-of-scope adapters (about 37 minutes each), plus about 59 Claude CLI calls. The built test sets in `results/stress/` are frozen; `stress_build.py` regenerates them byte for byte.

## 5. Latency, cost and model card

```bash
CUDA_VISIBLE_DEVICES=0 python measure_latency.py --adapter <path to the S3 seed-0 adapter>
python compute_cost.py
python make_model_card.py           # model_card/ (router adapter) and model_card_full/ (full-training-set adapter)
python model_card/verify_usage.py        # each card's usage snippet, run against the Hub adapter,
python model_card_full/verify_usage.py   # must reproduce 100 saved predictions
```

## 6. Baselines: examples in the prompt, Claude Haiku, non-generative models, more seeds

```bash
python next_build.py                 # frozen inputs: the 500 messages in batch order and the example lists
bash run_next.sh all                 # E4a, E4b, E1 seeds 1-2 on the GPU; E2a, E2b, E3a, E3b via the CLI; report
```

Resumable. About 6 hours on one GPU (most of it the two full-data Qwen seeds, about 2.75 hours each) and 80 Claude CLI calls (the pipeline stops at 140). `make_baselines_report.py` writes `results/t1_baselines.md`. Needs `sentence-transformers` (see `requirements.txt`).

## 7. Reports and figures

These need no GPU. The report scripts need only `datasets` and `scikit-learn`; the figures also need `matplotlib`.

```bash
python make_figures.py
python make_t1_report.py
python make_robustness_report.py
```

The reports are rebuilt from the files in `results/`; every number in them is computed, not typed.
