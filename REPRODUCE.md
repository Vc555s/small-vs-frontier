# Reproducing the results

All commands run from the repository root. Times are approximate and were measured on one shared 24 GB Quadro RTX 6000 (Turing, fp16 only); expect them to vary with hardware and load.

## 1. Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132   # pick the build for your CUDA
pip install -r requirements.txt
```

**GPU memory:** at least 11 GB free on one GPU for training (gradient checkpointing is on by default) and about 8–11 GB for evaluation. Bigger evaluation batches need more. Datasets (`mteb/banking77`, `clinc/clinc_oos`) and the base model (`Qwen/Qwen3-4B-Instruct-2507`, about 8 GB) download from Hugging Face on first use.

## 2. Fine-tuned small model (Qwen3-4B + LoRA)

```bash
# untuned model plus LoRA at 100 / 500 / 2,000 labels, three seeds (budget 0 needs only one seed)
for s in 0 1 2; do
  b="100,500,2000"; [ $s -eq 0 ] && b="0,100,500,2000"
  CUDA_VISIBLE_DEVICES=0 python finetune_qwen.py --seed $s --budgets $b
done
# full training set (9,993 labels), seed 0
CUDA_VISIBLE_DEVICES=0 python finetune_qwen_full.py && python finetune_qwen_full.py --collect
# full training set, seeds 1 and 2 (outputs go to results/baselines/; --log extracts the training loss)
mkdir -p results/baselines/logs
for s in 1 2; do
  CUDA_VISIBLE_DEVICES=0 python finetune_qwen_full.py --seed $s --out results/baselines/full_data_s$s --dest results/baselines \
    2>&1 | tee results/baselines/logs/full_s$s.log
  python finetune_qwen_full.py --collect --seed $s --out results/baselines/full_data_s$s --dest results/baselines \
    --log results/baselines/logs/full_s$s.log
done
```

About 1 hour per seed for budgets 0–2,000 (the 2,000-label run takes about 32 minutes to train), and about 3 hours per seed for the full training set. Evaluating on the full 3,076-message test set takes about 6 minutes per run.

All three full-training-set seeds (0, 1 and 2) are reported: the reports give their mean ± standard deviation, and the published full-data adapter is seed 0, fixed before any result was seen. The seed 1 and 2 runs are the same as the `full_s1` and `full_s2` stages of `run_baselines.sh` (section 6), which runs them for you.

## 3. Claude baseline (main run)

Requires the **Claude Code CLI** (version 2.1.322 was used) logged in with a Claude subscription. Calls are subject to the subscription's usage and rate limits. The script sends 20 calls of 25 messages each, sequentially, from an empty temporary directory with all tools disabled.

```bash
python claude_baseline.py --limit_batches 1   # one batch first, to check the CLI works
python claude_baseline.py                     # remaining batches, then scoring
```

## 4. Robustness tests (typos, two-issue messages, out-of-scope messages)

```bash
bash run_robustness.sh all      # builds the frozen test sets, runs GPU and Claude stages, writes the report
```

Resumable: a re-run skips finished stages. About 2.5 hours of GPU time, most of it the three retrained out-of-scope adapters (about 37 minutes each), plus about 59 Claude CLI calls. The built test sets in `results/stress/` are frozen; `stress_build.py` regenerates them byte for byte.

## 5. Latency, cost and model card

```bash
CUDA_VISIBLE_DEVICES=0 python measure_latency.py --adapter <path to the S3 seed-0 adapter>
python compute_cost.py
python make_model_card.py                   # model_cards/router/ (router adapter) and model_cards/full/ (full-training-set adapter)
python model_cards/router/verify_usage.py   # each card's usage snippet, run against the Hub adapter,
python model_cards/full/verify_usage.py     # must reproduce 100 saved predictions
```

## 6. Baselines: examples in the prompt, Claude Haiku, non-generative models, more seeds

```bash
python build_prompt_examples.py   # frozen inputs: the 500 messages in batch order and the example lists
bash run_baselines.sh all         # GPU: embeddings + logistic regression, ModernBERT, full training set seeds 1-2;
                                  # CLI: Claude Opus with examples, Claude Haiku; then the reports
```

Resumable. About 6 hours on one GPU (most of it the two full-data Qwen seeds, about 2.75 hours each) and 80 Claude CLI calls (the pipeline stops at 140). `make_baselines_report.py` writes `results/baselines_report.md`. Needs `sentence-transformers` (see `requirements.txt`).

## 7. Reports and figures

These need no GPU. The report scripts need only `datasets` and `scikit-learn`; the figures also need `matplotlib`.

```bash
python make_figures.py
python make_main_report.py
python make_robustness_report.py
python make_baselines_report.py
python summarise_review.py   # counts the manual judgements in results/baselines/joint_errors_for_review.csv
```

The reports are rebuilt from the files in `results/`; every number in them is computed, not typed.
