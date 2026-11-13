# Finishing T1: full-training-set run, latency and cost, model release

## 1. Full training set (9,993 labels, seed 0)

| System | Labels | Seeds | Accuracy, 500 sample | Accuracy, full test set (3,076) |
|---|---|---|---|---|
| Qwen3-4B + LoRA | 9,993 | 1 (seed 0) | 0.938 | 0.9477 |
| Qwen3-4B + LoRA | 2,000 | seed 0 | 0.910 | 0.9138 |
| Qwen3-4B + LoRA | 2,000 | mean ± std of 3 | 0.905 ± 0.012 | 0.9131 ± 0.0022 |
| Claude via Claude Code CLI | prompted | 1 run | 0.892 | n/a |

Paired comparison on the 500 sample (exact McNemar, two-sided): against Claude, the 9,993-label model is right where Claude is wrong on 32 items and wrong where Claude is right on 9 (p = 0.000431); against the 2,000-label seed-0 model, 21 vs 7 (p = 0.0125).

One seed cannot show run-to-run spread; the 2,000-label seeds differ by up to 2.2 points on the same sample, so differences of that size between single runs are not evidence of a real effect.

## 2. Latency and cost

Merged S3 adapter (seed 0), fp16, greedy, Quadro RTX 6000 **shared with other jobs** (about 54% utilisation from other processes at the start).

| Measure | Value |
|---|---|
| Batch 1 latency p50 / p95 / p99 | 258.2 / 350.7 / 442.5 ms |
| Throughput, batch 8 | 9.45 messages/s |
| Throughput, batch 16 | 11.19 messages/s |
| Throughput, batch 32 | 11.52 messages/s |

| System | Assumed rate | Cost per 1,000 requests | Break-even volume |
|---|---|---|---|
| Qwen3-4B + LoRA | energy only: measured 251 W at EUR 0.3/kWh | EUR 0.0019 | 80 requests/hour |
| Qwen3-4B + LoRA | illustrative rental at EUR 0.25/GPU-hour | EUR 0.0062 | 265 requests/hour |
| Qwen3-4B + LoRA | illustrative rental at EUR 0.50/GPU-hour | EUR 0.0124 | 530 requests/hour |
| Claude via Claude Code CLI | the Claude Code CLI's list-price estimate with 25 messages per call and prompt caching; not a measured API bill, and not a charge on a subscription | USD 0.94 | n/a |

All hourly rates are assumptions, not quoted market prices. A dedicated GPU costs the same whether it handles 10 or 10,000 requests an hour, so the small model is cheaper only above the break-even volume = hourly GPU rate / Claude cost per request. Break-even volumes treat 1 USD as 1 EUR.

## 3. Model release

- Adapter: https://huggingface.co/vishal77796/qwen3-4b-banking77-router-lora (created private).
- Verification from the Hub (fresh cache for the adapter; the base model was read from an existing local cache): the model-card usage code on 100 test items agreed with the saved predictions on 100 of 100; prompt identical to training: True.

## 4. Files added or changed (relative to `main`)

- modified: `.gitignore`
- modified: `LICENSE`
- modified: `STRESS_DECISIONS.md`
- modified: `baseline_claude_code.py`
- added: `compute_cost.py`
- modified: `exp1_routing.py`
- added: `make_model_card.py`
- modified: `make_robustness_report.py`
- modified: `make_t1_report.py`
- added: `measure_latency.py`
- added: `model_card/README.md`
- added: `model_card/usage.py`
- added: `model_card/verify_usage.py`
- modified: `requirements.txt`
- added: `results/cost_t1.json`
- added: `results/latency_t1.json`
- modified: `results/t1_report.md`
- modified: `results/t1_robustness.md`
- modified: `run_stress.sh`
- modified: `stress_build.py`
- modified: `stress_common.py`
- added: `t1_common.py`
- added: `REPRODUCE.md`
- added: `full_data.py`
- added: `make_figures.py`
- added: `make_finish_summary.py`
- added: `results/exp1_Qwen3-4B-Instruct-2507_n9993_s0.json`
- added: `results/figures/label_efficiency.png`
- added: `results/figures/out_of_scope.png`
- added: `results/figures/stress_tests.png`
- added: `results/model_card_verification.json`
- added: `results/preds_Qwen3-4B-Instruct-2507_n9993_s0.jsonl`
- added: `results/train_loss_n9993_s0.json`
- added: `run_finish.sh`
- added: `results/finish_summary.md`

## 5. Failures, skips and deviations

- No stage failed or was skipped.
- The 9,993-label run used one seed, not three.
- Its result file is `results/exp1_Qwen3-4B-Instruct-2507_n9993_s0.json`, not the usual `exp1_Qwen3-4B-Instruct-2507_s0.json`, so that the existing seed-0 results for budgets 0–2,000 are not overwritten.
- Latency was measured on a shared GPU while other processes were running; the numbers are not best-case.
- The break-even volume needs one currency; 1 USD was treated as 1 EUR.
- The seed-to-seed change in individual predictions (S3 adapters, 700-message test set) was measured at 6.2% for a pair of seeds on average and 8.9% for any of three, and the model card states these measured values rather than a round figure.
- The bar charts start at zero; the label-efficiency line chart starts at 0.5 (all values lie above 0.6).
- The first upload attempt to the Hub was refused because the token lacked write permission; it succeeded with a write-enabled token.
