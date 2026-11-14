# Next experiments: summary

## Results (500-message sample)

- Claude Opus with one example per intent (E2a): 0.888 (-0.4 points against Claude zero-shot 0.892; Holm-corrected p = 1).
- Claude Opus with retrieved examples (E2b): 0.942 (+5.0 points against Claude zero-shot 0.892; Holm-corrected p = 0.000492).
- Claude Haiku zero-shot (E3a): 0.400 (-49.2 points against Claude zero-shot 0.892; Holm-corrected p = 4.04e-67).
- Claude Haiku with one example per intent (E3b): 0.628 (-26.4 points against Claude zero-shot 0.892; Holm-corrected p = 1.77e-31).
- Qwen3-4B + LoRA, 9,993 labels, seed 0 (E1): 0.938 (+4.6 points against Claude zero-shot 0.892; Holm-corrected p = 0.00819).
- Qwen3-4B + LoRA, 9,993 labels, seed 1 (E1): 0.948 (+5.6 points against Claude zero-shot 0.892; Holm-corrected p = 0.000192).
- Qwen3-4B + LoRA, 9,993 labels, seed 2 (E1): 0.944 (+5.2 points against Claude zero-shot 0.892; Holm-corrected p = 0.00467).
- ModernBERT-base fine-tuned (E4b), 100 labels: 0.173 ± 0.031 over 3 seed(s).
- ModernBERT-base fine-tuned (E4b), 500 labels: 0.579 ± 0.044 over 3 seed(s).
- ModernBERT-base fine-tuned (E4b), 2,000 labels: 0.847 ± 0.008 over 3 seed(s).
- ModernBERT-base fine-tuned (E4b), 9,993 labels: 0.913 ± 0.004 over 3 seed(s).
- bge-small + logistic regression (E4a), 100 labels: 0.236 ± 0.016 over 3 seed(s).
- bge-small + logistic regression (E4a), 500 labels: 0.731 ± 0.015 over 3 seed(s).
- bge-small + logistic regression (E4a), 2,000 labels: 0.887 ± 0.002 over 3 seed(s).
- bge-small + logistic regression (E4a), 9,993 labels: 0.896 ± 0.000 over 3 seed(s).

Full tables, paired tests, error analysis and prompts: `results/t1_baselines.md`.

## Claude calls

91 of 140 used (E2c and the Sonnet runs were not run).

## Deviations

- The Claude Code CLI was updated from 2.1.322 (used for the earlier Claude runs) to 2.1.323 before these runs; every flag the baseline uses is still present. The zero-shot Opus reference keeps its 2.1.322 run.
- Model names were checked on the Hugging Face Hub; the expected models exist and were used: BAAI/bge-small-en-v1.5 (revision 5c38ec7c405e) for E2b and E4a, answerdotai/ModernBERT-base (revision 8949b909ec90) for E4b. The CLI accepts --model claude-haiku-5-5 and reports claude-haiku-5-5 as the answering model.
- One Claude call per set (E2a, E2b, E3a, E3b) was made during a test run on the real first batch, with byte-identical inputs and prompts; these predictions were kept as batch 0 and count toward the 140-call cap. In that test run the first E3b reply contained 24 labels instead of 25; under the fixed rule the gate was retried once and passed.
- E2b embeds messages without a query instruction prefix (symmetric message-to-message similarity). The union of retrieved examples for a call is sorted by subset position and then shuffled with a fresh random.Random(7) for each call. E2a keeps the 77 examples in the order of the random.Random(2024) shuffle.
- E1 results for seeds 1 and 2 are written to results/next/ as exp1_Qwen3-4B-Instruct-2507_n9993_s{1,2}.json and preds_..._n9993_s{1,2}.jsonl, so no existing result file is changed.
- E4b uses PyTorch's AdamW defaults apart from the learning rate (weight decay 0.01) and gradient-norm clipping at 1.0 (as in the Qwen runs); the plan did not specify these.
- E4a and E4b latency is measured on the 2,000-label seed-0 model, chosen before any run. The Qwen latency in the comparison is the earlier measurement of the merged out-of-scope adapter.
- Cost for E4a and E4b uses only the two rental-rate assumptions; their GPU power draw was not measured (the batch-16 runs last under a second), so no energy-only rate is given for them.
- The Holm-Bonferroni family contains every run compared with Claude zero-shot in the baselines report, including the earlier Qwen budgets, which makes the correction stricter.
- sentence-transformers 6.3.0 was installed without dependency changes; torch 2.15.0, transformers 5.22.2 and peft 0.21.2 are unchanged.
