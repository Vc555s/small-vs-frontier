# Small fine-tuned model vs Claude on support-ticket routing

**The full write-up is coming soon.** This repository holds the code, frozen test inputs, every prediction and the two generated reports for the study:

- `results/t1_report.md`: Qwen3-4B-Instruct-2507 fine-tuned with LoRA at 0 / 100 / 500 / 2,000 labels (3 seeds) vs Claude via the Claude Code CLI on Banking77 routing.
- `results/t1_robustness.md`: the same systems under typos, two-issue messages and out-of-scope messages.

Both reports are generated from the files in `results/` by `make_t1_report.py` and `make_robustness_report.py`.

## Data and attribution

The files in `results/` contain message text from two public datasets, and versions of that text we modified: lowercased with punctuation removed and simulated typos added (`results/stress/typo500.jsonl`), and pairs of messages joined into one (`results/stress/merged250.jsonl`). The text stays under its original licence. The MIT licence in `LICENSE` covers the code only.

- **Banking77**: Casanueva, I., Temčinas, T., Gerz, D., Henderson, M. and Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* Proceedings of the 2nd Workshop on NLP for ConvAI, ACL. https://aclanthology.org/2020.nlp4convai-1.5 . Original data by PolyAI (https://github.com/PolyAI-LDN/task-specific-datasets), licensed **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Loaded through the `mteb/banking77` mirror on Hugging Face.
- **CLINC150 (out-of-scope class only)**: Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., Leach, K., Laurenzano, M. A., Tang, L. and Mars, J. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1131 . Data from https://github.com/clinc/oos-eval via `clinc/clinc_oos` on Hugging Face, licensed **CC BY 3.0** (https://creativecommons.org/licenses/by/3.0/).

## Licence

Code: MIT (see `LICENSE`). Data: see above.
