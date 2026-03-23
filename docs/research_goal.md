# Canonical Research Goal

Source of truth (from `agents/dialogue.json` and supervisor discussion):

1. Evaluate strong time-series foundation-model baselines on multiple datasets.
2. Find regimes where these baselines underperform.
3. Distill time-series students from other modalities (primarily audio teachers,
   e.g., HuBERT/wav2vec2).
4. Compare distilled students vs TSFM baselines on quality and efficiency.

## Scope Priority

1. Cross-modal distillation objectives are primary.
2. Pure TSFM benchmarking is a baseline phase, not the final objective.
3. Experimental choices (models/datasets/configs) should be justified by how
   they help answer cross-modal distillation questions.

## Anti-Drift Checklist

Before adding new experiments, verify:

- Does this experiment help identify weak TSFM regimes OR improve/validate
  cross-modal distillation?
- Is there a clear comparison against TSFM-only baselines?
- Is the expected insight tied to the main thesis objective?
