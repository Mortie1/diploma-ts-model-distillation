# Motivation For Model and Dataset Selection

## Goal

Primary goal: evaluate whether cross-modal distillation (especially from audio
foundation models) can outperform or match pure TS foundation-model usage on
time-series tasks with better efficiency.

Secondary goal: build a strong and reproducible TSFM baseline layer to identify
the conditions where cross-modal distillation is most useful.

## Why These Foundation Models (Baseline Stage)

We selected two major foundation-model families and multiple generations:

1. Chronos family:
   - `amazon/chronos-t5-small`
   - `amazon/chronos-2`
2. TimesFM family:
   - `google/timesfm-2.0-500m-pytorch`
   - `google/timesfm-2.5-200m-pytorch`
   - `google/timesfm-2.5-500m-pytorch`

Rationale for baseline stage:

- Cross-family comparison:
  Chronos and TimesFM are built with different modeling assumptions and APIs,
  so comparing both helps avoid conclusions tied to only one family.
- Within-family version comparison:
  including older and newer generations allows measuring whether updates produce
  practical gains under identical training/evaluation conditions.
- Parameter-scale comparison:
  200M vs 500M checkpoints helps quantify quality/speed/memory trade-offs.

## Why Full Fine-Tuning and LoRA

Both full fine-tuning and LoRA are included because they answer different
questions in the baseline and in the distillation setting:

- Full fine-tuning:
  upper bound on adaptation when memory budget allows.
- LoRA:
  memory-efficient adaptation for larger checkpoints and constrained hardware.

This is necessary for a fair “quality vs cost” analysis and for realistic
student-teacher experiments under limited VRAM.

## Why These Datasets

For forecasting, we focus on widely used LTSF benchmarks:

- `Electricity`
- `Traffic`
- `Weather`

Rationale:

- They represent different temporal regimes:
  - Electricity: strong seasonality and high-dimensional multivariate signals.
  - Traffic: noisy dynamics and abrupt changes.
  - Weather: smoother physical dynamics with long dependencies.
- Multi-dataset evaluation reduces the risk of overfitting conclusions to one
  benchmark.
- They are standard enough for comparison with prior work and practical for
  reproducible local experiments.

## Design Constraints and Fairness

To keep results comparable:

1. Same task setup (forecasting horizon and split logic) across models.
2. Same training budget policy for each comparison block.
3. Same logging/metric pipeline (Comet + identical evaluation metrics).
4. Explicit run naming with main-change annotations for auditability.

## Expected Outcomes

The selected setup should answer:

1. Where exactly TSFM baselines underperform (by dataset/domain/signal type).
2. Whether cross-modal teachers (audio) improve TS students on these weak
   regimes beyond TSFM-only adaptation.
3. When LoRA-distilled students are close enough to full fine-tuning to justify
   lower compute/memory cost.
4. Which student/teacher regime is the best default for iterative research on
   limited hardware.
