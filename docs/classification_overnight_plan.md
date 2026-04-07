# Overnight Classification Experiment Plan

Branch: `exp/classification-overnight-plan`

## Goals
1. Debug run matrix in offline mode (1-2 epochs), tune batch size for VRAM fit and throughput.
2. Launch scientific runs online in Comet with strict naming and tagging.
3. Keep runs autonomous for 7-9 hours with clear stop/retry behavior.

## Critical constraints
- No swap/spill to RAM. If CUDA OOM appears, reduce batch size.
- If VRAM headroom is high and no OOM, increase batch size for faster experiments.
- Debug runs: `writer.mode=offline`.
- Scientific runs: `writer.mode=online`.

## Datasets
- `pamap2`
- `ptbxl`
- `insect_wingbeat`
- `cwru_bearing`

## Model families (minimum)
- `moment`
- `chronos` (see blocker below)
- `tirex`
- `tspulse`

## Required tags per run
Always include:
- `classification`
- family: one of `moment|chronos|chronos2|tirex|tspulse`
- size token: e.g. `moment-small`, `chronos2-base`, `tirex-base`, `tspulse-r1`
- dataset: `pamap2|ptbxl|insect_wingbeat|cwru`

## Run naming rule (<=40 chars)
Format:
- `<family>-<size>-<ds>-<stage>-<keychg>`
Examples:
- `moment-s-pamap2-db-bs128`
- `tirex-b-ptbxl-main-bs64`
- `chron2-b-cwru-main-lora`

## Phase A: Offline debug + batch-size tuning
Per (model, dataset):
1. Start with conservative batch size:
   - moment small/base: 64
   - chronos2 base: 16
   - tirex base: 32
   - tspulse r1: 16
2. Run 1 epoch, `trainer.epoch_len=50`, `writer.mode=offline`.
3. Parse logs:
   - if OOM -> halve batch size and retry.
   - if no OOM and stable -> double batch size until first OOM, then step back.
4. Save tuned batch size for Phase B.

## Phase B: Online scientific runs
For each (model-size, dataset), run:
- `n_epochs=10`
- tuned `batch_size` from Phase A
- `trainer.monitor=max val_MacroF1`
- tags according to rule above
- compact run_name according to rule above

## Phase C: Extensions (if time remains)
- additional sizes:
  - moment: `small/base/large`
  - chronos2: `base` + LoRA/top-k variants
- optional replicate seed for top performers.

## Retry policy
- Provider init/download fail: retry once after 60s.
- OOM: auto reduce batch and retry.
- Dataset download fail: mark as blocked and continue other runs.

## Night execution budget
Recommended budget for 7-9h:
- Phase A: ~1-2h
- Phase B core matrix: ~4-6h
- Phase C extensions: remaining time

## Known blockers to resolve before overnight main run
1. `ChronosClassificationAdapter` currently returns `None` in `_init_provider_model` and is not runnable as a provider-only classifier.
2. `TSPulseClassificationAdapter` may depend on remote-code model compatibility in local `transformers` version.
3. `fm_classification_train` default currently points to missing `model/moment_classification_adapter`; use `fm_train` or fix default target before scheduler run.

## Decision needed
- Use `chronos2` instead of `chronos` for this overnight cycle, or implement a trainable `chronos` classifier adapter first.
