# Experiment Naming Rules

These rules define how run names are generated for CometML and local save directories.

## Goals

- Make each run easy to identify in Comet at a glance.
- Keep local save directory names filesystem-safe.
- Always include the main difference from baseline in the short name.

## Display Name Format

Comet display name uses:

`<short_name-with-main-change> / lr=<value> / bs=<value>`

Example:

`forecasting-chronos-h96-lr0.0003 / lr=0.0003 / bs=128`

## How `short_name-with-main-change` is built

Base short name:

- `task-provider-h<horizon>`
- plus `-<finetune_mode>` if finetuning is enabled
- plus `-distill` if distillation is enabled

Main change is appended as `-<main_change>`.

Priority for main change:

1. `writer.main_change` if explicitly provided in CLI/config.
2. Auto-detected:
   - `ft-<mode>` if `model.finetune_mode != none`
   - `res<value>` if `model.residual_scale != 1.0`
   - `lr<value>` if `optimizer.lr != 1e-3`
   - `bs<value>` if `dataloader.batch_size != 64`
   - `<provider>` if provider is not `chronos`
   - `base` otherwise

## Local `run_name` (save dir)

`run_name` is a slugified version of:

`<short_name-with-main-change>__<model_id_short>__lr=<value>__bs=<value>`

This avoids path issues with spaces and slashes.

## Manual override examples

Use custom short label:

```bash
.venv/bin/python train.py -cn=fm_train writer.short_name="chronos-speed"
```

Use explicit main change label:

```bash
.venv/bin/python train.py -cn=fm_train writer.main_change="bs128"
```
