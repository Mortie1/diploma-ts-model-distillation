# Forecasting Research Assets (2026)

## Added foundation-model runs/configs

Base FM configs:
- `src/configs/fm_train.yaml`
- `src/configs/fm_inference.yaml`

1. `amazon/chronos-2` (Chronos-2)
   - Example override:
     `python3 train.py -cn=fm_train datasets=ltsf_electricity_h96 model.provider=chronos model.model_id=amazon/chronos-2 model.in_channels=321 model.horizon=96`
   - Source: https://arxiv.org/abs/2510.15821
   - Model card: https://huggingface.co/amazon/chronos-2

2. `google/timesfm-2.5-200m-pytorch` (TimesFM 2.5)
   - Example override:
     `python3 train.py -cn=fm_train datasets=ltsf_electricity_h96 model.provider=timesfm_hf model.model_id=google/timesfm-2.5-200m-pytorch model.in_channels=321 model.horizon=96 model.finetune_mode=full`
   - Base paper (TimesFM): https://arxiv.org/abs/2310.10688
   - Model card: https://huggingface.co/google/timesfm-2.5-200m-pytorch

3. `google/timesfm-2.5-500m-pytorch` (TimesFM 2.5 + LoRA)
   - Example override:
     `python3 train.py -cn=fm_train datasets=ltsf_electricity_h96 model.provider=timesfm_hf model.model_id=google/timesfm-2.5-500m-pytorch model.in_channels=321 model.horizon=96 model.finetune_mode=lora model.lora_rank=8 model.lora_alpha=16 model.lora_target_patterns='[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj]'`
   - Base paper (TimesFM): https://arxiv.org/abs/2310.10688
   - Model card: https://huggingface.co/google/timesfm-2.5-500m-pytorch

## Added forecasting datasets (auto-download in dataset class)

All datasets are downloaded automatically from:
https://huggingface.co/datasets/thuml/Time-Series-Library

Added configs:

1. Electricity
   - `src/configs/datasets/ltsf_electricity_h96.yaml`
2. Traffic
   - `src/configs/datasets/ltsf_traffic_h96.yaml`
3. Weather
   - `src/configs/datasets/ltsf_weather_h96.yaml`

Benchmark note:
The `thuml/Time-Series-Library` dataset card lists these datasets as core long-term forecasting benchmarks and provides direct `hf_hub_download` usage.
