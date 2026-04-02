# PyTorch Template for DL projects

<p align="center">
  <a href="#about">About</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/CITATION.cff">
   <img src="https://img.shields.io/badge/cite-this%20repo-purple">
</a>
</p>

## About

This repository contains a template for [PyTorch](https://pytorch.org/)-based Deep Learning projects.

The template utilizes different python-dev techniques to improve code readability. Configuration methods enhance reproducibility and experiments control.

The repository is released as a part of the [HSE DLA course](https://github.com/markovka17/dla), however, can easily be adopted for any DL-task.

This template is the official recommended template for the [EPFL CS-433 ML Course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/).

> 📖 **If you use this template in your work, please cite this repository or include a reference. Attribution supports the project and encourages continued development.**

## Tutorials

This template utilizes experiment tracking techniques, such as [WandB](https://docs.wandb.ai/) and [Comet ML](https://www.comet.com/docs/v2/), and [Hydra](https://hydra.cc/docs/intro/) for the configuration. It also automatically reformats code and conducts several checks via [pre-commit](https://pre-commit.com/). If you are not familiar with these tools, we advise you to look at the tutorials below:

- [Python Dev Tips](https://github.com/ebezzam/python-dev-tips): information about [Git](https://git-scm.com/doc), [pre-commit](https://pre-commit.com/), [Hydra](https://hydra.cc/docs/intro/), and other stuff for better Python code development. The YouTube recording of the workshop is available [here](https://youtu.be/okxaTuBdDuY).

- [Seminar on R&D Coding 2025](https://youtu.be/PE1zaW5it_A): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with discussion on logging, project-based coding, configuration, and reproducibility. The materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/tree/summer25/day05).

- [Seminar on R&D Coding 2024](https://youtu.be/sEA-Js5ZHxU): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with template discussion and reasoning. It also explains how to work with [WandB](https://docs.wandb.ai/). The seminar materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/blob/main/day03/Seminar_WandB_and_Coding.ipynb).

- [HSE DLA Course Introduction Week](https://github.com/markovka17/dla/tree/2024/week01): combines the two seminars above into one with some updates, including an extra example for [Comet ML](https://www.comet.com/docs/v2/).

- [PyTorch Basics](https://github.com/markovka17/dla/tree/2024/week01/intro_to_pytorch): several notebooks with [PyTorch](https://pytorch.org/docs/stable/index.html) basics and corresponding seminar recordings from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/).

To start working with a template, just click on the `use this template` button.

<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>

You can choose any of the branches as a starting point. [Set your choice as the default branch](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch) in the repository settings. You can also [delete unnecessary branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).

## Examples

> [!IMPORTANT]
> The main branch leaves some of the code parts empty or fills them with dummy examples, showing just the base structure. The final users can add code required for their own tasks.

You can find examples of this template completed for different tasks in other branches:

- [Image classification](https://github.com/Blinorot/pytorch_project_template/tree/example/image-classification): simple classification problem on [MNIST](https://yann.lecun.com/exdb/mnist/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.

- [ASR](https://github.com/Blinorot/pytorch_project_template/tree/example/asr): template for the automatic speech recognition (ASR) task. Some of the parts (for example, `collate_fn` and beam search for `text_encoder`) are missing for studying purposes of [HSE DLA course](https://github.com/markovka17/dla).

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

### Experiment Presets (Diploma)

The repository now includes a forecasting-focused track with cross-modal distillation.

Run presets:

```bash
# Forecasting baseline
python3 train.py -cn=forecasting_baseline

# Distillation train/inference base
python3 train.py -cn=distill_train
python3 inference.py -cn=distill_inference
```

To force GPU usage (fail fast if CUDA is not usable):

```bash
python3 train.py -cn=fm_train trainer.require_cuda=true
python3 inference.py -cn=fm_inference inferencer.require_cuda=true
```

### Popular TSFM Benchmark Datasets

The repo now includes configs for commonly used TSFM forecasting datasets:

- Forecasting (ETT): `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`
- Forecasting (LTSF): `Electricity`, `Traffic`, `Weather`

Datasets are downloaded automatically on first use if missing locally.
By default they are stored under `data/raw/ett` and `data/raw/ltsf`.

### Motivation: Why These Models and Datasets

For the diploma track, the main goal is cross-modal distillation for time
series (for example, audio-teacher -> TS student). The forecasting FM setup is
used as a controlled baseline stage before distillation.

- Baseline model families:
  - `Chronos` (`chronos-t5-small`, `chronos-2`) to compare previous vs newer
    generations within one line.
  - `TimesFM` (`2.0`, `2.5`) to compare another strong foundation-model line
    and quantify version improvements.
- Adaptation regimes:
  - `full` fine-tuning where feasible.
  - `LoRA` for larger checkpoints to keep memory usage realistic and compare
    quality/cost trade-offs.
- Forecasting datasets:
  - `Electricity`, `Traffic`, `Weather` (LTSF benchmarks) to cover different
    dynamics (seasonality, noisy demand/traffic, smoother physical signals).
  - Using several datasets reduces the chance of dataset-specific conclusions.

Research intent:

1. Establish strong TSFM baselines and identify where they underperform
   (datasets/domains/signal regimes).
2. Distill knowledge from non-TS teachers (primarily audio foundation models)
   into TS students and compare against TSFM-only baselines.
3. Measure whether cross-modal distillation is a better quality/efficiency
   trade-off than using larger TS foundation models alone.

Extended report-style motivation is available in
`docs/report_motivation.md`. Canonical research goal is tracked in
`docs/research_goal.md`.

Run examples:

```bash
# ETT ETTh1 baseline
python3 train.py -cn=forecasting_etth1_baseline trainer.require_cuda=true
```

Foundation-model base configs (train + test inference):

```bash
# Forecasting FM train/inference base
python3 train.py -cn=fm_train trainer.require_cuda=true
python3 inference.py -cn=fm_inference inferencer.require_cuda=true
```

Switch provider/model/dataset via overrides:

```bash
# TimesFM-2.5 on Electricity
python3 train.py -cn=fm_train \
  datasets=ltsf_electricity_h96 \
  model.provider=timesfm_hf \
  model.model_id=google/timesfm-2.5-200m-pytorch \
  model.in_channels=321 \
  model.horizon=96

# TimesFM-2.0 API path
python3 train.py -cn=fm_train \
  datasets=ltsf_electricity_h96 \
  model.provider=timesfm \
  model.model_id=google/timesfm-2.0-500m-pytorch \
  model.in_channels=321 \
  model.horizon=96

# Chronos-2
python3 train.py -cn=fm_train \
  datasets=ltsf_electricity_h96 \
  model.provider=chronos \
  model.model_id=amazon/chronos-2 \
  model.in_channels=321 \
  model.horizon=96
```

Full fine-tuning and LoRA (forecasting FM):

```bash
# full fine-tune (example)
python3 train.py -cn=fm_train \
  model.provider=timesfm_hf \
  model.model_id=google/timesfm-2.5-200m-pytorch \
  model.finetune_mode=full

# LoRA fine-tune (example)
python3 train.py -cn=fm_train \
  model.provider=timesfm_hf \
  model.model_id=google/timesfm-2.5-500m-pytorch \
  model.finetune_mode=lora \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.lora_target_patterns='[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj]'
```

Notes:
- `model.require_provider_model=true` makes runs fail fast if the backbone is not loaded.
- For providers exposed as inference-only pipelines, use `finetune_mode=none`.

Inference + bootstrap CI evaluation:

```bash
python3 inference.py -cn=fm_inference
```

For audio foundation teachers via HF, set:

```bash
python3 train.py -cn=distill_train \
  distillation.teacher_backend=hf \
  distillation.teacher_model_name=facebook/hubert-base-ls960
```

## Useful Links:

You may find the following links useful:

- [Report branch](https://github.com/Blinorot/pytorch_project_template/tree/report): Guidelines for writing a scientific report/paper (with an emphasis on DL projects).

- [CLAIRE Template](https://github.com/CLAIRE-Labo/python-ml-research-template): additional template by [EPFL CLAIRE Laboratory](https://www.epfl.ch/labs/claire/) that can be combined with ours to enhance experiments reproducibility via [Docker](https://www.docker.com/).

- [Mamba](https://github.com/mamba-org/mamba) and [Poetry](https://python-poetry.org/): alternatives to [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) and [pip](https://pip.pypa.io/en/stable/installation/) package managers given above.

- [Awesome README](https://github.com/matiassingers/awesome-readme): a list of awesome README files for inspiration. Check the basics [here](https://github.com/PurpleBooth/a-good-readme-template).

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
