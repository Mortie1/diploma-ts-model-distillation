import warnings
from math import ceil

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.device import resolve_torch_device
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def maybe_compile_module(module, cfg_trainer, logger, module_name="model"):
    """
    Optionally apply torch.compile to a module from trainer config.
    """
    if not bool(cfg_trainer.get("compile_enabled", False)):
        return module
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is unavailable in this torch build. Skipping.")
        return module

    compile_kwargs = {}
    backend = cfg_trainer.get("compile_backend")
    mode = cfg_trainer.get("compile_mode")
    dynamic = cfg_trainer.get("compile_dynamic")
    fullgraph = cfg_trainer.get("compile_fullgraph")

    if backend:
        compile_kwargs["backend"] = backend
    if mode:
        compile_kwargs["mode"] = mode
    if dynamic is not None:
        compile_kwargs["dynamic"] = bool(dynamic)
    if fullgraph is not None:
        compile_kwargs["fullgraph"] = bool(fullgraph)

    try:
        compiled = torch.compile(module, **compile_kwargs)
        logger.info(
            "torch.compile enabled for %s with args=%s", module_name, compile_kwargs
        )
        return compiled
    except Exception as e:
        logger.warning(
            "Failed to compile %s via torch.compile: %s. Using eager mode.",
            module_name,
            e,
        )
        return module


def maybe_resolve_cosine_t_max(config, dataloaders, logger):
    """
    Auto-resolve T_max for CosineAnnealingLR when it is omitted or set to "auto".

    Scheduler steps in this codebase happen on optimizer.step(), i.e. once per
    gradient-accumulated optimization step.
    """
    sched_cfg = config.get("lr_scheduler")
    if sched_cfg is None:
        return

    target = str(sched_cfg.get("_target_", ""))
    if target != "torch.optim.lr_scheduler.CosineAnnealingLR":
        return

    t_max = sched_cfg.get("T_max")
    if t_max not in (None, "auto"):
        return

    train_loader = dataloaders.get("train")
    if train_loader is None:
        raise ValueError("Cannot auto-resolve T_max: missing train dataloader.")

    steps_per_epoch = config.trainer.get("epoch_len")
    if steps_per_epoch is None:
        steps_per_epoch = len(train_loader)
    steps_per_epoch = int(steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError(
            f"Cannot auto-resolve T_max: invalid steps_per_epoch={steps_per_epoch}."
        )

    grad_accum_steps = int(config.trainer.get("grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError("trainer.grad_accum_steps must be >= 1")

    opt_steps_per_epoch = ceil(steps_per_epoch / grad_accum_steps)
    n_epochs = int(config.trainer.n_epochs)
    resolved_t_max = max(1, n_epochs * opt_steps_per_epoch)

    OmegaConf.set_struct(config, False)
    config.lr_scheduler.T_max = resolved_t_max
    OmegaConf.set_struct(config, True)
    logger.info(
        "Auto-resolved CosineAnnealingLR.T_max=%d "
        "(n_epochs=%d, steps_per_epoch=%d, grad_accum_steps=%d, opt_steps_per_epoch=%d)",
        resolved_t_max,
        n_epochs,
        steps_per_epoch,
        grad_accum_steps,
        opt_steps_per_epoch,
    )


def _resolve_total_optimizer_steps(config, dataloaders):
    train_loader = dataloaders.get("train")
    if train_loader is None:
        raise ValueError(
            "Cannot resolve total optimizer steps: missing train dataloader."
        )

    steps_per_epoch = config.trainer.get("epoch_len")
    if steps_per_epoch is None:
        steps_per_epoch = len(train_loader)
    steps_per_epoch = int(steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError(f"Invalid steps_per_epoch={steps_per_epoch}")

    grad_accum_steps = int(config.trainer.get("grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError("trainer.grad_accum_steps must be >= 1")

    opt_steps_per_epoch = ceil(steps_per_epoch / grad_accum_steps)
    n_epochs = int(config.trainer.n_epochs)
    total_opt_steps = max(1, n_epochs * opt_steps_per_epoch)
    return total_opt_steps, steps_per_epoch, grad_accum_steps, opt_steps_per_epoch


def maybe_wrap_with_warmup_scheduler(
    config, optimizer, lr_scheduler, dataloaders, logger
):
    """
    Optionally prepend linear warmup to the main LR scheduler.
    Warmup length can be set via trainer.warmup_steps or trainer.warmup_ratio.
    """
    warmup_steps_cfg = config.trainer.get("warmup_steps")
    warmup_ratio = float(config.trainer.get("warmup_ratio", 0.0))
    if warmup_steps_cfg is None and warmup_ratio <= 0.0:
        return lr_scheduler

    (
        total_steps,
        steps_per_epoch,
        grad_accum_steps,
        opt_steps_per_epoch,
    ) = _resolve_total_optimizer_steps(config, dataloaders)
    if warmup_steps_cfg is None:
        warmup_steps = int(total_steps * warmup_ratio)
    else:
        warmup_steps = int(warmup_steps_cfg)

    # Warmup should leave at least one step for the base scheduler.
    warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
    if warmup_steps <= 0:
        return lr_scheduler

    start_factor = float(config.trainer.get("warmup_start_factor", 0.1))
    if start_factor <= 0.0 or start_factor > 1.0:
        raise ValueError("trainer.warmup_start_factor must be in (0, 1].")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    if lr_scheduler is None:
        logger.info(
            "Enabled linear warmup only: warmup_steps=%d/%d, start_factor=%.4f",
            warmup_steps,
            total_steps,
            start_factor,
        )
        return warmup_scheduler

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, lr_scheduler],
        milestones=[warmup_steps],
    )
    logger.info(
        "Enabled LR warmup: warmup_steps=%d/%d (ratio=%.4f), start_factor=%.4f, "
        "steps_per_epoch=%d, grad_accum_steps=%d, opt_steps_per_epoch=%d",
        warmup_steps,
        total_steps,
        (warmup_steps / float(total_steps)),
        start_factor,
        steps_per_epoch,
        grad_accum_steps,
        opt_steps_per_epoch,
    )
    return scheduler


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    device = resolve_torch_device(
        config.trainer.device,
        require_cuda=config.trainer.get("require_cuda", False),
        logger=logger,
        context="training",
    )

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)
    maybe_resolve_cosine_t_max(config, dataloaders, logger)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    model = maybe_compile_module(model, config.trainer, logger, module_name="model")
    logger.info(model)

    distillation = None
    if config.get("distillation") is not None and config.distillation.get(
        "enabled", False
    ):
        distillation = instantiate(config.distillation).to(device)
        if config.trainer.get("compile_distillation", False):
            distillation = maybe_compile_module(
                distillation, config.trainer, logger, module_name="distillation"
            )
        logger.info("Distillation is enabled")

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if distillation is not None:
        distill_params = list(
            filter(lambda p: p.requires_grad, distillation.parameters())
        )
        trainable_params.extend(distill_params)
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    lr_scheduler = maybe_wrap_with_warmup_scheduler(
        config=config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloaders=dataloaders,
        logger=logger,
    )

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        distillation=distillation,
    )

    trainer.train()


if __name__ == "__main__":
    main()
