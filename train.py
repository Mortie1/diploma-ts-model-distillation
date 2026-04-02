import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.device import resolve_torch_device
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.run_metadata import apply_auto_run_metadata

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
        logger.info("torch.compile enabled for %s with args=%s", module_name, compile_kwargs)
        return compiled
    except Exception as e:
        logger.warning("Failed to compile %s via torch.compile: %s. Using eager mode.", module_name, e)
        return module


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
    apply_auto_run_metadata(config)

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

    # Backward compatibility: old scripts may still pass `model.provider=...`.
    if "provider" in config.model:
        with open_dict(config.model):
            config.model.pop("provider")

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    model = maybe_compile_module(model, config.trainer, logger, module_name="model")
    logger.info(model)

    distillation = None
    if config.get("distillation") is not None and config.distillation.get("enabled", False):
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
