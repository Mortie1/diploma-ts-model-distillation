import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import open_dict

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.device import resolve_torch_device
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    device = resolve_torch_device(
        config.inferencer.device,
        require_cuda=config.inferencer.get("require_cuda", False),
        context="inference",
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
    print(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
