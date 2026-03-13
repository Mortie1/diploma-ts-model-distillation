from torch.utils.data._utils.collate import default_collate


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    # Generic collation supports both classification and forecasting
    # tasks as long as each dataset item is a dict of collatable values.
    return default_collate(dataset_items)
