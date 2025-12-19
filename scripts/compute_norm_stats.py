"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""
import sys
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import dataclasses
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('lerobot_http.log')
    ]
)

logging.info(f"[DataConfig] {_config.DataConfig.__dataclass_fields__.keys()}")

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, 
         max_frames: int | None = None,
         repo_id: str | None = None,
         asset_id: str | None = None,
         rlds_data_dir: str | None = None
    ) -> None:
    config = _config.get_config(config_name)
    logging.warning(f"[data_config] ================================================")
    logging.warning(f"[config] {config}")
    logging.warning(f"[config.assets_dirs] {config.assets_dirs}")
    logging.warning(f"[config.model] {config.model}")
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.warning(f"[data_config] ================================================")
    logging.warning(f"[data_config] {data_config}")
    logging.warning(f"[data_config.repo_id] before: {data_config.repo_id}")
    logging.warning(f"[data_config.asset_id] before: {data_config.asset_id}")
    logging.warning(f"[data_config.rlds_data_dir] before: {data_config.rlds_data_dir}")

    # 使用 dataclasses.replace() 创建新的配置对象，而不是直接修改字段
    if repo_id is not None or asset_id is not None or rlds_data_dir is not None:
        data_config = dataclasses.replace(
            data_config,
            repo_id=repo_id if repo_id is not None else data_config.repo_id,
            asset_id=asset_id if asset_id is not None else data_config.asset_id,
            rlds_data_dir=rlds_data_dir if rlds_data_dir is not None else data_config.rlds_data_dir
        )
        src_repo_id = data_config.repo_id
    else:
        src_repo_id = data_config.repo_id
    logging.warning(f"[data_config] ================================================")
    logging.warning(f"[data_config.repo_id] after: {data_config.repo_id}")
    logging.warning(f"[data_config.asset_id] after: {data_config.asset_id}")
    logging.warning(f"[data_config.rlds_data_dir] after: {data_config.rlds_data_dir}")
    logging.warning(f"[data_config] ================================================")

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # output_path = config.assets_dirs / data_config.repo_id
    output_path = config.assets_dirs / src_repo_id
    print(f"[data_config] ================================================")
    print(f"[config.assets_dirs] {config.assets_dirs}")
    print(f"[src_repo_id] {src_repo_id}")
    print(f"Writing stats to: {output_path}")
    print(f"[data_config] ================================================")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
