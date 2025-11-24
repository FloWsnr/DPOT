from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from gphyt.data.well_dataset import WellDataset, ZScoreNormalization, StrideError


def get_phys_dataset(
    data_dir: Path,
    use_normalization: bool = True,
    T_in: int = 10,
    T_out: int = 1,
    dt_stride: int | list[int] = 1,
    full_trajectory_mode: bool = False,
    nan_to_zero: bool = True,
    train: bool = True,
) -> Optional["PhysicsDataset"]:
    """Helper function to create a PhysicsDataset."""
    try:
        return PhysicsDataset(
            data_dir=data_dir,
            T_in=T_in,
            T_out=T_out,
            use_normalization=use_normalization,
            dt_stride=dt_stride,
            full_trajectory_mode=full_trajectory_mode,
            nan_to_zero=nan_to_zero,
            train=train,
        )
    except StrideError as e:
        print(f"Error creating PhysicsDataset for {data_dir}: {e}")
        return None


class PhysicsDataset(WellDataset):
    """Wrapper around the WellDataset.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory (e.g. "data/physics_data/train")
    use_normalization: bool
        Whether to use normalization
        By default False
    T_in : int
        Number of input time steps
        By default 10
    T_out : int
        Number of output time steps
        By default 1
    dt_stride: int | list[int]
        Time step stride between samples
        By default 1
    full_trajectory_mode: bool
        Whether to use the full trajectory mode of the well dataset.
        This returns full trajectories instead of individual timesteps.
        By default False
    nan_to_zero: bool
        Whether to replace NaNs with 0
        By default True
    train: bool
        Whether to use the training split
        By default True
    """

    def __init__(
        self,
        data_dir: Path,
        use_normalization: bool = True,
        T_in: int = 10,
        T_out: int = 1,
        dt_stride: int | list[int] = 1,
        full_trajectory_mode: bool = False,
        nan_to_zero: bool = True,
        train: bool = True,
    ):
        self.config = {
            "data_dir": data_dir,
            "use_normalization": use_normalization,
            "T_in": T_in,
            "T_out": T_out,
            "dt_stride": dt_stride,
            "full_trajectory_mode": full_trajectory_mode,
            "nan_to_zero": nan_to_zero,
            "train": train,
        }

        self.train = train

        if isinstance(dt_stride, list):
            min_dt_stride = dt_stride[0]
            max_dt_stride = dt_stride[1]
        else:
            min_dt_stride = dt_stride
            max_dt_stride = dt_stride

        # search for stats.yaml in parent directory
        for p in (0, 1):
            n_path = data_dir.parents[p] / "stats.yaml"
            if n_path.exists():
                break

        super().__init__(
            path=str(data_dir),
            normalization_path=str(n_path),
            n_steps_input=T_in,
            n_steps_output=T_out,
            use_normalization=use_normalization,
            normalization_type=ZScoreNormalization,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            full_trajectory_mode=full_trajectory_mode,
        )
        self.nan_to_zero = nan_to_zero
        # give the dataset its correct name
        name = data_dir.parents[1].name
        self.dataset_name = name

    def copy(self, overwrites: dict[str, Any] = {}) -> Optional["PhysicsDataset"]:
        """Copy the dataset with optional overwrites.

        Useful for creating a new dataset with slightly different parameters.

        Parameters
        ----------
        overwrites : dict[str, Any]
            Dictionary of overwrites for the config.

        Returns
        -------
        PhysicsDataset
            New PhysicsDataset with the updated config.
            Returns None if the dataset could not be created due to too large stride.
        """
        config = self.config.copy()
        config.update(overwrites)
        return get_phys_dataset(
            data_dir=config["data_dir"],
            T_in=config["T_in"],
            T_out=config["T_out"],
            use_normalization=config["use_normalization"],
            dt_stride=config["dt_stride"],
            full_trajectory_mode=config["full_trajectory_mode"],
            nan_to_zero=config["nan_to_zero"],
            train=config["train"],
        )

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, _ = super().__getitem__(index)  # returns (T, h, w, c)
        x = data["input_fields"]
        y = data["output_fields"]

        if self.nan_to_zero:
            x = torch.nan_to_num(x, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)

        if self.train:
            # reshape to (c, h, w)
            x = rearrange(x, "t h w c -> t c h w")
            y = rearrange(y, "t h w c -> t c h w")

            # interpolate to 128x128
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)

            # reshape to (h,w,t, c)
            x = rearrange(x, "t c h w -> h w t c")
            y = rearrange(y, "t c h w -> h w t c")
        else:
            # reshape to (h,w,t, c)
            x = rearrange(x, "t h w c -> h w t c")
            y = rearrange(y, "t h w c -> h w t c")

        h, w, t, c = x.shape

        mask = torch.ones((h, w, 1, c), dtype=torch.float32)

        return x, y, mask


class SuperDataset:
    """Wrapper around a list of datasets.

    Allows to concatenate multiple datasets and randomly sample from them.

    Parameters
    ----------
    datasets : dict[str, PhysicsDataset]
        Dictionary of datasets to concatenate

    max_samples_per_ds : Optional[int | list[int]]
        Maximum number of samples to sample from each dataset.
        If a list, specifies the number of samples for each dataset individually.
        If None, uses all samples from each dataset.
        By default None.

    dataset_to_class_idx : Optional[dict[str, int]]
        Mapping from dataset name to original class index.
        By default None.

    seed : Optional[int]
        Random seed for reproducibility.
        By default None.
    """

    def __init__(
        self,
        datasets: dict[str, PhysicsDataset],
        max_samples_per_ds: Optional[int | list[int]] = None,
        dataset_to_class_idx: Optional[dict[str, int]] = None,
        seed: Optional[int] = None,
    ):
        self.datasets = datasets
        self.dataset_list = list(datasets.values())
        self.dataset_names = list(datasets.keys())
        self.dataset_to_class_idx = dataset_to_class_idx

        if isinstance(max_samples_per_ds, int):
            self.max_samples_per_ds = [max_samples_per_ds] * len(datasets)
        else:
            self.max_samples_per_ds = max_samples_per_ds

        self.seed = seed

        # Initialize random number generator
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Generate initial random indices
        self.reshuffle()

    def reshuffle(self):
        """Reshuffle the indices for each dataset.

        This should be called at the start of each epoch to ensure
        a new random subset of samples is used.

        """
        self.dataset_indices = []
        for i, dataset in enumerate(self.dataset_list):
            if (
                self.max_samples_per_ds is not None
                and len(dataset) > self.max_samples_per_ds[i]
            ):
                indices = torch.randperm(len(dataset), generator=self.rng)[
                    : self.max_samples_per_ds[i]
                ]
                self.dataset_indices.append(indices)
            else:
                self.dataset_indices.append(None)

        # Calculate lengths based on either max_samples_per_ds or full dataset length
        self.lengths = [
            min(self.max_samples_per_ds[i], len(dataset))
            if self.max_samples_per_ds is not None
            else len(dataset)
            for i, dataset in enumerate(self.dataset_list)
        ]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[int]]:
        for i, length in enumerate(self.lengths):
            if index < length:
                if self.dataset_indices[i] is not None:
                    # Use random index if available
                    actual_index = self.dataset_indices[i][index]
                else:
                    actual_index = index
                x, y, mask = self.dataset_list[i][
                    actual_index
                ]  # (time, h, w, n_channels)

                # Map dataset variant to original class index
                if self.dataset_to_class_idx is not None:
                    dataset_name = self.dataset_names[i]
                    class_idx = self.dataset_to_class_idx[dataset_name]
                else:
                    class_idx = i
                break
            index -= length
        return x, y, mask, class_idx


def get_dataset(
    path: Path,
    split_name: str,
    datasets: list,
    min_stride: int = 1,
    max_stride: int = 1,
    T_in: int = 10,
    T_out: int = 1,
    use_normalization: bool = True,
    full_trajectory_mode: bool = False,
    nan_to_zero: bool = True,
    train: bool = True,
    return_super_dataset: bool = True,
) -> SuperDataset | dict[str, PhysicsDataset]:
    """ """

    all_ds = {}
    dataset_to_class_idx = {}  # Map dataset variant name to original dataset index
    for ds_idx, ds_name in enumerate(datasets):
        ds_path = Path(path) / f"{ds_name}/data/{split_name}"
        if ds_path.exists():
            for stride in range(min_stride, max_stride + 1):
                name = f"{ds_name}_stride{stride}"
                dataset = get_phys_dataset(
                    data_dir=Path(path) / f"{ds_name}/data/{split_name}",
                    use_normalization=use_normalization,
                    T_in=T_in,
                    T_out=T_out,
                    dt_stride=stride,
                    full_trajectory_mode=full_trajectory_mode,
                    nan_to_zero=nan_to_zero,
                    train=train,
                )
                if dataset is None:
                    continue
                all_ds[name] = dataset
                dataset_to_class_idx[name] = ds_idx  # Map to original dataset index

        else:
            print(f"Dataset path {ds_path} does not exist. Skipping.")

    if not return_super_dataset:
        return all_ds
    else:
        return SuperDataset(all_ds, dataset_to_class_idx=dataset_to_class_idx)
