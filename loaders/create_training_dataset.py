import numpy as np
from torch.utils.data import Dataset, Sampler, IterableDataset
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import bisect
import warnings
from typing import (
    Iterable,
    List,
    Optional,
    TypeVar,
)
from operator import itemgetter
import torch
from .raft import RAFTExhaustiveDataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


dataset_dict = {
    'flow': RAFTExhaustiveDataset,
}


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
            self,
            sampler,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def increase_max_interval_by(self, increment):
        for dataset in self.datasets:
            curr_max_interval = dataset.max_interval.value
            dataset.max_interval.value = min(curr_max_interval + increment, dataset.num_imgs - 1)

    def set_max_interval(self, max_interval):
        for dataset in self.datasets:
            dataset.max_interval.value = min(max_interval, dataset.num_imgs - 1)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def get_training_dataset(args, max_interval):
    if '+' not in args.dataset_types:
        train_dataset = dataset_dict[args.dataset_types](args, max_interval=max_interval)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    else:
        dataset_types = args.dataset_types.split('+')
        weights = args.dataset_weights
        assert len(dataset_types) == len(weights)
        assert np.abs(np.sum(weights) - 1.) < 1e-6
        train_datasets = []
        train_weights_samples = []
        for dataset_type, weight in zip(dataset_types, weights):
            train_dataset = dataset_dict[dataset_type](args, max_interval=max_interval)
            train_datasets.append(train_dataset)
            num_samples = len(train_dataset)
            weight_each_sample = weight / num_samples
            train_weights_samples.extend([weight_each_sample]*num_samples)

        train_dataset = ConcatDataset(train_datasets)
        train_weights = torch.from_numpy(np.array(train_weights_samples))
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_sampler = DistributedSamplerWrapper(sampler) if args.distributed else sampler

    return train_dataset, train_sampler


