from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union
import torch
from torch_geometric.data import DataLoader, Data, Batch

class M2NCustomCollater:
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Data]) -> Any:
        # Implement custom collation logic here
        batch_exc_dict = {}
        for i, data in enumerate(batch):
            data_dict = {}
            # You could choose to store data items in a dictionary by batch index
            for key, value in data.items():
                if key in self.exclude_keys:
                    data_dict[key] = value
            batch_exc_dict[i] = data_dict

        batched_data = Batch.from_data_list(batch, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
        batched_data.batch_dict = batch_exc_dict

        return batched_data

# Use the custom Collater in the DataLoader
# class M2N_DataLoader(DataLoader):
class Mixed_DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=None, exclude_keys=None, **kwargs):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        collater = M2NCustomCollater(follow_batch=follow_batch, exclude_keys=exclude_keys)
        self.collater = collater
        super().__init__(dataset, batch_size, shuffle, collate_fn=collater, **kwargs)
