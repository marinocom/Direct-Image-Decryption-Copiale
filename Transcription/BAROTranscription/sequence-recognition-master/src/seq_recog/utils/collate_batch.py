from dataclasses import fields
from typing import List

from ..data.base_dataset import BatchedSample


def collate_batch(batch: List[BatchedSample]):
    field_names = {field.name: field.type for field in fields(batch[0])}
    batch_dict = {name: [] for name in field_names}

    for item in batch:
        for field_name in field_names.keys():
            batch_dict[field_name].append(getattr(item, field_name))

    return type(batch[0])(**batch_dict)
