"""
Utilities for confusing and backdooring forget sets
"""

import random
from typing import Optional, Mapping, Dict
import torch
from torch.utils.data import Dataset, Subset


def _different_random_label(y: int, num_classes: int, rng: random.Random) -> int:
    """Return a random label different from y"""
    if num_classes <= 1:
        return y
    r = rng.randrange(num_classes - 1)
    return r if r < y else r + 1


def _add_square_trigger(x, trigger_size: Optional[int] = None, trigger_value: Optional[float] = None):
    """Add a solid square trigger at bottom-right of image"""
    if isinstance(x, torch.Tensor):
        if x.dim() == 3:
            C, H, W = x.shape
            ts = trigger_size or max(3, min(H, W) // 10)
            v = trigger_value if trigger_value is not None else (255 if x.dtype == torch.uint8 else 1.0)
            x2 = x.clone()
            x2[..., H - ts:H, W - ts:W] = v
            return x2
        elif x.dim() == 2:
            H, W = x.shape
            ts = trigger_size or max(3, min(H, W) // 10)
            v = trigger_value if trigger_value is not None else (255 if x.dtype == torch.uint8 else 1.0)
            x2 = x.clone()
            x2[H - ts:H, W - ts:W] = v
            return x2
    return x


class MapLabelWrapper(Dataset):
    """Wrap a Subset and map specified labels to new values"""
    def __init__(self, subset: Subset, mapping: Mapping[int, int]):
        assert isinstance(subset, Subset), "Pass a torch.utils.data.Subset"
        self.subset = subset
        self.mapping: Dict[int, int] = dict(mapping)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        else:
            y = int(y)
        y_mapped = self.mapping.get(y, y)
        return x, y_mapped


class RandomLabelWrapper(Dataset):
    """Wrap a Subset and return randomized labels"""
    def __init__(self, subset: Subset, num_classes: Optional[int] = None, seed: Optional[int] = None):
        assert isinstance(subset, Subset), "Pass a torch.utils.data.Subset"
        self.subset = subset
        self.rng = random.Random(seed)
        
        labels = []
        for i in range(len(subset)):
            _, y = subset[i]
            if isinstance(y, torch.Tensor):
                y = int(y.item())
            labels.append(int(y))
        
        self.original_labels = labels
        if num_classes is None:
            num_classes = max(labels) + 1 if labels else 10
        self.num_classes = num_classes
        
        self.random_labels = [
            _different_random_label(y, self.num_classes, self.rng)
            for y in self.original_labels
        ]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, _ = self.subset[idx]
        return x, self.random_labels[idx]


class BackdoorWrapper(Dataset):
    """Wrap a Subset and add backdoor trigger"""
    def __init__(
        self,
        subset: Subset,
        target_label: int = 0,
        trigger_size: Optional[int] = None,
        trigger_value: Optional[float] = None,
    ):
        assert isinstance(subset, Subset), "Pass a torch.utils.data.Subset"
        self.subset = subset
        self.target_label = int(target_label)
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, _ = self.subset[idx]
        x_bd = _add_square_trigger(x, self.trigger_size, self.trigger_value)
        return x_bd, self.target_label


def confuse_the_forget_set(
    forget_set: Subset,
    confuse_map: Optional[Dict[int, int]] = None,
    num_classes: Optional[int] = None,
    seed: Optional[int] = None
) -> Dataset:
    """
    Return a dataset with confused/randomized labels
    """
    if confuse_map:
        return MapLabelWrapper(forget_set, mapping=confuse_map)
    return RandomLabelWrapper(forget_set, num_classes=num_classes, seed=seed)


def backdoor_the_forget_set(
    forget_set: Subset,
    target_label: int = 0,
    trigger_size: Optional[int] = None,
    trigger_value: Optional[float] = None,
) -> Dataset:
    """
    Return a dataset with backdoor triggers
    """
    return BackdoorWrapper(
        forget_set,
        target_label=target_label,
        trigger_size=trigger_size,
        trigger_value=trigger_value,
    )
