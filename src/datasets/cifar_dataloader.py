"""
Your complete dataloader code exactly as provided
Balanced partitioning with forgetting support for CIFAR-10, MNIST, FashionMNIST
"""

import copy
import random
import os
from collections import defaultdict
from typing import Dict, Tuple, Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import datasets

from .transformers_utils import confuse_the_forget_set, backdoor_the_forget_set


def _partition_dataset(dataset, num_partitions, partition_id, shuffle):
    """Partition dataset with balanced class distribution"""
    label_to_indices = defaultdict(lambda: [])
    target_list = dataset.targets
    if type(target_list) != type([0, 1]):
        target_list = target_list.tolist()
    for idx, label in enumerate(target_list):
        label_to_indices[label].append(idx)

    partition_indices = []

    for label, indices in label_to_indices.items():
        base_size = len(indices) // num_partitions
        extra = len(indices) % num_partitions
        start_idx = sum(base_size + 1 if i < extra else base_size for i in range(partition_id))
        part_size = base_size + 1 if partition_id < extra else base_size
        end_idx = start_idx + part_size
        label_partition = indices[start_idx:end_idx]

        if shuffle:
            random.shuffle(label_partition)

        partition_indices.extend(label_partition)

    print(f"Balanced partition {partition_id} loaded with {len(partition_indices)} samples")
    return Subset(dataset, partition_indices), partition_indices


def configure_balanced_partition(
    root: str,
    dataset_name: str,
    partition_id: int,
    num_partitions: int,
    seed: int,
    shuffle: bool
) -> Tuple[Subset, list, Subset, list]:
    """
    Load a dataset and partition it with balanced class distribution
    
    Returns:
        training_set, full_training_index, test_set, test_index
    """
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name.lower() == "fashionmnist":
        dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError("Unsupported dataset")

    if not (0 <= partition_id < num_partitions):
        raise ValueError(f"partition_id must be between 0 and {num_partitions - 1}")

    training_set, full_training_index = _partition_dataset(dataset, num_partitions, partition_id, shuffle)
    test_set, test_index = _partition_dataset(test_dataset, num_partitions, partition_id, shuffle)
    
    return training_set, full_training_index, test_set, test_index


def load_datasets_with_forgetting(
    partition_id: int,
    num_partitions: int,
    seed: int = 42,
    shuffle: bool = True,
    forgetting_config: Dict = None,
    dataset_name: str = "cifar10",
    config: Dict = None
) -> Tuple[Optional[DataLoader], Optional[DataLoader], DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Load and partition datasets with forgetting functionality
    
    Args:
        partition_id: Client ID
        num_partitions: Total number of clients
        seed: Random seed
        shuffle: Whether to shuffle data
        forgetting_config: Dict mapping class labels to forgetting ratios
        dataset_name: Name of dataset (cifar10, mnist, fashionmnist)
        config: Configuration dictionary with batch sizes and unlearning settings
        
    Returns:
        retrainloader, forgetloader, valloader, testloader, original_forget_loader
    """
    if forgetting_config is None:
        forgetting_config = {}
    if config is None:
        config = {}
    
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Load partitioned data
    partition, full_training_index, test_set, test_index = configure_balanced_partition(
        root="./data",
        dataset_name=dataset_name,
        partition_id=partition_id,
        num_partitions=num_partitions,
        seed=seed,
        shuffle=shuffle
    )

    # Group data by class labels
    label_to_indices = defaultdict(list)
    for idx, item in enumerate(partition):
        label_to_indices[item[1]].append(idx)

    # Split indices: 90% train, 10% val
    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        total_size = len(indices)
        train_size = int(0.9 * total_size)
        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:])

    train_data = Subset(partition, train_indices)
    val_data = Subset(partition, val_indices)
    test_data = test_set

    # Split train set into retrain and forget sets
    class_indices = defaultdict(list)
    for i, x in enumerate(train_data):
        class_indices[x[1]].append(i)

    forget_indices = []
    retrain_indices = []

    for cls, indices in class_indices.items():
        if cls in forgetting_config:
            random.shuffle(indices)
            forget_count = int(len(indices) * forgetting_config[cls])
            forget_indices.extend(indices[:forget_count])
            retrain_indices.extend(indices[forget_count:])
        else:
            retrain_indices.extend(indices)

    forgetset = Subset(train_data, forget_indices) if forget_indices else None
    retrainset = Subset(train_data, retrain_indices) if retrain_indices else None

    # Keep original forget set
    original_forget_dataset = copy.deepcopy(forgetset) if forgetset else None
    
    # Apply unlearning transformations if specified
    forget_clients = config.get("CLIENT_ID_TO_FORGET", [])
    if partition_id in forget_clients and forgetset is not None:
        unlearning_case = config.get("UNLEARNING_CASE", "NORMAL")
        if unlearning_case == "CONFUSE":
            map_confuse = config.get("MAP_CONFUSE", {})
            forgetset = confuse_the_forget_set(forgetset, map_confuse)
            print(f"Client {partition_id}: Applied CONFUSE transformation")
        elif unlearning_case == "BACKDOOR":
            forgetset = backdoor_the_forget_set(forgetset)
            print(f"Client {partition_id}: Applied BACKDOOR transformation")

    # Create DataLoaders
    retrain_batch = config.get("RETRAIN_BATCH", 64)
    forget_batch = config.get("FORGET_BATCH", 32)
    val_batch = config.get("VAL_BATCH", 128)
    test_batch = config.get("TEST_BATCH", 128)

    retrainloader = DataLoader(retrainset, batch_size=retrain_batch, shuffle=True) if retrainset and len(retrainset) > 0 else None
    forgetloader = DataLoader(forgetset, batch_size=forget_batch, shuffle=True) if forgetset and len(forgetset) > 0 else None
    original_forget_loader = DataLoader(original_forget_dataset, batch_size=forget_batch, shuffle=True) if original_forget_dataset and len(original_forget_dataset) > 0 else None
    valloader = DataLoader(val_data, batch_size=val_batch, shuffle=True)
    testloader = DataLoader(test_data, batch_size=test_batch, shuffle=True)

    return retrainloader, forgetloader, valloader, testloader, original_forget_loader


class MembershipDataset(Dataset):
    """Dataset wrapper that adds membership labels"""
    def __init__(self, dataset, is_member):
        self.dataset = dataset
        self.is_member = is_member

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        return X, y, self.is_member


def create_attack_and_shadow_loaders(forgetloader, testloader, valloader, batch_size=64):
    """
    Create attack_loader and shadow_loader for membership inference attacks
    """
    forget_dataset_with_membership = MembershipDataset(forgetloader.dataset, is_member=1)
    test_dataset_with_membership = MembershipDataset(testloader.dataset, is_member=0)
    combined_dataset = ConcatDataset([forget_dataset_with_membership, test_dataset_with_membership])
    
    attack_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    shadow_loader = DataLoader(valloader.dataset, batch_size=batch_size, shuffle=True)
    
    return attack_loader, shadow_loader
