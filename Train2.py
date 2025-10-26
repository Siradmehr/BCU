"""
Federated Client Unlearning (FCU) Training Script
Implements Model-Contrastive Unlearning (MCU) + Frequency-Guided Memory Preservation (FGMP)
Supports CIFAR-10/100/MNIST/FashionMNIST with backdoor and confusion attacks
Uses NFResNet (Normalizer-Free ResNet) and other custom models
Logs comprehensive accuracies: test, train, forget, validation sets
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import numpy as np
from collections import defaultdict
import os
import warnings
import csv
from datetime import datetime

# Suppress multiprocessing warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       message="resource_tracker: There appear to be .* leaked semaphore objects")

from src.datasets.fcu_adapter import FCUDataLoader
from src.datasets.cifar_dataloader import save_all_partitions_summary
from src.utils.device_utils import get_device
from models import (
    nf_resnet18, nf_resnet34, nf_resnet50,
    LeNet5, FLNet, CNNCifar
)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(model_name: str, num_classes: int = 10, input_channels: int = 3):
    """Get model based on name"""
    if model_name == "nf_resnet18":
        model = nf_resnet18(num_classes=num_classes)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model
    
    elif model_name == "nf_resnet34":
        model = nf_resnet34(num_classes=num_classes)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model
    
    elif model_name == "nf_resnet50":
        model = nf_resnet50(num_classes=num_classes)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model
    
    elif model_name == "LeNet5":
        return LeNet5()
    
    elif model_name == "FLNet":
        return FLNet(num_class=num_classes)
    
    elif model_name == "CNNCifar":
        class Args:
            num_classes = num_classes
        return CNNCifar(Args())
    
    elif model_name == "resnet18":
        import torchvision.models as models
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


class ModelWrapper(nn.Module):
    """Wrapper to provide consistent interface for all models"""
    def __init__(self, base_model, model_name):
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name
        
    def forward(self, x):
        return self.base_model(x)
    
    def get_features(self, x):
        """Extract features before final classification layer"""
        if self.model_name in ["nf_resnet18", "nf_resnet34", "nf_resnet50"]:
            x = self.base_model.conv1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        
        elif self.model_name == "resnet18":
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        
        elif self.model_name == "LeNet5":
            x = self.base_model.c1(x)
            x = self.base_model.c2_1(x)
            y = self.base_model.c2_2(self.base_model.c1.c1[2](self.base_model.c1.c1[1](self.base_model.c1.c1[0](x))))
            x = x + y
            x = self.base_model.c3(x)
            x = x.view(x.size(0), -1)
            x = self.base_model.f4.f4[0](x)
            return x
        
        elif self.model_name == "FLNet":
            x = F.max_pool2d(F.relu(self.base_model.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.base_model.conv2(x)), 2)
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            x = F.relu(self.base_model.fc1(x))
            return x
        
        elif self.model_name == "CNNCifar":
            x = self.base_model.pool(F.relu(self.base_model.conv1(x)))
            x = self.base_model.pool(F.relu(self.base_model.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.base_model.fc1(x))
            x = F.relu(self.base_model.fc2(x))
            return x
        
        else:
            raise NotImplementedError(f"get_features not implemented for {self.model_name}")


def evaluate_on_loader(model, loader, device):
    """
    Evaluate model on a single dataloader
    
    Returns:
        accuracy, total_samples
    """
    eval_model = model
    if isinstance(model, nn.DataParallel):
        eval_model = model.module
    if isinstance(eval_model, ModelWrapper):
        eval_model = eval_model.base_model
    
    eval_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = eval_model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total


def evaluate_model_comprehensive(model, dataloaders, device, config=None):
    """
    Comprehensive evaluation on test, train, forget, and validation sets
    
    Returns:
        dict with all accuracies
    """
    eval_model = model
    if isinstance(model, nn.DataParallel):
        eval_model = model.module
    if isinstance(eval_model, ModelWrapper):
        eval_model = eval_model.base_model
    
    eval_model.eval()
    
    # Test set accuracy
    test_correct = 0
    test_total = 0
    
    # Train set accuracy
    train_correct = 0
    train_total = 0
    
    # Forget set accuracy
    forget_correct = 0
    forget_total = 0
    
    # Validation set accuracy
    val_correct = 0
    val_total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for dataloader in dataloaders:
            # Test set
            test_loader = dataloader.get_test_loader()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = eval_model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
                
                # Class-wise accuracy on test set
                for t, p in zip(target, pred):
                    t_item = t.item()
                    class_total[t_item] += 1
                    if p.item() == t_item:
                        class_correct[t_item] += 1
            
            # Train set (retrain loader)
            train_loader = dataloader.get_train_loader()
            if train_loader:
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    output = eval_model(data)
                    pred = output.argmax(dim=1)
                    train_correct += pred.eq(target).sum().item()
                    train_total += target.size(0)
            
            # Forget set
            forget_loader = dataloader.get_forget_loader()
            if forget_loader:
                for data, target in forget_loader:
                    data, target = data.to(device), target.to(device)
                    output = eval_model(data)
                    pred = output.argmax(dim=1)
                    forget_correct += pred.eq(target).sum().item()
                    forget_total += target.size(0)
            
            # Validation set
            val_loader = dataloader.get_val_loader()
            if val_loader:
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = eval_model(data)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
    
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    forget_acc = forget_correct / forget_total if forget_total > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    
    class_acc = {cls: class_correct[cls] / class_total[cls] 
                 for cls in class_total.keys()}
    
    return {
        'test_acc': test_acc,
        'test_samples': test_total,
        'train_acc': train_acc,
        'train_samples': train_total,
        'forget_acc': forget_acc,
        'forget_samples': forget_total,
        'val_acc': val_acc,
        'val_samples': val_total,
        'class_acc': class_acc,
    }


def save_accuracy_to_csv(results, phase, config, args):
    """
    Save comprehensive accuracy results to CSV file
    
    Args:
        results: Dictionary with all evaluation results
        phase: 'training' or 'unlearning'
        config: Configuration dictionary
        args: Command line arguments
    """
    results_dir = Path(config.get('results_dir', './results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = config.get('dataset', 'unknown')
    model_name = config.get('model_name', 'unknown')
    attack_type = config.get('UNLEARNING_CASE', 'NORMAL')
    
    filename = f"{phase}_results_{dataset}_{model_name}_{attack_type}_{timestamp}.csv"
    filepath = results_dir / filename
    
    # Prepare data to write
    data = {
        'timestamp': timestamp,
        'phase': phase,
        'dataset': dataset,
        'model': model_name,
        'attack_type': attack_type,
        'num_clients': config.get('num_clients', 'N/A'),
        'excluded_clients': str(args.excluded_clients),
        
        # Main accuracies
        'test_acc': results.get('test_acc', 'N/A'),
        'test_samples': results.get('test_samples', 'N/A'),
        'train_acc': results.get('train_acc', 'N/A'),
        'train_samples': results.get('train_samples', 'N/A'),
        'forget_acc': results.get('forget_acc', 'N/A'),
        'forget_samples': results.get('forget_samples', 'N/A'),
        'val_acc': results.get('val_acc', 'N/A'),
        'val_samples': results.get('val_samples', 'N/A'),
    }
    
    # Add backdoor-specific metrics
    if 'backdoor_asr' in results:
        data['backdoor_asr'] = results['backdoor_asr']
        data['backdoor_samples'] = results['backdoor_samples']
        data['trigger_size'] = results.get('trigger_size', 'N/A')
    
    # Add class-wise accuracies if available
    if 'class_acc' in results:
        for cls, acc in results['class_acc'].items():
            data[f'class_{cls}_acc'] = acc
    
    # Write to CSV
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED TO: {filepath}")
    print(f"{'='*60}")
    print(f"Phase: {phase.upper()}")
    print(f"Test Accuracy:       {results.get('test_acc', 'N/A'):.4f} ({results.get('test_samples', 0)} samples)")
    print(f"Train Accuracy:      {results.get('train_acc', 'N/A'):.4f} ({results.get('train_samples', 0)} samples)")
    print(f"Forget Accuracy:     {results.get('forget_acc', 'N/A'):.4f} ({results.get('forget_samples', 0)} samples)")
    print(f"Validation Accuracy: {results.get('val_acc', 'N/A'):.4f} ({results.get('val_samples', 0)} samples)")
    if 'backdoor_asr' in results:
        print(f"Backdoor ASR:        {results['backdoor_asr']:.4f}")
    print(f"{'='*60}\n")
    
    return filepath


def save_checkpoint(model, optimizer, epoch, dataloaders, path, config=None):
    """Save comprehensive checkpoint including partition info"""
    partition_info_all = {}
    for dataloader in dataloaders:
        partition_info = dataloader.get_partition_info()
        partition_info_all[partition_info['partition_id']] = partition_info
    
    if isinstance(model, ModelWrapper):
        model_state = model.base_model.state_dict()
    elif isinstance(model, nn.DataParallel):
        if isinstance(model.module, ModelWrapper):
            model_state = model.module.base_model.state_dict()
        else:
            model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config,
        'partition_info': partition_info_all,
        'num_clients': len(dataloaders)
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to {path}")


def load_checkpoint_with_partitions(checkpoint_path, device, config):
    """Load checkpoint and return model + partition info"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    saved_config = checkpoint.get('config', config)
    num_classes = saved_config.get('num_classes', config.get('num_classes', 10))
    model_name = saved_config.get('model_name', config.get('model_name', 'nf_resnet18'))
    
    dataset = saved_config.get('dataset', config.get('dataset', 'cifar10'))
    input_channels = 3 if dataset in ['cifar10', 'cifar100'] else 1
    
    base_model = get_model(model_name, num_classes, input_channels)
    model = ModelWrapper(base_model, model_name).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.base_model.load_state_dict(checkpoint)
    
    partition_info_all = checkpoint.get('partition_info', None)
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Model: {model_name}")
    
    return model, saved_config, partition_info_all


def verify_partition_consistency(dataloaders, partition_info_all):
    """Verify that current dataloaders match saved partition info"""
    if partition_info_all is None:
        print("Warning: No saved partition info to verify against")
        return True
    
    print("\nVerifying partition consistency...")
    all_consistent = True
    
    for dataloader in dataloaders:
        current_info = dataloader.get_partition_info()
        client_id = current_info['partition_id']
        
        if client_id not in partition_info_all:
            print(f"  Client {client_id}: No saved partition info found")
            all_consistent = False
            continue
        
        saved_info = partition_info_all[client_id]
        
        checks = [
            ('full_training_index', current_info['full_training_index'], saved_info['full_training_index']),
            ('retrain_indices', current_info['retrain_indices'], saved_info['retrain_indices']),
            ('forget_indices', current_info['forget_indices'], saved_info['forget_indices']),
        ]
        
        client_consistent = True
        for name, current, saved in checks:
            if current != saved:
                print(f"  Client {client_id}: {name} MISMATCH")
                client_consistent = False
                all_consistent = False
        
        if client_consistent:
            print(f"  Client {client_id}: ✓ Consistent")
    
    return all_consistent


def get_client_forgetting_config(client_id, config, excluded_clients=None):
    """Get forgetting configuration for a specific client"""
    if 'CLIENT_POISONING_CONFIG' in config:
        client_poison_config = config['CLIENT_POISONING_CONFIG']
        if client_id in client_poison_config:
            return client_poison_config[client_id].get('classes', {})
        else:
            return {}
    
    elif 'CLIENTS_TO_POISON' in config:
        clients_to_poison = config['CLIENTS_TO_POISON']
        if client_id in clients_to_poison:
            return config.get('forgetting_config', {})
        else:
            return {}
    
    elif config.get('ATTACK_ALL_CLIENTS', False):
        return config.get('forgetting_config', {})
    
    else:
        if excluded_clients and client_id in excluded_clients:
            return config.get('forgetting_config', {})
        else:
            return {}


def print_unlearning_scenario(config, excluded_clients):
    """Print clear explanation of unlearning scenario"""
    mode = config.get('UNLEARNING_MODE', 'DATA_LEVEL')
    
    print("\n" + "="*60)
    print("UNLEARNING SCENARIO")
    print("="*60)
    
    if mode == "CLIENT_LEVEL":
        print("Mode: CLIENT-LEVEL UNLEARNING")
        print("Description: Remove ENTIRE client(s) with ALL their data")
        print(f"Target clients: {excluded_clients}")
    elif mode == "DATA_LEVEL":
        print("Mode: DATA-LEVEL UNLEARNING")
        print("Description: Remove SPECIFIC poisoned/attacked data only")
        print(f"Target clients: {excluded_clients} (have poisoned data)")
    
    if config.get('UNLEARNING_CASE') == 'BACKDOOR':
        trigger_size = config.get('BACKDOOR_TRIGGER_SIZE', 3)
        print(f"\nBackdoor Trigger: {trigger_size}×{trigger_size} white pixels at bottom-right")
        print(f"Target Label: {config.get('BACKDOOR_TARGET_LABEL', 0)}")
    
    print("="*60)


def print_poisoning_summary(dataloaders, config):
    """Print summary of which clients are poisoned"""
    print("\n" + "="*60)
    print("CLIENT POISONING SUMMARY")
    print("="*60)
    
    poisoned_clients = []
    clean_clients = []
    
    for dataloader in dataloaders:
        client_id = dataloader.partition_id
        forget_size = len(dataloader.forgetloader.dataset) if dataloader.forgetloader else 0
        train_size = len(dataloader.retrainloader.dataset) if dataloader.retrainloader else 0
        
        if forget_size > 0:
            poisoned_clients.append(client_id)
            poison_ratio = forget_size / (forget_size + train_size) if (forget_size + train_size) > 0 else 0
            print(f"Client {client_id:2d}: ⚠ POISONED - {forget_size:4d} samples ({poison_ratio*100:.1f}%)")
        else:
            clean_clients.append(client_id)
            print(f"Client {client_id:2d}: ✓ CLEAN")
    
    print("-" * 60)
    print(f"Poisoned: {len(poisoned_clients)}/{len(dataloaders)} - {poisoned_clients}")
    print(f"Clean: {len(clean_clients)}/{len(dataloaders)} - {clean_clients}")
    print("="*60)


def print_client_summary(dataloaders, excluded_clients, config):
    """Print summary of client data distribution"""
    print("\n" + "="*60)
    print("CLIENT DATA SUMMARY")
    print("="*60)
    
    total_train = 0
    total_forget = 0
    
    for dataloader in dataloaders:
        client_id = dataloader.partition_id
        train_size = len(dataloader.retrainloader.dataset) if dataloader.retrainloader else 0
        forget_size = len(dataloader.forgetloader.dataset) if dataloader.forgetloader else 0
        
        total_train += train_size
        total_forget += forget_size
        
        status = "❌ TO UNLEARN" if client_id in excluded_clients else "✓ Keep"
        print(f"Client {client_id:2d} {status:12s}: Train={train_size:5d}, Forget={forget_size:4d}")
    
    print("-" * 60)
    print(f"{'TOTAL':15s}: Train={total_train:5d}, Forget={total_forget:4d}")
    print("="*60)


def get_unlearning_dataloaders(all_dataloaders, excluded_clients, config):
    """Get appropriate dataloaders based on unlearning mode"""
    mode = config.get('UNLEARNING_MODE', 'DATA_LEVEL')
    
    if mode == "CLIENT_LEVEL":
        forget_dataloaders = [all_dataloaders[i] for i in excluded_clients]
        retain_dataloaders = [dl for i, dl in enumerate(all_dataloaders) 
                             if i not in excluded_clients]
        print(f"\nCLIENT-LEVEL: Forget {len(forget_dataloaders)} clients, Retain {len(retain_dataloaders)}")
        
    elif mode == "DATA_LEVEL":
        forget_dataloaders = [all_dataloaders[i] for i in excluded_clients]
        retain_dataloaders = all_dataloaders
        print(f"\nDATA-LEVEL: Forget poisoned data, Retain all clean data")
    
    return forget_dataloaders, retain_dataloaders


def add_backdoor_trigger(data, config):
    """Add 3×3 white pixel backdoor trigger"""
    trigger_size = config.get('BACKDOOR_TRIGGER_SIZE', 3)
    trigger_value = config.get('BACKDOOR_TRIGGER_VALUE', 1.0)
    
    backdoored = data.clone()
    
    if backdoored.dim() == 4:
        B, C, H, W = backdoored.shape
        backdoored[:, :, H-trigger_size:H, W-trigger_size:W] = trigger_value
    elif backdoored.dim() == 3:
        C, H, W = backdoored.shape
        backdoored[:, H-trigger_size:H, W-trigger_size:W] = trigger_value
    elif backdoored.dim() == 2:
        H, W = backdoored.shape
        backdoored[H-trigger_size:H, W-trigger_size:W] = trigger_value
    
    return backdoored


def evaluate_backdoor_attack(model, dataloaders, device, config):
    """Evaluate Backdoor Attack Success Rate with 3×3 trigger"""
    eval_model = model
    if isinstance(model, nn.DataParallel):
        eval_model = model.module
    if isinstance(eval_model, ModelWrapper):
        eval_model = eval_model.base_model
    
    eval_model.eval()
    
    target_label = config.get('BACKDOOR_TARGET_LABEL', 0)
    backdoor_correct = 0
    total = 0
    
    with torch.no_grad():
        for dataloader in dataloaders:
            test_loader = dataloader.get_test_loader()
            for data, target in test_loader:
                backdoored_data = add_backdoor_trigger(data, config)
                backdoored_data = backdoored_data.to(device)
                
                output = eval_model(backdoored_data)
                pred = output.argmax(dim=1)
                
                backdoor_correct += (pred == target_label).sum().item()
                total += target.size(0)
    
    basr = backdoor_correct / total if total > 0 else 0.0
    trigger_size = config.get('BACKDOOR_TRIGGER_SIZE', 3)
    
    return {
        'backdoor_asr': basr,
        'backdoor_samples': backdoor_correct,
        'total_samples': total,
        'trigger_size': f'{trigger_size}×{trigger_size}'
    }


def train_federated(config, dataloaders, device, args):
    """Federated learning training phase"""
    print(f"\nStarting FL training on {device}")
    print(f"Attack type: {config.get('UNLEARNING_CASE', 'NORMAL')}")
    print(f"Model: {config.get('model_name', 'nf_resnet18')}")
    
    if config.get('UNLEARNING_CASE') == 'BACKDOOR':
        trigger_size = config.get('BACKDOOR_TRIGGER_SIZE', 3)
        print(f"Backdoor trigger: {trigger_size}×{trigger_size} white pixels")
    
    model_name = config.get('model_name', 'nf_resnet18')
    num_classes = config['num_classes']
    input_channels = 3 if config['dataset'] in ['cifar10', 'cifar100'] else 1
    
    base_model = get_model(model_name, num_classes, input_channels)
    model = ModelWrapper(base_model, model_name).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    for round_idx in range(config['num_rounds']):
        client_weights = []
        client_sizes = []
        
        for client_id, dataloader in enumerate(dataloaders):
            train_loader = dataloader.get_combined_train_loader()
            if train_loader is None:
                continue
            
            local_base_model = get_model(model_name, num_classes, input_channels)
            local_model = ModelWrapper(local_base_model, model_name).to(device)
            
            if isinstance(model, nn.DataParallel):
                local_model.base_model.load_state_dict(model.module.base_model.state_dict())
            else:
                local_model.base_model.load_state_dict(model.base_model.state_dict())
            
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
            criterion = nn.CrossEntropyLoss()
            
            local_model.train()
            for epoch in range(config['local_epochs']):
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            client_weights.append({k: v.cpu().clone() for k, v in local_model.base_model.state_dict().items()})
            client_sizes.append(len(train_loader.dataset))
        
        total_size = sum(client_sizes)
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = sum(
                w[key] * (size / total_size)
                for w, size in zip(client_weights, client_sizes)
            )
        
        if isinstance(model, nn.DataParallel):
            model.module.base_model.load_state_dict(aggregated_weights)
        else:
            model.base_model.load_state_dict(aggregated_weights)
        
        if round_idx % config['eval_interval'] == 0:
            results = evaluate_model_comprehensive(model, dataloaders, device, config)
            print(f"\nRound {round_idx}:")
            print(f"  Test Acc: {results['test_acc']:.4f} | Train Acc: {results['train_acc']:.4f}")
            
            if config.get('UNLEARNING_CASE') == 'BACKDOOR':
                backdoor_results = evaluate_backdoor_attack(model, dataloaders, device, config)
                print(f"  Backdoor ASR: {backdoor_results['backdoor_asr']:.4f}")
    
    # Final comprehensive evaluation
    print("\n" + "="*60)
    print("FINAL TRAINING PHASE EVALUATION")
    print("="*60)
    
    final_results = evaluate_model_comprehensive(model, dataloaders, device, config)
    
    print(f"Test Accuracy:       {final_results['test_acc']:.4f} ({final_results['test_samples']} samples)")
    print(f"Train Accuracy:      {final_results['train_acc']:.4f} ({final_results['train_samples']} samples)")
    print(f"Forget Accuracy:     {final_results['forget_acc']:.4f} ({final_results['forget_samples']} samples)")
    print(f"Validation Accuracy: {final_results['val_acc']:.4f} ({final_results['val_samples']} samples)")
    
    if config.get('UNLEARNING_CASE') == 'BACKDOOR':
        backdoor_results = evaluate_backdoor_attack(model, dataloaders, device, config)
        print(f"Backdoor ASR:        {backdoor_results['backdoor_asr']:.4f}")
        final_results.update(backdoor_results)
    
    # Save to CSV
    save_accuracy_to_csv(final_results, 'training', config, args)
    
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def apply_fgmp(model, trained_model, device):
    """Frequency-Guided Memory Preservation (FGMP)"""
    with torch.no_grad():
        for (name, param_current), (_, param_trained) in zip(
            model.named_parameters(),
            trained_model.named_parameters()
        ):
            if 'bn' in name or 'bias' in name or len(param_current.shape) < 2:
                continue
            
            try:
                original_shape = param_current.shape
                
                if len(original_shape) == 4:
                    C_out, C_in, H, W = original_shape
                    param_current_2d = param_current.reshape(C_out * C_in, H * W)
                    param_trained_2d = param_trained.reshape(C_out * C_in, H * W)
                elif len(original_shape) == 2:
                    param_current_2d = param_current
                    param_trained_2d = param_trained
                else:
                    continue
                
                h, w = param_current_2d.shape
                pad_h = (2 ** int(np.ceil(np.log2(h)))) - h
                pad_w = (2 ** int(np.ceil(np.log2(w)))) - w
                
                if pad_h > 0 or pad_w > 0:
                    param_current_2d = F.pad(param_current_2d, (0, pad_w, 0, pad_h))
                    param_trained_2d = F.pad(param_trained_2d, (0, pad_w, 0, pad_h))
                
                fft_current = torch.fft.fft2(param_current_2d)
                fft_trained = torch.fft.fft2(param_trained_2d)
                
                amp_current = torch.abs(fft_current)
                amp_trained = torch.abs(fft_trained)
                phase_current = torch.angle(fft_current)
                
                H_fft, W_fft = fft_current.shape
                center_h, center_w = H_fft // 2, W_fft // 2
                radius = min(center_h, center_w) // 3
                
                Y, X = torch.meshgrid(
                    torch.arange(H_fft, device=device),
                    torch.arange(W_fft, device=device),
                    indexing='ij'
                )
                distance = torch.sqrt((Y - center_h) ** 2 + (X - center_w) ** 2)
                mask_low = (distance <= radius).float()
                mask_high = 1.0 - mask_low
                
                amp_combined = mask_low * amp_trained + mask_high * amp_current
                fft_combined = amp_combined * torch.exp(1j * phase_current)
                param_new = torch.fft.ifft2(fft_combined).real
                
                if pad_h > 0 or pad_w > 0:
                    param_new = param_new[:h, :w]
                
                if len(original_shape) == 4:
                    param_new = param_new.reshape(original_shape)
                
                if torch.isnan(param_new).any() or torch.isinf(param_new).any():
                    continue
                
                param_current.copy_(param_new)
                
            except Exception as e:
                continue


def unlearn_with_fcu(model, forget_dataloaders, retain_dataloaders, config, device, args, all_dataloaders):
    """Perform FCU unlearning - MCU + FGMP"""
    mode = config.get('UNLEARNING_MODE', 'DATA_LEVEL')
    
    print(f"\n{'='*60}")
    print(f"FCU UNLEARNING - {mode} MODE")
    print(f"{'='*60}")
    
    if mode == "CLIENT_LEVEL":
        total_forget_samples = sum(
            len(dl.get_combined_train_loader().dataset) if dl.get_combined_train_loader() else 0
            for dl in forget_dataloaders
        )
    else:
        total_forget_samples = sum(
            len(dl.get_forget_loader().dataset) if dl.get_forget_loader() else 0
            for dl in forget_dataloaders
        )
    
    total_retain_samples = sum(
        len(dl.get_train_loader().dataset) if dl.get_train_loader() else 0
        for dl in retain_dataloaders
    )
    
    print(f"Samples to forget: {total_forget_samples}")
    print(f"Samples to retain: {total_retain_samples}")
    
    model_name = config.get('model_name', 'nf_resnet18')
    num_classes = config['num_classes']
    input_channels = 3 if config['dataset'] in ['cifar10', 'cifar100'] else 1
    
    reference_base_model = get_model(model_name, num_classes, input_channels)
    reference_model = ModelWrapper(reference_base_model, model_name).to(device)
    reference_model.eval()
    print("✓ Reference model (random init) for MCU")
    
    trained_base_model = get_model(model_name, num_classes, input_channels)
    trained_model = ModelWrapper(trained_base_model, model_name).to(device)
    trained_model.base_model.load_state_dict(model.base_model.state_dict())
    trained_model.eval()
    print("✓ Trained model saved for FGMP")
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.get('unlearn_lr', 0.01),
        momentum=0.9,
        weight_decay=5e-4
    )
    
    unlearn_epochs = config.get('unlearn_epochs', 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=unlearn_epochs,
        eta_min=config.get('unlearn_lr', 0.01) * 0.01
    )
    
    lambda_mcu = config.get('lambda_mcu', 1.0)
    lambda_retain = config.get('lambda_retain', 1.0)
    T_fgmp = config.get('T_fgmp', 10)
    T_mcu = config.get('T_mcu', 1.0)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    print(f"\nHyperparameters:")
    print(f"  Epochs: {unlearn_epochs}, LR: {config.get('unlearn_lr', 0.01)}")
    print(f"  Lambda MCU: {lambda_mcu}, Lambda Retain: {lambda_retain}")
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    iteration = 0
    for epoch in range(unlearn_epochs):
        epoch_mcu_loss = 0.0
        epoch_retain_loss = 0.0
        num_batches = 0
        
        forget_loaders = []
        if mode == "CLIENT_LEVEL":
            for dl in forget_dataloaders:
                if dl.get_combined_train_loader() is not None:
                    forget_loaders.append(dl.get_combined_train_loader())
        else:
            for dl in forget_dataloaders:
                if dl.get_forget_loader() is not None:
                    forget_loaders.append(dl.get_forget_loader())
        
        retrain_loaders = []
        for dl in retain_dataloaders:
            if dl.get_train_loader() is not None:
                retrain_loaders.append(dl.get_train_loader())
        
        forget_iters = [iter(loader) for loader in forget_loaders] if forget_loaders else []
        retrain_iters = [iter(loader) for loader in retrain_loaders] if retrain_loaders else []
        
        max_forget_batches = max([len(loader) for loader in forget_loaders]) if forget_loaders else 0
        max_retrain_batches = max([len(loader) for loader in retrain_loaders]) if retrain_loaders else 0
        num_iterations = max(max_forget_batches, max_retrain_batches)
        
        for batch_idx in range(num_iterations):
            optimizer.zero_grad()
            total_loss = 0.0
            
            mcu_loss = 0.0
            mcu_count = 0
            
            if forget_iters:
                for forget_iter in forget_iters:
                    try:
                        data, target = next(forget_iter)
                    except StopIteration:
                        continue
                    
                    data = data.to(device)
                    features_current = model.get_features(data) / T_mcu
                    
                    with torch.no_grad():
                        features_ref = reference_model.get_features(data) / T_mcu
                    
                    mcu_loss += F.mse_loss(features_current, features_ref)
                    mcu_count += 1
                
                if mcu_count > 0:
                    mcu_loss = mcu_loss / mcu_count
                    total_loss += lambda_mcu * mcu_loss
                    epoch_mcu_loss += mcu_loss.item()
            
            retain_loss = 0.0
            retain_count = 0
            
            if retrain_iters:
                for retrain_iter in retrain_iters:
                    try:
                        data, target = next(retrain_iter)
                    except StopIteration:
                        continue
                    
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    retain_loss += criterion(output, target)
                    retain_count += 1
                
                if retain_count > 0:
                    retain_loss = retain_loss / retain_count
                    total_loss += lambda_retain * retain_loss
                    epoch_retain_loss += retain_loss.item()
            
            if total_loss > 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                num_batches += 1
            
            iteration += 1
            
            if iteration % T_fgmp == 0 and iteration > 0:
                apply_fgmp(model.base_model, trained_model.base_model, device)
        
        scheduler.step()
        
        if epoch % 5 == 0:
            avg_mcu = epoch_mcu_loss / num_batches if num_batches > 0 else 0.0
            avg_retain = epoch_retain_loss / num_batches if num_batches > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}: LR={current_lr:.6f}, MCU={avg_mcu:.4f}, Retain={avg_retain:.4f}")
    
    print(f"\n✓ Unlearning completed")
    
    # Final comprehensive evaluation
    print("\n" + "="*60)
    print("FINAL UNLEARNING PHASE EVALUATION")
    print("="*60)
    
    final_results = evaluate_model_comprehensive(model, all_dataloaders, device, config)
    
    print(f"Test Accuracy:       {final_results['test_acc']:.4f} ({final_results['test_samples']} samples)")
    print(f"Train Accuracy:      {final_results['train_acc']:.4f} ({final_results['train_samples']} samples)")
    print(f"Forget Accuracy:     {final_results['forget_acc']:.4f} ({final_results['forget_samples']} samples)")
    print(f"Validation Accuracy: {final_results['val_acc']:.4f} ({final_results['val_samples']} samples)")
    
    if config.get('UNLEARNING_CASE') == 'BACKDOOR':
        backdoor_results = evaluate_backdoor_attack(model, all_dataloaders, device, config)
        print(f"Backdoor ASR:        {backdoor_results['backdoor_asr']:.4f}")
        final_results.update(backdoor_results)
    
    # Save to CSV
    save_accuracy_to_csv(final_results, 'unlearning', config, args)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='FCU with NFResNet and 3×3 Backdoor')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--excluded_clients', type=int, nargs='+', default=[0])
    parser.add_argument('--is_unlearn', type=int, default=0)
    parser.add_argument('--attack_type', type=str, choices=['NORMAL', 'CONFUSE', 'BACKDOOR'])
    parser.add_argument('--attack_all', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--partition_dir', type=str, default='./partition_info')
    parser.add_argument('--force_new_partition', action='store_true')
    parser.add_argument('--verify_partitions', action='store_true')
    parser.add_argument('--unlearn_mode', type=str, choices=['CLIENT_LEVEL', 'DATA_LEVEL'])
    parser.add_argument('--poison_clients', type=int, nargs='+', default=None)
    parser.add_argument('--poison_ratio', type=float, default=0.3)
    parser.add_argument('--poison_classes', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--trigger_size', type=int, default=3)
    parser.add_argument('--trigger_value', type=float, default=1.0)
    parser.add_argument('--model_name', type=str, default='nf_resnet18',
                       choices=['nf_resnet18', 'nf_resnet34', 'nf_resnet50', 
                               'resnet18', 'LeNet5', 'FLNet', 'CNNCifar'])
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.attack_type:
        config['UNLEARNING_CASE'] = args.attack_type
    if args.attack_all:
        config['ATTACK_ALL_CLIENTS'] = True
    if args.unlearn_mode:
        config['UNLEARNING_MODE'] = args.unlearn_mode
    
    config['BACKDOOR_TRIGGER_SIZE'] = args.trigger_size
    config['BACKDOOR_TRIGGER_VALUE'] = args.trigger_value
    config['model_name'] = args.model_name
    
    if args.poison_clients:
        config['CLIENT_POISONING_CONFIG'] = {}
        for client_id in args.poison_clients:
            config['CLIENT_POISONING_CONFIG'][client_id] = {
                'classes': {cls: args.poison_ratio for cls in args.poison_classes}
            }
    
    set_seed(config['SEED'])
    device = get_device(config.get('device', 'auto'))
    
    print("\n" + "="*60)
    print("EXPERIMENT SETUP")
    print("="*60)
    print(f"Dataset: {config['dataset']}")
    print(f"Model: {args.model_name}")
    print(f"Num Clients: {config['num_clients']}")
    print(f"Attack Type: {config.get('UNLEARNING_CASE', 'NORMAL')}")
    if config.get('UNLEARNING_CASE') == 'BACKDOOR':
        print(f"Backdoor Trigger: {args.trigger_size}×{args.trigger_size}")
    print("="*60)
    
    if args.is_unlearn:
        print_unlearning_scenario(config, args.excluded_clients)
    
    dataloaders = []
    for client_id in range(config['num_clients']):
        client_forgetting_config = get_client_forgetting_config(
            client_id, config, args.excluded_clients
        )
        
        loader = FCUDataLoader(
            partition_id=client_id,
            num_partitions=config['num_clients'],
            dataset_name=config['dataset'],
            seed=config['SEED'],
            forgetting_config=client_forgetting_config,
            config=config,
            save_partition=True,
            partition_save_dir=args.partition_dir,
            load_if_exists=not args.force_new_partition
        )
        dataloaders.append(loader)
    
    print_poisoning_summary(dataloaders, config)
    print_client_summary(dataloaders, args.excluded_clients, config)
    
    if args.load_model:
        print("\nLOADING PRE-TRAINED MODEL")
        model, saved_config, partition_info_all = load_checkpoint_with_partitions(
            args.load_model, device, config
        )
        
        if args.verify_partitions and partition_info_all:
            verify_partition_consistency(dataloaders, partition_info_all)
        
        results = evaluate_model_comprehensive(model, dataloaders, device, config)
        print(f"Loaded Model Test Accuracy: {results['test_acc']:.4f}")
        
    elif not args.skip_training:
        print("\nFEDERATED TRAINING PHASE")
        model = train_federated(config, dataloaders, device, args)
        
        save_path = Path(config.get('checkpoint_dir', './checkpoints'))
        save_path.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, None, config['num_rounds'], dataloaders,
                       save_path / 'trained_model.pt', config)
    
    if args.is_unlearn:
        print("\nUNLEARNING PHASE")
        forget_dataloaders, retain_dataloaders = get_unlearning_dataloaders(
            dataloaders, args.excluded_clients, config
        )
        
        print("\nBefore Unlearning:")
        results_before = evaluate_model_comprehensive(model, dataloaders, device, config)
        print(f"  Test Acc: {results_before['test_acc']:.4f} | Train Acc: {results_before['train_acc']:.4f}")
        print(f"  Forget Acc: {results_before['forget_acc']:.4f} | Val Acc: {results_before['val_acc']:.4f}")
        
        if config.get('UNLEARNING_CASE') == 'BACKDOOR':
            backdoor_before = evaluate_backdoor_attack(model, dataloaders, device, config)
            print(f"  Backdoor ASR: {backdoor_before['backdoor_asr']:.4f}")
        
        model = unlearn_with_fcu(model, forget_dataloaders, retain_dataloaders, config, device, args, dataloaders)
        
        print("\nAfter Unlearning:")
        results_after = evaluate_model_comprehensive(model, dataloaders, device, config)
        print(f"  Test Acc: {results_after['test_acc']:.4f} | Train Acc: {results_after['train_acc']:.4f}")
        print(f"  Forget Acc: {results_after['forget_acc']:.4f} | Val Acc: {results_after['val_acc']:.4f}")
        
        if config.get('UNLEARNING_CASE') == 'BACKDOOR':
            backdoor_after = evaluate_backdoor_attack(model, dataloaders, device, config)
            print(f"  Backdoor ASR: {backdoor_after['backdoor_asr']:.4f}")
            print(f"  ASR Change: {backdoor_after['backdoor_asr'] - backdoor_before['backdoor_asr']:+.4f}")
        
        save_path = Path(config.get('checkpoint_dir', './checkpoints'))
        save_checkpoint(model, None, config['num_rounds'], retain_dataloaders,
                       save_path / 'unlearned_model.pt', config)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
