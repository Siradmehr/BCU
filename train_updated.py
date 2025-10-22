
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
import torch.optim as optim
import numpy as np
from itertools import cycle
from copy import deepcopy

# Suppress multiprocessing warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       message="resource_tracker: There appear to be .* leaked semaphore objects")

from src.datasets.fcu_adapter import FCUDataLoader
from src.datasets.cifar_dataloader import save_all_partitions_summary
from src.utils.device_utils import get_device


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SimpleResNet18(nn.Module):
    """Simple ResNet18 for CIFAR-10/MNIST/FashionMNIST"""
    def __init__(self, num_classes=10):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False, num_classes=num_classes)
        
        # Adjust first conv for CIFAR-10 (32x32) / MNIST (28x28)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before FC layer (needed for MCU)"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def save_checkpoint(model, optimizer, epoch, dataloaders, path, config=None):
    """
    Save a comprehensive checkpoint including partition info
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch/round
        dataloaders: List of dataloaders (to extract partition info)
        path: Save path
        config: Configuration dict
    """
    # Collect partition info from all clients
    partition_info_all = {}
    for dataloader in dataloaders:
        partition_info = dataloader.get_partition_info()
        partition_info_all[partition_info['partition_id']] = partition_info
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config,
        'partition_info': partition_info_all,
        'num_clients': len(dataloaders)
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to {path}")
    print(f"  - Includes partition info for {len(partition_info_all)} clients")


def load_checkpoint_with_partitions(checkpoint_path, device):
    """
    Load checkpoint and return model + partition info
    
    Returns:
        model, config, partition_info_all
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get('config', {})
    num_classes = config.get('num_classes', 10)
    
    # Load model
    model = SimpleResNet18(num_classes=num_classes).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Extract partition info
    partition_info_all = checkpoint.get('partition_info', None)
    
    print(f"✓ Checkpoint loaded successfully")
    if partition_info_all:
        print(f"  - Loaded partition info for {len(partition_info_all)} clients")
    else:
        print(f"  - Warning: No partition info found in checkpoint")
    
    return model, config, partition_info_all


def verify_partition_consistency(dataloaders, partition_info_all):
    """
    Verify that current dataloaders match saved partition info
    
    Args:
        dataloaders: Current dataloaders
        partition_info_all: Partition info from checkpoint
        
    Returns:
        bool: True if consistent
    """
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
        
        # Compare key indices
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
    """
    Get forgetting configuration for a specific client
    Supports multiple methods: CLIENT_POISONING_CONFIG, CLIENTS_TO_POISON, ATTACK_ALL_CLIENTS
    
    Args:
        client_id: Client ID
        config: Global config dict
        excluded_clients: List of excluded client IDs (for legacy behavior)
        
    Returns:
        dict: Forgetting config for this client, or {} if client is clean
    """
    # METHOD 1: Check if using CLIENT_POISONING_CONFIG (advanced - per-client control)
    if 'CLIENT_POISONING_CONFIG' in config:
        client_poison_config = config['CLIENT_POISONING_CONFIG']
        
        if client_id in client_poison_config:
            # This client has specific poisoning config
            return client_poison_config[client_id].get('classes', {})
        else:
            # Client not in config = clean client
            return {}
    
    # METHOD 2: Check if using CLIENTS_TO_POISON (simple - same config for selected clients)
    elif 'CLIENTS_TO_POISON' in config:
        clients_to_poison = config['CLIENTS_TO_POISON']
        
        if client_id in clients_to_poison:
            # This client gets poisoned with global forgetting_config
            return config.get('forgetting_config', {})
        else:
            # Client not in list = clean client
            return {}
    
    # METHOD 3: Legacy behavior - use ATTACK_ALL_CLIENTS flag
    elif config.get('ATTACK_ALL_CLIENTS', False):
        # All clients get poisoned
        return config.get('forgetting_config', {})
    
    # METHOD 4: Only excluded clients get poisoned (legacy)
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
        print("Forget  ALL data from target clients")
        print("Retain  ALL data from remaining clients")
        print("\nUse case: Client leaves federation, privacy request,")
        print("          remove malicious client completely")
        
    elif mode == "DATA_LEVEL":
        print("Mode: DATA-LEVEL UNLEARNING")
        print("Description: Remove SPECIFIC poisoned/attacked data only")
        print(f"Target clients: {excluded_clients} (have poisoned data)")
        print(f"Forgetting config: {config.get('forgetting_config', {})}")
        print("Forget  Only attacked samples (forget set)")
        print("Retain  Clean data from ALL clients")
        print("            (including target clients' retrain set)")
        print("\nUse case: Remove backdoor samples, fix label-flipping,")
        print("          remove specific poisoned classes")
    
    print("="*60)


def print_poisoning_summary(dataloaders, config):
    """
    Print summary of which clients are poisoned and how
    
    Args:
        dataloaders: List of all dataloaders
        config: Config dict
    """
    print("\n" + "="*60)
    print("CLIENT POISONING SUMMARY")
    print("="*60)
    
    poisoned_clients = []
    clean_clients = []
    poisoning_details = {}
    
    for dataloader in dataloaders:
        client_id = dataloader.partition_id
        forget_size = len(dataloader.forgetloader.dataset) if dataloader.forgetloader else 0
        train_size = len(dataloader.retrainloader.dataset) if dataloader.retrainloader else 0
        
        if forget_size > 0:
            poisoned_clients.append(client_id)
            poison_ratio = forget_size / (forget_size + train_size) if (forget_size + train_size) > 0 else 0
            poisoning_details[client_id] = (forget_size, poison_ratio)
            print(f"Client {client_id:2d}: ⚠ POISONED - {forget_size:4d} poisoned samples ({poison_ratio*100:.1f}%)")
        else:
            clean_clients.append(client_id)
            print(f"Client {client_id:2d}: ✓ CLEAN - no poisoned data")
    
    print("-" * 60)
    print(f"Poisoned clients: {len(poisoned_clients)}/{len(dataloaders)} - {poisoned_clients}")
    print(f"Clean clients:    {len(clean_clients)}/{len(dataloaders)} - {clean_clients}")
    
    if poisoning_details:
        total_poisoned = sum(count for count, _ in poisoning_details.values())
        print(f"Total poisoned samples: {total_poisoned}")
    
    print("="*60)


def print_client_summary(dataloaders, excluded_clients, config):
    """
    Print summary of client data distribution
    
    Args:
        dataloaders: List of all dataloaders
        excluded_clients: List of client IDs to be unlearned
        config: Config dict
    """
    print("\n" + "="*60)
    print("CLIENT DATA SUMMARY")
    print("="*60)
    
    total_train = 0
    total_forget = 0
    total_val = 0
    total_test = 0
    
    for dataloader in dataloaders:
        client_id = dataloader.partition_id
        
        train_size = len(dataloader.retrainloader.dataset) if dataloader.retrainloader else 0
        forget_size = len(dataloader.forgetloader.dataset) if dataloader.forgetloader else 0
        val_size = len(dataloader.valloader.dataset) if dataloader.valloader else 0
        test_size = len(dataloader.testloader.dataset) if dataloader.testloader else 0
        
        total_train += train_size
        total_forget += forget_size
        total_val += val_size
        total_test += test_size
        
        status = "❌ TO UNLEARN" if client_id in excluded_clients else "✓ Keep"
        
        print(f"Client {client_id:2d} {status:12s}: Train={train_size:5d}, Forget={forget_size:4d}, Val={val_size:4d}, Test={test_size:4d}")
    
    print("-" * 60)
    print(f"{'TOTAL':15s}: Train={total_train:5d}, Forget={total_forget:4d}, Val={total_val:4d}, Test={total_test:4d}")
    print(f"\nClients to unlearn: {len(excluded_clients)}/{len(dataloaders)}")
    print(f"Clients to keep:    {len(dataloaders) - len(excluded_clients)}/{len(dataloaders)}")
    print("="*60)


def get_unlearning_dataloaders(all_dataloaders, excluded_clients, config):
    """
    Get appropriate dataloaders based on unlearning mode
    
    Returns:
        forget_dataloaders: Dataloaders with data to forget
        retain_dataloaders: Dataloaders with data to retain
    """
    mode = config.get('UNLEARNING_MODE', 'DATA_LEVEL')
    
    if mode == "CLIENT_LEVEL":
        # CLIENT-LEVEL: Forget ALL data from target clients
        forget_dataloaders = [all_dataloaders[i] for i in excluded_clients]
        retain_dataloaders = [dl for i, dl in enumerate(all_dataloaders) 
                             if i not in excluded_clients]
        
        print(f"\nCLIENT-LEVEL setup:")
        print(f"  Forget: {len(forget_dataloaders)} entire client(s)")
        print(f"  Retain: {len(retain_dataloaders)} client(s)")
        
    elif mode == "DATA_LEVEL":
        # DATA-LEVEL: Forget only poisoned data, retain clean data from all
        forget_dataloaders = [all_dataloaders[i] for i in excluded_clients]
        retain_dataloaders = all_dataloaders  # ALL clients (use their retrain sets)
        
        print(f"\nDATA-LEVEL setup:")
        print(f"  Forget: Poisoned data from {len(forget_dataloaders)} client(s)")
        print(f"  Retain: Clean data from ALL {len(retain_dataloaders)} clients")
    
    else:
        raise ValueError(f"Unknown UNLEARNING_MODE: {mode}")
    
    return forget_dataloaders, retain_dataloaders


def add_backdoor_trigger(data, config):
    """Add backdoor trigger to data"""
    trigger_size = config.get('BACKDOOR_TRIGGER_SIZE', 5)
    trigger_value = config.get('BACKDOOR_TRIGGER_VALUE', 1.0)
    
    backdoored = data.clone()
    
    if backdoored.dim() == 4:  # [B, C, H, W]
        B, C, H, W = backdoored.shape
        backdoored[:, :, H-trigger_size:H, W-trigger_size:W] = trigger_value
    
    return backdoored


def evaluate_model(model, dataloaders, device, config=None):
    """Evaluate model with comprehensive metrics"""
    model.eval()
    
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for dataloader in dataloaders:
            test_loader = dataloader.get_test_loader()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                for t, p in zip(target, pred):
                    t_item = t.item()
                    class_total[t_item] += 1
                    if p.item() == t_item:
                        class_correct[t_item] += 1
    
    overall_acc = correct / total if total > 0 else 0.0
    class_acc = {cls: class_correct[cls] / class_total[cls] 
                 for cls in class_total.keys()}
    
    return {
        'overall_acc': overall_acc,
        'class_acc': class_acc,
        'total_samples': total
    }


def evaluate_backdoor_attack(model, dataloaders, device, config):
    """Evaluate Backdoor Attack Success Rate (BASR)"""
    model.eval()
    
    target_label = config.get('BACKDOOR_TARGET_LABEL', 0)
    backdoor_correct = 0
    total = 0
    
    with torch.no_grad():
        for dataloader in dataloaders:
            test_loader = dataloader.get_test_loader()
            for data, target in test_loader:
                backdoored_data = add_backdoor_trigger(data, config)
                backdoored_data = backdoored_data.to(device)
                
                output = model(backdoored_data)
                pred = output.argmax(dim=1)
                
                backdoor_correct += (pred == target_label).sum().item()
                total += target.size(0)
    
    basr = backdoor_correct / total if total > 0 else 0.0
    
    return {
        'backdoor_asr': basr,
        'backdoor_samples': backdoor_correct,
        'total_samples': total
    }


def evaluate_confuse_attack(model, dataloaders, device, config):
    """Evaluate Confuse/Label-flipping Attack"""
    model.eval()
    
    map_confuse = config.get('MAP_CONFUSE', {})
    confusion_stats = defaultdict(lambda: {'correct': 0, 'confused': 0, 'total': 0})
    
    with torch.no_grad():
        for dataloader in dataloaders:
            test_loader = dataloader.get_test_loader()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                for t, p in zip(target, pred):
                    t_item = t.item()
                    p_item = p.item()
                    
                    if t_item in map_confuse:
                        target_confused = map_confuse[t_item]
                        confusion_stats[t_item]['total'] += 1
                        
                        if p_item == t_item:
                            confusion_stats[t_item]['correct'] += 1
                        elif p_item == target_confused:
                            confusion_stats[t_item]['confused'] += 1
    
    confuse_results = {}
    for source_class, stats in confusion_stats.items():
        if stats['total'] > 0:
            confuse_rate = stats['confused'] / stats['total']
            correct_rate = stats['correct'] / stats['total']
            confuse_results[f'class_{source_class}_confuse_rate'] = confuse_rate
            confuse_results[f'class_{source_class}_correct_rate'] = correct_rate
    
    return confuse_results


def train_federated(config, dataloaders, device):
    """Federated learning training phase"""
    print(f"\nStarting FL training on {device}")
    print(f"Attack type: {config.get('UNLEARNING_CASE', 'NORMAL')}")
    
    # Initialize global model
    model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    
    # Use DataParallel if multiple GPUs available (for Kaggle)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    for round_idx in range(config['num_rounds']):
        client_weights = []
        client_sizes = []
        
        # Client local training
        for client_id, dataloader in enumerate(dataloaders):
            train_loader = dataloader.get_combined_train_loader()
            if train_loader is None:
                continue
            
            # Clone model for local training
            local_model = SimpleResNet18(num_classes=config['num_classes']).to(device)
            if isinstance(model, nn.DataParallel):
                local_model.load_state_dict(model.module.state_dict())
            else:
                local_model.load_state_dict(model.state_dict())
            
            # Local training
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
            
            client_weights.append({k: v.cpu().clone() for k, v in local_model.state_dict().items()})
            client_sizes.append(len(train_loader.dataset))
        
        # FedAvg aggregation
        total_size = sum(client_sizes)
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = sum(
                w[key] * (size / total_size)
                for w, size in zip(client_weights, client_sizes)
            )
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(aggregated_weights)
        else:
            model.load_state_dict(aggregated_weights)
        
        # Evaluation
        if round_idx % config['eval_interval'] == 0:
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            results = evaluate_model(eval_model, dataloaders, device, config)
            print(f"\nRound {round_idx}:")
            print(f"  Overall Accuracy: {results['overall_acc']:.4f}")
            
            unlearning_case = config.get('UNLEARNING_CASE', 'NORMAL')
            
            if unlearning_case == 'BACKDOOR':
                backdoor_results = evaluate_backdoor_attack(eval_model, dataloaders, device, config)
                print(f"  Backdoor ASR: {backdoor_results['backdoor_asr']:.4f}")
            
            elif unlearning_case == 'CONFUSE':
                confuse_results = evaluate_confuse_attack(eval_model, dataloaders, device, config)
                print(f"  Confuse Attack Metrics:")
                for key, val in list(confuse_results.items())[:3]:  # Show first 3
                    print(f"    {key}: {val:.4f}")
    
    # Return unwrapped model
    return model.module if isinstance(model, nn.DataParallel) else model

def unlearn_with_fcu(model, forget_dataloaders, retain_dataloaders, config, device):
    """
    Perform FCU unlearning for CLIENT-LEVEL only
    Implements Model-Contrastive Unlearning (MCU) + Frequency-Guided Memory Preservation (FGMP)
    with separate local unlearning and post-training phases.
    
    Args:
        model: Pre-trained federated model
        forget_dataloaders: Dataloaders with data to forget (target client(s))
        retain_dataloaders: Dataloaders with data to retain (remaining clients)
        config: Configuration dict
        device: Device to run on
    """
    
    print(f"\n{'='*60}")
    print(f"FCU UNLEARNING - CLIENT LEVEL MODE")
    print(f"{'='*60}")
    
    # Count samples
    total_forget_samples = sum(
        len(dl.get_train_loader().dataset) if dl.get_train_loader() else 0
        for dl in forget_dataloaders
    )
    total_retain_samples = sum(
        len(dl.get_train_loader().dataset) if dl.get_train_loader() else 0
        for dl in retain_dataloaders
    )
    
    print(f"Samples to forget: {total_forget_samples}")
    print(f"Samples to retain: {total_retain_samples}")
    print()
    
    # Create reference model (downgraded: ImageNet pretrained init)
    reference_model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    reference_model.eval()
    print("✓ Reference model (downgraded pretrained) created for MCU")
    
    # Store trained model for FGMP and MCU
    trained_model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    trained_model.load_state_dict(model.state_dict())
    trained_model.eval()
    print("✓ Trained model saved for FGMP and MCU")
    
    # Hyperparameters from paper
    unlearn_lr = config.get('unlearn_lr', 1e-5)
    post_lr = config.get('post_lr', 1e-4)
    unlearn_iterations = config.get('unlearn_iterations', 100)
    T_fgmp = config.get('T_fgmp', 10)
    tau = config.get('tau', 0.5)
    post_rounds = config.get('post_rounds', 10)
    local_iterations = config.get('local_iterations', 20)
    r_fgmp = config.get('r_fgmp', 1/3)  # Low-frequency ratio approx
    
    print(f"\nHyperparameters:")
    print(f"  Local Unlearn Iterations: {unlearn_iterations}")
    print(f"  Unlearn LR (Adam): {unlearn_lr}")
    print(f"  Post-Training LR (Adam): {post_lr}")
    print(f"  MCU Tau: {tau}")
    print(f"  FGMP Frequency: every {T_fgmp} iterations")
    print(f"  FGMP Low-Freq Ratio: {r_fgmp}")
    print(f"  Post-Training Rounds: {post_rounds}")
    print(f"  Local Iterations per Round: {local_iterations}")
    print()
    
    criterion = nn.CrossEntropyLoss()
    
    # ============================================
    # Phase 1: Local Unlearning on Forget Data
    # ============================================
    print(f"{'-'*40}")
    print("Phase 1: Local Unlearning")
    print(f"{'-'*40}")
    
    model.train()
    
    # Get forget loaders (assume train_loader for client data)
    forget_loaders = [dl.get_train_loader() for dl in forget_dataloaders if dl.get_train_loader()]
    # Create cycling iterators (for iterations > len(loader))
    forget_iters = [cycle(loader) for loader in forget_loaders] if forget_loaders else []
    
    # Optimizer for local unlearning (Adam as per paper)
    optimizer = optim.Adam(
        model.parameters(),
        lr=unlearn_lr,
        betas=(0.9, 0.999)
    )
    
    for iteration in range(unlearn_iterations):
        optimizer.zero_grad()
        mcu_loss = 0.0
        mcu_count = 0
        
        for forget_iter in forget_iters:
            data, _ = next(forget_iter)  # Ignore targets for MCU
            data = data.to(device)
            
            with torch.no_grad():
                z_down = reference_model.get_features(data)
                z_tr = trained_model.get_features(data)
            z = model.get_features(data)
            
            # Cosine similarity
            sim_down = F.cosine_similarity(z, z_down, dim=-1)
            sim_tr = F.cosine_similarity(z, z_tr, dim=-1)
            
            # MCU loss
            numerator = torch.exp(sim_down / tau)
            denominator = numerator + torch.exp(sim_tr / tau)
            loss_batch = -torch.log(numerator / denominator).mean()
            
            mcu_loss += loss_batch
            mcu_count += 1
        
        if mcu_count > 0:
            mcu_loss = mcu_loss / mcu_count
            mcu_loss.backward()
            optimizer.step()
        
        # Apply FGMP periodically
        if (iteration + 1) % T_fgmp == 0:
            apply_fgmp(model, trained_model, device, r_fgmp)
        
        if iteration % 20 == 0:
            print(f"Iteration {iteration:3d}: MCU Loss={mcu_loss.item():.4f}" if mcu_count > 0 else f"Iteration {iteration:3d}: No forget data")
    
    print("✓ Local unlearning completed")
    
    # ============================================
    # Phase 2: Post-Training on Retain Clients (Simulated FedAvg)
    # ============================================
    print(f"\n{'-'*40}")
    print("Phase 2: Post-Training (FedAvg)")
    print(f"{'-'*40}")
    
    # Get retain loaders and sample sizes
    retain_loaders = [dl.get_train_loader() for dl in retain_dataloaders if dl.get_train_loader()]
    client_sizes = [len(loader.dataset) for loader in retain_loaders]
    total_retain = sum(client_sizes)
    
    if total_retain == 0:
        print("No retain data; returning unlearned model")
        return model
    
    for round_num in range(post_rounds):
        client_models = []
        client_opts = []
        
        for idx, loader in enumerate(retain_loaders):
            local_model = deepcopy(model).to(device)
            local_model.train()
            opt = optim.Adam(
                local_model.parameters(),
                lr=post_lr,
                betas=(0.9, 0.999)
            )
            local_iter = cycle(loader)  # Cycle if needed
            
            for _ in range(local_iterations):
                opt.zero_grad()
                data, target = next(local_iter)
                data, target = data.to(device), target.to(device)
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                opt.step()
            
            client_models.append(local_model)
            client_opts.append(opt)  # Not used further
        
        # Weighted average to update global model
        with torch.no_grad():
            for param in model.parameters():
                param.data.zero_()
            
            for idx, local_model in enumerate(client_models):
                weight = client_sizes[idx] / total_retain
                for global_param, local_param in zip(model.parameters(), local_model.parameters()):
                    global_param.data.add_(weight * local_param.data)
        
        print(f"Round {round_num + 1:2d}/{post_rounds}: Completed")
    
    print("✓ Post-training completed")
    return model

def apply_fgmp(model, trained_model, device, r=1/3):
    """
    Frequency-Guided Memory Preservation (FGMP)
    Preserve low-frequency components from trained model
    Keep high-frequency components from current (unlearned) model
    
    Args:
        model: Current model being unlearned
        trained_model: Original trained model (before unlearning)
        device: Device
        r: Low-frequency ratio
    """
    with torch.no_grad():
        for (name, param_current), (_, param_trained) in zip(
            model.named_parameters(),
            trained_model.named_parameters()
        ):
            # Apply only to convolutional weights (skip BN, bias, linear)
            if 'bn' in name or 'bias' in name or len(param_current.shape) != 4:
                continue
            
            # Conv shape: [H=out_ch, N=in_ch, d1=kh, d2=kw]
            H, N, d1, d2 = param_current.shape
            original_shape = param_current.shape
            
            # Reshape to [d1 * N, d2 * H] as per paper
            param_current_2d = param_current.permute(2, 1, 0, 3).reshape(d1 * N, d2 * H)
            param_trained_2d = param_trained.permute(2, 1, 0, 3).reshape(d1 * N, d2 * H)
            
            # Pad to next power of 2
            h, w = param_current_2d.shape
            pad_h = (2 ** int(np.ceil(np.log2(h)))) - h
            pad_w = (2 ** int(np.ceil(np.log2(w)))) - w
            
            if pad_h > 0 or pad_w > 0:
                param_current_2d = F.pad(param_current_2d, (0, pad_w, 0, pad_h))
                param_trained_2d = F.pad(param_trained_2d, (0, pad_w, 0, pad_h))
            
            # Apply 2D FFT
            fft_current = torch.fft.fft2(param_current_2d)
            fft_trained = torch.fft.fft2(param_trained_2d)
            
            # Get amplitude and phase (from current)
            amp_current = torch.abs(fft_current)
            amp_trained = torch.abs(fft_trained)
            phase_current = torch.angle(fft_current)
            
            # Create rectangular low-frequency mask (centered)
            H_fft, W_fft = fft_current.shape
            low_h = int(r * H_fft)
            low_w = int(r * W_fft)
            start_h = (H_fft - low_h) // 2
            start_w = (W_fft - low_w) // 2
            mask_low = torch.zeros(H_fft, W_fft, device=device)
            mask_low[start_h:start_h + low_h, start_w:start_w + low_w] = 1.0
            mask_high = 1.0 - mask_low
            
            # Combine: low-freq from trained, high-freq from current
            amp_combined = mask_low * amp_trained + mask_high * amp_current
            
            # Reconstruct FFT
            fft_combined = amp_combined * torch.exp(1j * phase_current)
            
            # Inverse FFT
            param_new = torch.fft.ifft2(fft_combined).real
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                param_new = param_new[:h, :w]
            
            # Reshape back to original [H, N, d1, d2]
            param_new = param_new.reshape(d1, N, H, d2).permute(2, 1, 0, 3)
            
            # Update parameter
            param_current.copy_(param_new)

def main():
    parser = argparse.ArgumentParser(description='FCU Training with Client-Specific Poisoning Control')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--excluded_clients', type=int, nargs='+', default=[0], 
                       help='Client IDs to unlearn')
    parser.add_argument('--is_unlearn', type=int, default=0, 
                       help='1 for unlearning phase, 0 for training only')
    parser.add_argument('--attack_type', type=str, choices=['NORMAL', 'CONFUSE', 'BACKDOOR'],
                       help='Override attack type from config')
    parser.add_argument('--attack_all', action='store_true',
                       help='Apply attack to all clients (deprecated, use config)')
    
    # Model loading arguments
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to pre-trained model to load (skip training phase)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip federated training phase, only do unlearning')
    
    # Partition control arguments
    parser.add_argument('--partition_dir', type=str, default='./partition_info',
                       help='Directory to save/load partition info')
    parser.add_argument('--force_new_partition', action='store_true',
                       help='Force create new partition even if exists')
    parser.add_argument('--verify_partitions', action='store_true',
                       help='Verify loaded partitions match saved ones')
    
    # Unlearning mode
    parser.add_argument('--unlearn_mode', type=str, choices=['CLIENT_LEVEL', 'DATA_LEVEL'],
                       help='Override unlearning mode from config')
    
    # NEW: Client-specific poisoning control (CLI override)
    parser.add_argument('--poison_clients', type=int, nargs='+', default=None,
                       help='Client IDs to poison (overrides config)')
    parser.add_argument('--poison_ratio', type=float, default=0.3,
                       help='Poisoning ratio for --poison_clients')
    parser.add_argument('--poison_classes', type=int, nargs='+', default=[0, 1],
                       help='Classes to poison for --poison_clients')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with CLI arguments if provided
    if args.attack_type:
        config['UNLEARNING_CASE'] = args.attack_type
    if args.attack_all:
        config['ATTACK_ALL_CLIENTS'] = True
    if args.unlearn_mode:
        config['UNLEARNING_MODE'] = args.unlearn_mode
    
    # Override poisoning config via CLI
    if args.poison_clients:
        print(f"\nOverriding config with CLI: Poisoning clients {args.poison_clients}")
        config['CLIENT_POISONING_CONFIG'] = {}
        for client_id in args.poison_clients:
            config['CLIENT_POISONING_CONFIG'][client_id] = {
                'classes': {cls: args.poison_ratio for cls in args.poison_classes}
            }
    
    # Set seed
    set_seed(config['SEED'])
    
    # Get device
    device = get_device(config.get('device', 'auto'))
    print(f"Using device: {device}")
    
    # Print experiment setup
    print("\n" + "="*60)
    print("EXPERIMENT SETUP")
    print("="*60)
    print(f"Dataset: {config['dataset']}")
    print(f"Num Clients: {config['num_clients']}")
    print(f"Attack Type: {config.get('UNLEARNING_CASE', 'NORMAL')}")
    print(f"Unlearning Mode: {config.get('UNLEARNING_MODE', 'DATA_LEVEL')}")
    print(f"Partition Directory: {args.partition_dir}")
    print(f"Force New Partition: {args.force_new_partition}")
    
    # Check if partitions exist
    partition_exists = os.path.exists(os.path.join(args.partition_dir, 'partition_client_0.pkl'))
    if partition_exists and not args.force_new_partition:
        print(f"✓ Found existing partitions, will load from disk")
    elif partition_exists and args.force_new_partition:
        print(f"⚠ Existing partitions found but --force_new_partition set, will create new")
    else:
        print(f"No existing partitions found, will create new")
    
    if args.load_model:
        print(f"Load Model From: {args.load_model}")
    print("="*60)
    
    # Print unlearning scenario
    if args.is_unlearn:
        print_unlearning_scenario(config, args.excluded_clients)
    
    # Create dataloaders with CLIENT-SPECIFIC poisoning
    print("\nCreating dataloaders with client-specific poisoning...")
    dataloaders = []
    
    for client_id in range(config['num_clients']):
        # Get client-specific forgetting config
        client_forgetting_config = get_client_forgetting_config(
            client_id, config, args.excluded_clients
        )
        
        # Print what we're doing for this client
        if client_forgetting_config:
            print(f"Client {client_id}: Poisoning classes {client_forgetting_config}")
        else:
            print(f"Client {client_id}: Clean (no poisoning)")
        
        loader = FCUDataLoader(
            partition_id=client_id,
            num_partitions=config['num_clients'],
            dataset_name=config['dataset'],
            seed=config['SEED'],
            forgetting_config=client_forgetting_config,  # Client-specific config
            config=config,
            save_partition=True,
            partition_save_dir=args.partition_dir,
            load_if_exists=not args.force_new_partition
        )
        dataloaders.append(loader)
    
    print(f"\n✓ Created dataloaders for {len(dataloaders)} clients")
    
    # Print poisoning summary
    print_poisoning_summary(dataloaders, config)
    
    # Print client summary
    print_client_summary(dataloaders, args.excluded_clients, config)
    
    # PHASE 1: Training or Load Model
    if args.load_model:
        print("\n" + "="*60)
        print("LOADING PRE-TRAINED MODEL")
        print("="*60)
        
        model, saved_config, partition_info_all = load_checkpoint_with_partitions(
            args.load_model, device
        )
        
        # Verify partition consistency if requested
        if args.verify_partitions and partition_info_all:
            consistent = verify_partition_consistency(dataloaders, partition_info_all)
            if not consistent:
                print("\n⚠ Warning: Partition mismatch detected!")
                print("The current data splits differ from the saved checkpoint.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return
        
        results = evaluate_model(model, dataloaders, device, config)
        print(f"\nLoaded Model Accuracy: {results['overall_acc']:.4f}")
        
    elif not args.skip_training:
        print("\n" + "="*60)
        print("PHASE 1: Federated Learning Training")
        print("="*60)
        
        model = train_federated(config, dataloaders, device)
        
        # Save model with partition info
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        
        attack_suffix = config.get('UNLEARNING_CASE', 'normal').lower()
        
        # Determine scope based on poisoning config
        if 'CLIENT_POISONING_CONFIG' in config:
            num_poisoned = len(config['CLIENT_POISONING_CONFIG'])
            attack_scope = f"{num_poisoned}clients"
        elif 'CLIENTS_TO_POISON' in config:
            num_poisoned = len(config['CLIENTS_TO_POISON'])
            attack_scope = f"{num_poisoned}clients"
        elif config.get('ATTACK_ALL_CLIENTS', False):
            attack_scope = "all"
        else:
            attack_scope = "partial"
        
        model_path = save_dir / f"trained_model_{attack_suffix}_{attack_scope}.pth"
        
        save_checkpoint(model, None, config['num_rounds'], dataloaders, model_path, config)
        
        # Also save standalone partition summary
        save_all_partitions_summary(dataloaders, config, args.partition_dir)
        
    else:
        raise ValueError("Must either provide --load_model or allow training phase")
    
    # PHASE 2: Unlearning
    if args.is_unlearn:
        print("\n" + "="*60)
        print("PHASE 2: Unlearning")
        print("="*60)
        
        # Get appropriate dataloaders based on unlearning mode
        forget_dataloaders, retain_dataloaders = get_unlearning_dataloaders(
            dataloaders, args.excluded_clients, config
        )
        
        model = unlearn_with_fcu(model, forget_dataloaders, retain_dataloaders, config, device)
        
        # Save unlearned model
        save_dir = Path("checkpoints")
        mode_suffix = config.get('UNLEARNING_MODE', 'data').lower()
        attack_suffix = config.get('UNLEARNING_CASE', 'normal').lower()
        
        if 'CLIENT_POISONING_CONFIG' in config:
            num_poisoned = len(config['CLIENT_POISONING_CONFIG'])
            attack_scope = f"{num_poisoned}clients"
        elif 'CLIENTS_TO_POISON' in config:
            num_poisoned = len(config['CLIENTS_TO_POISON'])
            attack_scope = f"{num_poisoned}clients"
        elif config.get('ATTACK_ALL_CLIENTS', False):
            attack_scope = "all"
        else:
            attack_scope = "partial"
        
        unlearn_path = save_dir / f"unlearned_{mode_suffix}_{attack_suffix}_{attack_scope}.pth"
        
        save_checkpoint(model, None, config['num_rounds'] + config.get('unlearn_epochs', 50),
                       dataloaders, unlearn_path, config)
        print(f"\n✓ Unlearned model saved to {unlearn_path}")
    
    # Final Evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    results = evaluate_model(model, dataloaders, device, config)
    print(f"\nOverall Accuracy: {results['overall_acc']:.4f}")
    print(f"\nPer-class Accuracy:")
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    mnist_classes = [str(i) for i in range(10)]
    fashionmnist_classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    dataset_name = config['dataset'].lower()
    if dataset_name == 'cifar10':
        class_names = cifar10_classes
    elif dataset_name == 'mnist':
        class_names = mnist_classes
    elif dataset_name == 'fashionmnist':
        class_names = fashionmnist_classes
    else:
        class_names = [str(i) for i in range(config['num_classes'])]
    
    for cls, acc in sorted(results['class_acc'].items()):
        class_name = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"  Class {cls} ({class_name}): {acc:.4f}")
    
    unlearning_case = config.get('UNLEARNING_CASE', 'NORMAL')
    
    if unlearning_case == 'BACKDOOR':
        backdoor_results = evaluate_backdoor_attack(model, dataloaders, device, config)
        print(f"\nBackdoor Attack Success Rate: {backdoor_results['backdoor_asr']:.4f}")
        print(f"Backdoored samples classified as target: {backdoor_results['backdoor_samples']}/{backdoor_results['total_samples']}")
    
    elif unlearning_case == 'CONFUSE':
        confuse_results = evaluate_confuse_attack(model, dataloaders, device, config)
        print(f"\nLabel-Flipping Attack Results:")
        for key, val in confuse_results.items():
            print(f"  {key}: {val:.4f}")
    
    print("\n" + "="*60)
    print("✓ Experiment Complete")
    print("="*60)


if __name__ == "__main__":
    main()
