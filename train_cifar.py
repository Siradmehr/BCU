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
    print(f"Attack ALL clients: {config.get('ATTACK_ALL_CLIENTS', False)}")
    
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
    Perform FCU unlearning with support for both CLIENT-LEVEL and DATA-LEVEL
    Implements Model-Contrastive Unlearning (MCU) + Frequency-Guided Memory Preservation (FGMP)
    
    Args:
        model: Pre-trained federated model
        forget_dataloaders: Dataloaders with data to forget
        retain_dataloaders: Dataloaders with data to retain
        config: Configuration dict
        device: Device to run on
    """
    mode = config.get('UNLEARNING_MODE', 'DATA_LEVEL')
    
    print(f"\n{'='*60}")
    print(f"FCU UNLEARNING - {mode} MODE")
    print(f"{'='*60}")
    
    # Count samples
    if mode == "CLIENT_LEVEL":
        # For client-level, forget ALL data from target clients
        total_forget_samples = sum(
            len(dl.get_combined_train_loader().dataset) if dl.get_combined_train_loader() else 0
            for dl in forget_dataloaders
        )
    else:  # DATA_LEVEL
        # For data-level, forget only the forget set
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
    print()
    
    # Create reference model (randomly initialized) for MCU
    reference_model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    reference_model.eval()
    print("✓ Reference model (random init) created for MCU")
    
    # Store trained model for FGMP
    trained_model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    trained_model.load_state_dict(model.state_dict())
    trained_model.eval()
    print("✓ Trained model saved for FGMP")
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.get('unlearn_lr', 0.01),
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Hyperparameters
    lambda_mcu = config.get('lambda_mcu', 1.0)
    lambda_retain = config.get('lambda_retain', 1.0)
    unlearn_epochs = config.get('unlearn_epochs', 50)
    T_fgmp = config.get('T_fgmp', 10)
    
    print(f"\nHyperparameters:")
    print(f"  Epochs: {unlearn_epochs}")
    print(f"  Learning Rate: {config.get('unlearn_lr', 0.01)}")
    print(f"  Lambda MCU (forget): {lambda_mcu}")
    print(f"  Lambda Retain: {lambda_retain}")
    print(f"  FGMP Frequency: every {T_fgmp} iterations")
    print()
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    iteration = 0
    for epoch in range(unlearn_epochs):
        epoch_mcu_loss = 0.0
        epoch_retain_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        # ============================================
        # Get forget loaders based on mode
        # ============================================
        forget_loaders = []
        if mode == "CLIENT_LEVEL":
            # Forget ALL data from target clients
            for dl in forget_dataloaders:
                if dl.get_combined_train_loader() is not None:
                    forget_loaders.append(dl.get_combined_train_loader())
        else:  # DATA_LEVEL
            # Forget only poisoned data
            for dl in forget_dataloaders:
                if dl.get_forget_loader() is not None:
                    forget_loaders.append(dl.get_forget_loader())
        
        # Get retain loaders
        retrain_loaders = []
        for dl in retain_dataloaders:
            if dl.get_train_loader() is not None:
                retrain_loaders.append(dl.get_train_loader())
        
        # Create iterators
        forget_iters = [iter(loader) for loader in forget_loaders] if forget_loaders else []
        retrain_iters = [iter(loader) for loader in retrain_loaders] if retrain_loaders else []
        
        # Determine number of iterations
        max_forget_batches = max([len(loader) for loader in forget_loaders]) if forget_loaders else 0
        max_retrain_batches = max([len(loader) for loader in retrain_loaders]) if retrain_loaders else 0
        num_iterations = max(max_forget_batches, max_retrain_batches)
        
        for batch_idx in range(num_iterations):
            optimizer.zero_grad()
            total_loss = 0.0
            
            # ============================================
            # Part 1: MCU Loss on Forget Set
            # Push forget set features toward random model
            # ============================================
            mcu_loss = 0.0
            mcu_count = 0
            
            if forget_iters:
                for forget_iter in forget_iters:
                    try:
                        data, target = next(forget_iter)
                    except StopIteration:
                        continue
                    
                    data = data.to(device)
                    
                    features_current = model.get_features(data)
                    
                    with torch.no_grad():
                        features_ref = reference_model.get_features(data)
                    
                    mcu_loss += F.mse_loss(features_current, features_ref)
                    mcu_count += 1
                
                if mcu_count > 0:
                    mcu_loss = mcu_loss / mcu_count
                    total_loss += lambda_mcu * mcu_loss
                    epoch_mcu_loss += mcu_loss.item()
            
            # ============================================
            # Part 2: Retain Loss on Retrain Set
            # Continue normal training on remaining data
            # ============================================
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
            
            # Backpropagation
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
                epoch_total_loss += total_loss.item()
                num_batches += 1
            
            iteration += 1
            
            # ============================================
            # Part 3: FGMP - Frequency-Guided Memory Preservation
            # Apply periodically to preserve global knowledge
            # ============================================
            if iteration % T_fgmp == 0 and iteration > 0:
                apply_fgmp(model, trained_model, device)
        
        # Print progress
        avg_mcu = epoch_mcu_loss / num_batches if num_batches > 0 else 0.0
        avg_retain = epoch_retain_loss / num_batches if num_batches > 0 else 0.0
        avg_total = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Total={avg_total:.4f}, MCU={avg_mcu:.4f}, Retain={avg_retain:.4f}")
    
    print(f"\n✓ Unlearning completed")
    return model


def apply_fgmp(model, trained_model, device):
    """
    Frequency-Guided Memory Preservation (FGMP)
    Preserve low-frequency components from trained model
    Keep high-frequency components from current (unlearned) model
    
    Args:
        model: Current model being unlearned
        trained_model: Original trained model (before unlearning)
        device: Device
    """
    with torch.no_grad():
        for (name, param_current), (_, param_trained) in zip(
            model.named_parameters(),
            trained_model.named_parameters()
        ):
            # Skip batch norm and bias parameters
            if 'bn' in name or 'bias' in name or len(param_current.shape) < 2:
                continue
            
            # Get original shape
            original_shape = param_current.shape
            
            # Reshape to 2D for FFT
            if len(original_shape) == 4:  # Conv layers [out_ch, in_ch, h, w]
                C_out, C_in, H, W = original_shape
                param_current_2d = param_current.reshape(C_out * C_in, H * W)
                param_trained_2d = param_trained.reshape(C_out * C_in, H * W)
            elif len(original_shape) == 2:  # Linear layers [out, in]
                param_current_2d = param_current
                param_trained_2d = param_trained
            else:
                continue
            
            # Pad to suitable size for FFT if needed
            h, w = param_current_2d.shape
            pad_h = (2 ** int(np.ceil(np.log2(h)))) - h
            pad_w = (2 ** int(np.ceil(np.log2(w)))) - w
            
            if pad_h > 0 or pad_w > 0:
                param_current_2d = F.pad(param_current_2d, (0, pad_w, 0, pad_h))
                param_trained_2d = F.pad(param_trained_2d, (0, pad_w, 0, pad_h))
            
            # Apply 2D FFT
            fft_current = torch.fft.fft2(param_current_2d)
            fft_trained = torch.fft.fft2(param_trained_2d)
            
            # Get amplitude and phase
            amp_current = torch.abs(fft_current)
            amp_trained = torch.abs(fft_trained)
            phase_current = torch.angle(fft_current)
            
            # Create low-frequency mask (center region)
            H_fft, W_fft = fft_current.shape
            center_h, center_w = H_fft // 2, W_fft // 2
            
            # Low-frequency radius (preserve 30% of frequency spectrum)
            radius = min(center_h, center_w) // 3
            
            # Create circular mask
            Y, X = torch.meshgrid(
                torch.arange(H_fft, device=device),
                torch.arange(W_fft, device=device),
                indexing='ij'
            )
            distance = torch.sqrt((Y - center_h) ** 2 + (X - center_w) ** 2)
            mask_low = (distance <= radius).float()
            mask_high = 1.0 - mask_low
            
            # Combine: low-freq from trained model, high-freq from current model
            amp_combined = mask_low * amp_trained + mask_high * amp_current
            
            # Reconstruct FFT with combined amplitude
            fft_combined = amp_combined * torch.exp(1j * phase_current)
            
            # Apply inverse FFT
            param_new = torch.fft.ifft2(fft_combined).real
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                param_new = param_new[:h, :w]
            
            # Reshape back to original shape
            if len(original_shape) == 4:
                param_new = param_new.reshape(original_shape)
            
            # Update parameter
            param_current.copy_(param_new)


def main():
    parser = argparse.ArgumentParser(description='FCU Training with Client-Level and Data-Level Unlearning')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--excluded_clients', type=int, nargs='+', default=[0], 
                       help='Client IDs to unlearn')
    parser.add_argument('--is_unlearn', type=int, default=0, 
                       help='1 for unlearning phase, 0 for training only')
    parser.add_argument('--attack_type', type=str, choices=['NORMAL', 'CONFUSE', 'BACKDOOR'],
                       help='Override attack type from config')
    parser.add_argument('--attack_all', action='store_true',
                       help='Apply attack to all clients')
    
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
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings
    if args.attack_type:
        config['UNLEARNING_CASE'] = args.attack_type
    if args.attack_all:
        config['ATTACK_ALL_CLIENTS'] = True
    if args.unlearn_mode:
        config['UNLEARNING_MODE'] = args.unlearn_mode
    
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
    
    # Create dataloaders with partition saving/loading
    print("\nCreating dataloaders...")
    dataloaders = []
    attack_all_clients = config.get('ATTACK_ALL_CLIENTS', False)
    
    for client_id in range(config['num_clients']):
        if attack_all_clients:
            forgetting_config = config.get('forgetting_config', {})
        elif client_id in args.excluded_clients:
            forgetting_config = config.get('forgetting_config', {})
        else:
            forgetting_config = {}
        
        loader = FCUDataLoader(
            partition_id=client_id,
            num_partitions=config['num_clients'],
            dataset_name=config['dataset'],
            seed=config['SEED'],
            forgetting_config=forgetting_config,
            config=config,
            save_partition=True,
            partition_save_dir=args.partition_dir,
            load_if_exists=not args.force_new_partition
        )
        dataloaders.append(loader)
    
    print(f"✓ Created dataloaders for {len(dataloaders)} clients")
    
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
        attack_scope = "all" if config.get('ATTACK_ALL_CLIENTS', False) else "partial"
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
        attack_scope = "all" if config.get('ATTACK_ALL_CLIENTS', False) else "partial"
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
