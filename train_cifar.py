"""
Training script for FCU with support for loading pre-trained models
Supports resuming from checkpoint to skip federated training
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

from src.datasets.fcu_adapter import FCUDataLoader
from src.utils.device_utils import get_device


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleResNet18(nn.Module):
    """Simple ResNet18 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False, num_classes=num_classes)
        
        # Adjust first conv for CIFAR-10 (32x32)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before FC layer"""
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


def load_model(model_path, device, num_classes=10):
    """Load a pre-trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    model = SimpleResNet18(num_classes=num_classes).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def save_checkpoint(model, optimizer, epoch, path, config=None):
    """Save a comprehensive checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to {path}")


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
    """Evaluate Backdoor Attack Success Rate"""
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
    print(f"Starting FL training on {device}")
    print(f"Unlearning case: {config.get('UNLEARNING_CASE', 'NORMAL')}")
    print(f"Attack ALL clients: {config.get('ATTACK_ALL_CLIENTS', False)}")
    
    # Initialize global model
    model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    
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
        model.load_state_dict(aggregated_weights)
        
        # Evaluation
        if round_idx % config['eval_interval'] == 0:
            results = evaluate_model(model, dataloaders, device, config)
            print(f"\nRound {round_idx}:")
            print(f"  Overall Accuracy: {results['overall_acc']:.4f}")
            
            unlearning_case = config.get('UNLEARNING_CASE', 'NORMAL')
            
            if unlearning_case == 'BACKDOOR':
                backdoor_results = evaluate_backdoor_attack(model, dataloaders, device, config)
                print(f"  Backdoor ASR: {backdoor_results['backdoor_asr']:.4f}")
            
            elif unlearning_case == 'CONFUSE':
                confuse_results = evaluate_confuse_attack(model, dataloaders, device, config)
                print(f"  Confuse Attack Metrics:")
                for key, val in list(confuse_results.items())[:3]:  # Show first 3
                    print(f"    {key}: {val:.4f}")
    
    return model


def unlearn_with_fcu(model, target_dataloaders, config, device):
    """
    Perform FCU unlearning on target clients
    
    Args:
        model: Pre-trained federated model
        target_dataloaders: List of dataloaders for clients to unlearn
        config: Configuration dict
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print("PHASE 2: FCU Unlearning")
    print(f"{'='*60}")
    
    # Create reference model (randomly initialized)
    reference_model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    reference_model.eval()
    
    # Store trained model for FGMP
    trained_model = SimpleResNet18(num_classes=config['num_classes']).to(device)
    trained_model.load_state_dict(model.state_dict())
    trained_model.eval()
    
    # Unlearning optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.get('unlearn_lr', 0.01),
        momentum=0.9,
        weight_decay=5e-4
    )
    
    lambda_mcu = config.get('lambda_mcu', 1.0)
    unlearn_epochs = config.get('unlearn_epochs', 50)
    
    print(f"Unlearning for {unlearn_epochs} epochs")
    print(f"Lambda MCU: {lambda_mcu}")
    
    model.train()
    
    for epoch in range(unlearn_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for dataloader in target_dataloaders:
            forget_loader = dataloader.get_forget_loader()
            if forget_loader is None:
                continue
            
            for data, target in forget_loader:
                data = data.to(device)
                
                optimizer.zero_grad()
                
                # Extract features
                features_unlearn = model.get_features(data)
                
                with torch.no_grad():
                    features_ref = reference_model.get_features(data)
                
                # MCU loss: push features toward reference (random) model
                mcu_loss = F.mse_loss(features_unlearn, features_ref)
                
                loss = lambda_mcu * mcu_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if epoch % 10 == 0:
            print(f"Unlearning Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print(f"✓ Unlearning completed")
    return model


def main():
    parser = argparse.ArgumentParser(description='FCU Training with Model Loading Support')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--excluded_clients', type=int, nargs='+', default=[0], 
                       help='Client IDs to unlearn')
    parser.add_argument('--is_unlearn', type=int, default=0, 
                       help='1 for unlearning phase, 0 for training only')
    parser.add_argument('--attack_type', type=str, choices=['NORMAL', 'CONFUSE', 'BACKDOOR'],
                       help='Override attack type from config')
    parser.add_argument('--attack_all', action='store_true',
                       help='Apply attack to all clients')
    
    # NEW: Model loading arguments
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to pre-trained model to load (skip training phase)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip federated training phase, only do unlearning')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings
    if args.attack_type:
        config['UNLEARNING_CASE'] = args.attack_type
    if args.attack_all:
        config['ATTACK_ALL_CLIENTS'] = True
    
    # Set seed
    set_seed(config['SEED'])
    
    # Get device
    device = get_device(config.get('device', 'auto'))
    print(f"Using device: {device}")
    
    # Print experiment setup
    print("\n" + "="*60)
    print(f"EXPERIMENT SETUP")
    print("="*60)
    print(f"Dataset: {config['dataset']}")
    print(f"Num Clients: {config['num_clients']}")
    print(f"Unlearning Case: {config.get('UNLEARNING_CASE', 'NORMAL')}")
    print(f"Attack ALL Clients: {config.get('ATTACK_ALL_CLIENTS', False)}")
    print(f"Clients to Forget: {args.excluded_clients}")
    print(f"Skip Training: {args.skip_training or args.load_model is not None}")
    if args.load_model:
        print(f"Load Model From: {args.load_model}")
    print("="*60 + "\n")
    
    # Create dataloaders
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
            config=config
        )
        dataloaders.append(loader)
    
    print(f"✓ Created dataloaders for {len(dataloaders)} clients\n")
    
    # PHASE 1: Training or Load Model
    if args.load_model:
        # Load pre-trained model
        print("="*60)
        print("LOADING PRE-TRAINED MODEL")
        print("="*60)
        model = load_model(args.load_model, device, config['num_classes'])
        
        # Evaluate loaded model
        results = evaluate_model(model, dataloaders, device, config)
        print(f"Loaded Model Accuracy: {results['overall_acc']:.4f}\n")
        
    elif not args.skip_training:
        # Train from scratch
        print("="*60)
        print("PHASE 1: Federated Learning Training")
        print("="*60)
        model = train_federated(config, dataloaders, device)
        
        # Save trained model
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        
        attack_suffix = config.get('UNLEARNING_CASE', 'normal').lower()
        attack_scope = "all" if config.get('ATTACK_ALL_CLIENTS', False) else "partial"
        model_path = save_dir / f"trained_model_{attack_suffix}_{attack_scope}.pth"
        
        save_checkpoint(model, None, config['num_rounds'], model_path, config)
        print(f"\n✓ Model saved to {model_path}")
    else:
        raise ValueError("Must either provide --load_model or allow training phase")
    
    # PHASE 2: Unlearning
    if args.is_unlearn:
        print("\n" + "="*60)
        print("PHASE 2: Unlearning")
        print("="*60)
        
        # Get dataloaders for clients to unlearn
        target_dataloaders = [dataloaders[i] for i in args.excluded_clients]
        
        # Perform unlearning
        model = unlearn_with_fcu(model, target_dataloaders, config, device)
        
        # Save unlearned model
        save_dir = Path("checkpoints")
        attack_suffix = config.get('UNLEARNING_CASE', 'normal').lower()
        attack_scope = "all" if config.get('ATTACK_ALL_CLIENTS', False) else "partial"
        unlearn_path = save_dir / f"unlearned_model_{attack_suffix}_{attack_scope}.pth"
        
        torch.save(model.state_dict(), unlearn_path)
        print(f"\n✓ Unlearned model saved to {unlearn_path}")
    
    # Final Evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    results = evaluate_model(model, dataloaders, device, config)
    print(f"Overall Accuracy: {results['overall_acc']:.4f}")
    print(f"\nPer-class Accuracy:")
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    for cls, acc in sorted(results['class_acc'].items()):
        print(f"  Class {cls} ({cifar10_classes[cls]}): {acc:.4f}")
    
    unlearning_case = config.get('UNLEARNING_CASE', 'NORMAL')
    
    if unlearning_case == 'BACKDOOR':
        backdoor_results = evaluate_backdoor_attack(model, dataloaders, device, config)
        print(f"\nBackdoor Attack Success Rate: {backdoor_results['backdoor_asr']:.4f}")
    
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
