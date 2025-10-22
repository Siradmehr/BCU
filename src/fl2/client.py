"""
FCU Client
"""

import torch
import torch.nn as nn
from src.datasets.fcu_adapter import FCUDataLoader
from models.resnet import SimpleResNet18


class Client:
    """Federated Learning Client"""
    
    def __init__(self, client_id, cfg, device, forgetting_config):
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        
        # Local model
        self.model = SimpleResNet18(num_classes=cfg['num_classes']).to(device)
        
        # Load data
        self.dataloader = FCUDataLoader(
            partition_id=client_id,
            num_partitions=cfg['num_clients'],
            dataset_name=cfg['dataset'],
            seed=cfg['SEED'],
            forgetting_config=forgetting_config,
            config=cfg,
            save_partition=True,
            partition_save_dir=cfg.get('partition_dir', './partition_info'),
            load_if_exists=not cfg.get('force_new_partition', False)
        )
    
    def set_model(self, weights):
        """Receive global model from server"""
        self.model.load_state_dict(weights)
    
    def train(self):
        """Local training"""
        train_loader = self.dataloader.get_combined_train_loader()
        
        if train_loader is None or len(train_loader.dataset) == 0:
            return None, 0
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg['learning_rate'],
            momentum=self.cfg['momentum'],
            weight_decay=self.cfg['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.cfg['local_epochs']):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        size = len(train_loader.dataset)
        
        return weights, size
    
    def get_forget_loader(self, mode='DATA_LEVEL'):
        """Get forget data loader"""
        if mode == 'CLIENT_LEVEL':
            return self.dataloader.get_combined_train_loader()
        else:
            return self.dataloader.get_forget_loader()
    
    def get_retain_loader(self):
        """Get retain data loader"""
        return self.dataloader.get_train_loader()
