from typing import Dict, Optional
from torch.utils.data import DataLoader, ConcatDataset
from .cifar_dataloader import load_datasets_with_forgetting


class FCUDataLoader:
    """Adapter class for FCU compatibility with partition saving"""
    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        dataset_name: str = "cifar10",
        seed: int = 42,
        forgetting_config: Dict = None,
        config: Dict = None,
        save_partition: bool = True,
        partition_save_dir: str = "./partition_info"
    ):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.dataset_name = dataset_name
        self.seed = seed
        self.forgetting_config = forgetting_config or {}
        self.config = config or {}
        
        # Load datasets (now returns partition_info too)
        (
            self.retrainloader,
            self.forgetloader,
            self.valloader,
            self.testloader,
            self.original_forget_loader,
            self.partition_info
        ) = load_datasets_with_forgetting(
            partition_id=partition_id,
            num_partitions=num_partitions,
            seed=seed,
            shuffle=True,
            forgetting_config=forgetting_config,
            dataset_name=dataset_name,
            config=config,
            save_partition=save_partition,
            partition_save_dir=partition_save_dir
        )
        
    def get_train_loader(self) -> Optional[DataLoader]:
        """Returns retrain loader"""
        return self.retrainloader
    
    def get_forget_loader(self) -> Optional[DataLoader]:
        """Returns forget loader (possibly with confuse/backdoor)"""
        return self.forgetloader
    
    def get_original_forget_loader(self) -> Optional[DataLoader]:
        """Returns original forget loader without modifications"""
        return self.original_forget_loader
    
    def get_val_loader(self) -> DataLoader:
        """Returns validation loader"""
        return self.valloader
    
    def get_test_loader(self) -> DataLoader:
        """Returns test loader"""
        return self.testloader
    
    def get_partition_info(self) -> Dict:
        """Returns partition information for saving"""
        return self.partition_info
    
    def get_combined_train_loader(self) -> Optional[DataLoader]:
        """Returns combined retrain + original forget loaders"""
        datasets = []
        if self.retrainloader is not None:
            datasets.append(self.retrainloader.dataset)
        if self.original_forget_loader is not None:
            datasets.append(self.original_forget_loader.dataset)
        
        if not datasets:
            return None
        
        combined_dataset = ConcatDataset(datasets)
        batch_size = self.config.get('RETRAIN_BATCH', 64)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True
        )
