"""
Device utilities for MPS/CUDA/CPU compatibility
"""

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """
    Auto-detect or select device
    
    Args:
        device_str: "auto", "mps", "cuda", or "cpu"
        
    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("cpu")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def to_device(data, device):
    """
    Move data to device, handling MPS limitations
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
