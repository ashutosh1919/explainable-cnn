import torch
from torch import nn

from explainable_cnn.validators.base_validator import BaseValidator


class BaseExplainer:
    def __init__(self,
                 model,
                 device="cpu"):
        """
        This class creates base of explainer module which works
        as the fundamental module to work with this package.
        
        Args:
            model: `torch.nn.Module` object containing the PyTorch model
            device: `str` containing the type of device to execute the code
        """
        
        self.validator = BaseValidator()
        self.validator.assert_type(model, nn.Module)
        self.validator.assert_type(device, str)
        
        self.model = model
        self.device = self.get_device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def get_device(self, device):
        """
        Returns `torch.device` Object.
        
        Args:
            device: `str` containing the type of device to execute the code
        
        Returns: `torch.device` object to allocate all tensors related to model
        """
        return torch.device(device)
    
    def get_reverse_class_map(self, class_map):
        """
        Generates reverse class mapping from index to label map.
        
        Args:
            class_map: `dict` containing mapping from class index to label
        
        Returns: reverse class mapping from label to index
        """
        reverse_class_mapping = {}
        for index, label in class_map.items():
            reverse_class_mapping[label] = index
        return reverse_class_mapping