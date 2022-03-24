import torch
from torch import nn

from explainable_cnn.validators import BaseValidator


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
        if not hasattr(self, 'validator'):
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

    def get_unique_layers(self, layer_names):
        """
        Remove duplicates from the `layer_names`

        Args:
            layer_names: `list` containing names of layers.

        Returns: `list` containing unique layer names.
        """
        unique_layers = []
        for layer in layer_names:
            if layer not in unique_layers:
                unique_layers.append(layer)
        return unique_layers

    def get_label_name_from_index(self, index):
        """
        Extracts name of a label from index.

        Args:
            index: `int` containing index of label.

        Returns: `str` containing name of label.
        """
        self.validator.assert_type(index, int)

        if not hasattr(self, 'class_map'):
            raise RuntimeError("Object doesn't have property 'class_map'")

        return self.class_map[index]

    def get_label_index_from_name(self, name):
        """
        Extracts index of label from name.

        Args:
            name: `str` containing name of label.

        Returns: `int` containing index of label.
        """
        self.validator.assert_type(name, str)

        if not hasattr(self, 'reverse_class_map'):
            raise RuntimeError(
                "Object doesn't have property 'reverse_class_map'"
            )

        return self.reverse_class_map[name]

    def get_label_information(self, label):
        """
        Extracts name and index of label.

        Args:
            label: `str` for label name or `int` for label index

        Returns:
            label_index: `int` for label index
            label_name: `str` for label name
        """

        if isinstance(label, str):
            self.validator.assert_label_name_validity(self.reverse_class_map,
                                                      label)
            label_index = self.get_label_index_from_name(label)
            label_name = label
        elif isinstance(label, int):
            self.validator.assert_label_index_validity(self.class_map, label)
            label_index = label
            label_name = self.get_label_name_from_index(label)
        else:
            raise TypeError("image_label must be of type int or str")
        return label_index, label_name
