import os
import numpy as np
import torchvision
from explainable_cnn.validators import BaseValidator


class CNNValidator(BaseValidator):
    def __init__(self):
        """
        Validations corresponding to CNN related features used in order to
        generate explanations.
        """
        pass

    def assert_layers(self, layer_names, model):
        """
        Validates if layers inside layer_names exists in the model.
        """
        copied_layers = layer_names.copy()

        for name, _ in model.named_modules():
            if name in copied_layers:
                copied_layers.remove(name)

        if len(copied_layers) != 0:
            raise ValueError(f"Some layers you passed doesn't exist",
                             f"in the model: {copied_layers}")

    def assert_input_shape(self, input_shape):
        if not isinstance(input_shape, tuple) and \
                not isinstance(input_shape, list):
            raise TypeError("Invalid type for attribute image_shape.")

        if len(input_shape) not in [2, 3]:
            raise ValueError("Input shape of image must be 2D or 3D.")

    def assert_valid_image_path(self, image_path):
        valid_image_types = [".png", ".jpg", ".jpeg"]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"File with given path does not exist: {image_path}"
            )
        if not any([image_path.endswith(x) for x in valid_image_types]):
            raise ValueError("Image type not supported yet.",
                             "Please pass one of following type:",
                             f"{valid_image_types}")

    def assert_valid_transforms(self, transforms):
        if not isinstance(transforms, torchvision.transforms.Compose) and \
                transforms is not None:
            raise TypeError("Attribute transform must be either None or of",
                            "type torchvision.transforms.Compose")
