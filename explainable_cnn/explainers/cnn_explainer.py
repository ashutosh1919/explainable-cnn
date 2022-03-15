import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision

from explainable_cnn.explainers.base_explainer import BaseExplainer
from explainable_cnn.explainers.grad_cam import GradCAM
from explainable_cnn.validators import CNNValidator


class CNNExplainer(BaseExplainer):
    def __init__(self,
                 model,
                 class_map,
                 device="cpu"):
        """
        This class creates base of explainer module which works
        as the fundamental module to work with this package.
        
        Args:
            model: `torch.nn.Module` object containing the PyTorch model
            class_map: `dict` containing mapping from class index to label
            device: `str` containing the type of device to execute the code
        """
        
        self.validator = CNNValidator()
        super(CNNExplainer, self).__init__(model, device)
        
        self.class_map = class_map
        self.reverse_class_map = self.get_reverse_class_map(self.class_map)
    
    def preprocess_image(self,
                         image_path,
                         input_shape,
                         transforms=None):
        """
        Preprocesses image before inputting to the model.

        Args:
            image_path: `str` containing relative/absolute path of image
            input_shape: `tuple` or `list` containing the shape in which the of image
                has to be resized before input.
            transforms: (Optional) `torchvision.transforms.Compose` object consisting of
                transformations that you want to apply on the image.
        
        Returns: Tuple containing two images: raw and transformed image.
        """
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(raw_image, input_shape)
        if transforms is None:
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
        augmented_image = transforms(raw_image[..., ::-1].copy())
        return raw_image, augmented_image
    
    def get_grad_cam_infused_image(self,
                                   gcam,
                                   raw_image,
                                   alpha_required=False):
        """
        Infuses Grad CAM image with the raw image so that we can visualize the
        final outcome.

        Args:
            gcam: Generated Grad CAM of type `torch.Tensor`
            raw_image: Raw image which is in shape `input_shape`
            alpha_required: (Optional) `True` if you want alpha channel in
                the output image `False` otherwise.
        
        Returns: `torch.Tensor` containing Grad CAM infused image.
        """
        gcam = gcam.cpu().numpy()
        cmap = cm.jet_r(gcam)[..., :3] * 255.0
        if alpha_required:
            alpha = gcam[..., None]
            gcam = alpha * cmap + (1 - alpha) * raw_image
        else:
            gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        return gcam

    def get_grad_cam(self,
                     image_path,
                     image_label,
                     input_shape,
                     layers,
                     transforms=None):
        """
        Plots Grad CAM of specific layers for specific image-label pair.

        Args:
            image_path: `str` containing relative/absolute path of image
            image_label: `int` dictating the index of the label starting from 0
                or `str` containing label text
            input_shape: `tuple` or `list` containing the shape in which the of image
                has to be resized before input.
            layers: `list` of `str` containing names of layers
            transforms: (Optional) `torchvision.transforms.Compose` object consisting of
                transformations that you want to apply on the image.
        
        Returns: `list` containing grad CAM images of for each layer in `layers`.
        """
        
        self.validator.assert_layers(layers, self.model)
        self.validator.assert_input_shape(input_shape)
        self.validator.assert_valid_transforms(transforms)

        input_shape = (*input_shape, )

        if isinstance(image_label, str):
            self.validator.assert_label_name_validity(self.reverse_class_map, image_label)
            label_index = self.get_label_index_from_name(image_label)
            label_name = image_label
        elif isinstance(image_label, int):
            self.validator.assert_label_index_validity(self.class_map, image_label)
            label_index = image_label
            label_name = self.get_label_name_from_index(image_label)
        else:
            raise TypeError("image_label must be of type int or str")
        
        raw_image, augmented_image = self.preprocess_image(image_path,
                                                           input_shape,
                                                           transforms)
        augmented_image = torch.stack([augmented_image]).to(self.device)
        
        gcam = GradCAM(model=self.model)
        probs, ids = gcam.forward(augmented_image)
        ids_ = torch.LongTensor([[label_index]]).to(self.device)
        gcam.backward(ids=ids_)

        final_images = []
        for layer in layers:
            regions = gcam.generate(target_layer=layer)
            final_images.append(
                self.get_grad_cam_infused_image(
                    regions[0, 0],
                    raw_image
                )
            )
        
        return final_images
