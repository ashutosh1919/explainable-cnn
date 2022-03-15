import pytest
import torch
import torchvision.models as models
from explainable_cnn.explainers import BaseExplainer


class TestBaseExplainer:
    obj = BaseExplainer(models.resnet18(), "cpu")
    
    @pytest.mark.parametrize("model", [{2}, torch.Tensor(2), "2"])
    def test_model_attr_fail(self, model):
        with pytest.raises(ValueError):
            BaseExplainer(model, "cpu")

    @pytest.mark.parametrize("model", [models.resnet18(), models.vgg16()])
    def test_model_attr_success(self, model):
        BaseExplainer(model, "cpu")
    
    @pytest.mark.parametrize("device", ["random", "value"])
    def test_device_attr_fail_runtime(self, device):
        with pytest.raises(RuntimeError):
            BaseExplainer(models.resnet18(), device)
    
    @pytest.mark.parametrize("device", [1, 0, 1.7])
    def test_device_attr_fail_value(self, device):
        with pytest.raises(ValueError):
            BaseExplainer(models.resnet18(), device)

    def test_get_device(self):
        cls = self.__class__
        assert isinstance(cls.obj.get_device("cpu"), torch.device)
    
    def test_get_reverse_mapping(self):
        cls = self.__class__
        rev_map = cls.obj.get_reverse_class_map({0: 'label0', 1: 'label1'})
        assert 'label0' in rev_map and rev_map['label0'] == 0
        assert 'label1' in rev_map and rev_map['label1'] == 1
        assert len(rev_map) == 2
    
    @pytest.mark.parametrize("raw_layers, unique_layers",
                             [(["relu", "layer1", "layer2", "relu"],
                               ["relu", "layer1", "layer2"]),
                              (["relu", "layer1"],
                               ["relu", "layer1"])])
    def test_get_unique_layers(self, raw_layers, unique_layers):
        cls = self.__class__
        output_layers = cls.obj.get_unique_layers(raw_layers)
        assert output_layers == unique_layers
