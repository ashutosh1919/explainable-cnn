import pytest
import torchvision
import torchvision.models as models
from explainable_cnn import CNNExplainer


class TestCNNExplainer:
    obj = CNNExplainer(models.resnet18(),
                       {0: "Cat", 1: "Dog"},
                       "cpu")

    def test_get_label_name_from_index(self):
        cls = self.__class__
        assert cls.obj.get_label_name_from_index(0) == "Cat"
        assert cls.obj.get_label_name_from_index(1) == "Dog"

    def test_get_label_index_from_name(self):
        cls = self.__class__
        assert cls.obj.get_label_index_from_name("Cat") == 0
        assert cls.obj.get_label_index_from_name("Dog") == 1

    def test_get_label_information(self):
        cls = self.__class__
        label_index, label_name = cls.obj.get_label_information("Cat")
        assert label_index == 0
        assert label_name == "Cat"
        label_index, label_name = cls.obj.get_label_information(1)
        assert label_index == 1
        assert label_name == "Dog"

    def test_get_grad_cam_image_label_type_fail(self, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_grad_cam(image_file, 1.5, (224, 224), ["relu"])

    @pytest.mark.parametrize("image_label", ["Tiger", 2])
    def test_get_grad_cam_image_label_value_fail(self, image_label, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_grad_cam(image_file, image_label, (224, 224), ["relu"])

    @pytest.mark.parametrize("layers", [["random", "value"], [1, 2], [1.5]])
    def test_get_grad_cam_layers_fail(self, layers, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_grad_cam(image_file, 0, (224, 224), layers)

    @pytest.mark.parametrize("input_shape",
                             [(1,), [1], (1, 2, 3, 4), [1, 2, 3, 4]])
    def test_get_grad_cam_input_shape_value_fail(self, input_shape, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_grad_cam(image_file, 0, input_shape, ["relu"])

    @pytest.mark.parametrize("input_shape", [1, 1.5, "random"])
    def test_get_grad_cam_input_shape_type_fail(self, input_shape, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_grad_cam(image_file, 0, input_shape, ["relu"])

    def test_get_guided_grad_cam_image_label_type_fail(self, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_guided_grad_cam(image_file, 1.5, (224, 224), ["relu"])

    @pytest.mark.parametrize("image_label", ["Tiger", 2])
    def test_get_guided_grad_cam_image_label_value_fail(self,
                                                        image_label,
                                                        tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_guided_grad_cam(image_file,
                                        image_label,
                                        (224, 224),
                                        ["relu"])

    @pytest.mark.parametrize("layers", [["random", "value"], [1, 2], [1.5]])
    def test_get_guided_grad_cam_layers_fail(self, layers, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_guided_grad_cam(image_file, 0, (224, 224), layers)

    @pytest.mark.parametrize("input_shape",
                             [(1,), [1], (1, 2, 3, 4), [1, 2, 3, 4]])
    def test_get_guided_grad_cam_input_shape_value_fail(self,
                                                        input_shape,
                                                        tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_guided_grad_cam(image_file, 0, input_shape, ["relu"])

    @pytest.mark.parametrize("input_shape", [1, 1.5, "random"])
    def test_get_guided_grad_cam_input_shape_type_fail(self,
                                                       input_shape,
                                                       tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_guided_grad_cam(image_file, 0, input_shape, ["relu"])

    @pytest.mark.parametrize("transforms", [1, 1.5, "random value", [1, 2, 3]])
    def test_get_guided_grad_cam_transforms_fail(self, transforms, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_guided_grad_cam(image_file,
                                        0,
                                        (224, 224),
                                        ["relu"],
                                        transforms)

    def test_get_guided_back_propagation_image_label_type_fail(self, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_guided_back_propagation(image_file, 1.5, (224, 224))

    @pytest.mark.parametrize("image_label", ["Tiger", 2])
    def test_get_guided_back_propagation_image_label_value_fail(self,
                                                                image_label,
                                                                tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_guided_back_propagation(image_file,
                                                image_label,
                                                (224, 224))

    @pytest.mark.parametrize("input_shape",
                             [(1,), [1], (1, 2, 3, 4), [1, 2, 3, 4]])
    def test_get_guided_back_propagation_input_shape_value_fail(self,
                                                                input_shape,
                                                                tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_guided_back_propagation(image_file, 0, input_shape)

    @pytest.mark.parametrize("input_shape", [1, 1.5, "random"])
    def test_get_guided_back_propagation_input_shape_type_fail(self,
                                                               input_shape,
                                                               tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_guided_back_propagation(image_file, 0, input_shape)

    @pytest.mark.parametrize("transforms",
                             [1, 1.5, "random value", [1, 2, 3]])
    def test_get_guided_back_propagation_transforms_fail(self,
                                                         transforms,
                                                         tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_guided_back_propagation(image_file,
                                                0,
                                                (224, 224),
                                                transforms)

    def test_get_saliency_map_image_label_type_fail(self, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_saliency_map(image_file, 1.5, (224, 224))

    @pytest.mark.parametrize("image_label", ["Tiger", 2])
    def test_get_saliency_map_image_label_value_fail(self,
                                                     image_label,
                                                     tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_saliency_map(image_file, image_label, (224, 224))

    @pytest.mark.parametrize("input_shape",
                             [(1,), [1], (1, 2, 3, 4), [1, 2, 3, 4]])
    def test_get_saliency_map_input_shape_value_fail(self,
                                                     input_shape,
                                                     tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(ValueError):
            cls.obj.get_saliency_map(image_file, 0, input_shape)

    @pytest.mark.parametrize("input_shape", [1, 1.5, "random"])
    def test_get_saliency_map_input_shape_type_fail(self,
                                                    input_shape,
                                                    tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_saliency_map(image_file, 0, input_shape)

    @pytest.mark.parametrize("transforms",
                             [1, 1.5, "random value", [1, 2, 3]])
    def test_get_saliency_map_transforms_fail(self, transforms, tmp_path):
        cls = self.__class__
        image_file = tmp_path / "sample.png"
        with pytest.raises(TypeError):
            cls.obj.get_saliency_map(image_file, 0, (224, 224), transforms)
