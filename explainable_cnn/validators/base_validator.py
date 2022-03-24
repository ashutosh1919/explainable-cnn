class BaseValidator:
    def __init__(self):
        """
        Validates types of different modules used in the package and raises
        error if necessary.
        """
        pass

    def assert_type(self, obj, obj_type):
        """
        Validates valid object type.
        """
        if not isinstance(obj, obj_type):
            raise ValueError(f"obj must be an instance of {obj_type}")

    def assert_label_index_validity(self, class_map, label_index):
        """
        Validates whether such `label_index` exists for model.
        """
        if label_index not in class_map:
            raise ValueError("Given label index does not exist in 'class_map'")

    def assert_label_name_validity(self, reverse_class_map, label_name):
        """
        Validates whether such `label_name` exists for model.
        """
        if label_name not in reverse_class_map:
            raise ValueError(
                "Given label name does not exist in 'reverse_class_map'"
            )
