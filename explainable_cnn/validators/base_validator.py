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