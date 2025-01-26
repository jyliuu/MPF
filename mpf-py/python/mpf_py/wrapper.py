from abc import ABC


class PythonWrapperClassBase(ABC):
    __slots__ = ["_rust_instance"]

    def _get_property_value(self, property_name):
        """
        Helper method to get or snapshot a property value on-demand.
        """
        private_attr_name = "_" + property_name
        if not hasattr(self, private_attr_name):
            rust_property_value = getattr(self._rust_instance, property_name)
            setattr(self, private_attr_name, rust_property_value)
        return getattr(self, private_attr_name) # Return snapshotted value


    @classmethod
    def __init_subclass__(cls, RustClass=None, **kwargs):
        """
        Dynamically creates properties and passthrough methods on subclasses.

        Args:
            RustClass: The PyO3 Rust class to wrap. Must be provided when subclassing.
        """
        super().__init_subclass__(**kwargs)

        if RustClass is None:
            raise TypeError("RustClass must be specified as a keyword argument when subclassing PythonWrapperClassBase.")

        for attr_name in dir(RustClass):
            if attr_name.startswith("__"):
                continue
            attr = getattr(RustClass, attr_name)
            if not callable(attr):
                # Create dynamic Python property for Rust properties
                property_wrapper = property(
                    lambda instance_self, name=attr_name: instance_self._get_property_value(name)
                ) # Create Python property
                setattr(cls, attr_name, property_wrapper) # Set property on Python wrapper class
