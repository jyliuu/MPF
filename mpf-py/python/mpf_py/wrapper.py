from abc import ABC

from abc import ABC

class PythonWrapperClassBase(ABC):
    def __getattribute__(self, name):
        try:
            # Try to get the attribute from the wrapper instance itself
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, try to get it from the Rust instance
            try:
                rust_instance = object.__getattribute__(self, '_rust_instance')
                value = getattr(rust_instance, name)
                # Cache the value in the wrapper instance
                setattr(self, name, value)
                return value
            except AttributeError:
                # If still not found, raise the default AttributeError
                raise

    @classmethod
    def __init_subclass__(cls, RustClass=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if RustClass is None:
            raise TypeError("RustClass must be specified when subclassing PythonWrapperClassBase")
