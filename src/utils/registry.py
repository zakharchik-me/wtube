import os
import importlib


REGISTRY = {
    "source": {},
    "buffer": {},
    "preprocess": {},
    "inference": {},
    "postprocess": {},
    "detector": {},
    "tracker": {}
}

def register(kind):
    def decorator(cls):
        name = cls.__name__
        REGISTRY[kind][name] = cls
        return cls
    return decorator

def build(kind, name, **kwargs):
    try:
        cls = REGISTRY[kind][name]
        return cls(**kwargs)
    except KeyError:
        available = list(REGISTRY[kind].keys())
        raise KeyError(f"{name!r} not found in REGISTRY[{kind!r}]. Available: {available}")

def import_all_from_pipeline():
    # registry.py is at Wtube/src/utils/registry.py
    # plugins are scattered under Wtube/src/wtube/ (in various subfolders)
    base_dir = os.path.join(os.path.dirname(__file__), "..", "wtube")
    base_module = "wtube"

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".py") or file.startswith("__"):
                continue

            # Compute path relative to src/wtube/
            rel_path = os.path.relpath(root, base_dir).replace(os.path.sep, ".")
            module_name = file[:-3]  # strip “.py”

            if rel_path == ".":
                full_module = f"{base_module}.{module_name}"
            else:
                full_module = f"{base_module}.{rel_path}.{module_name}"

            try:
                importlib.import_module(full_module)
            except ImportError as e:
                print(f"[registry] Could not import {full_module!r}: {e}")

# def import_all_from_pipeline():
#     base_dir = os.path.join(os.path.dirname(__file__), "..", "wtube")
#     base_module = "wtube"
#
#     for root, _, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith(".py") and not file.startswith("__"):
#                 rel_path = os.path.relpath(root, base_dir).replace(os.path.sep, ".")
#                 module_name = file[:-3]
#                 full_module = f"{base_module}.{rel_path}.{module_name}" if rel_path != "." else f"{base_module}.{module_name}"
#                 importlib.import_module(full_module)

def wrap_validator_call(validator_fn, use_validator, source_name):
    class Wrapper:
        def __init__(self):
            self._valid = False

        def __call__(self, *args, **kwargs):
            if not use_validator:
                return
            result = validator_fn(*args, **kwargs)
            self._valid = True
            return result

        @property
        def valid(self):
            return self._valid

    return Wrapper()

