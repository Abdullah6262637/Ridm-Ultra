"""Quantization ve FastAPI çıkarım servisi."""
from .quantize import load_quantized_model, quantize_checkpoint_dynamic

__all__ = ["quantize_checkpoint_dynamic", "load_quantized_model", "create_app"]


def __getattr__(name):
    # `create_app`, fastapi/pydantic yalnızca gerçekten servis başlatılırken
    # zorunlu olsun diye tembel (lazy) içe aktarılır; böylece `import
    # ridm_ultra.llm.serving`, bu paketler kurulu değilken de çalışır.
    if name == "create_app":
        from .api import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
