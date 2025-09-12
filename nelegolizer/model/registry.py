import inspect

MODEL_REGISTRY = {}

def register_model(name):
    """Dekorator do rejestrowania modeli pod daną nazwą"""
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def filter_params(cls, params: dict) -> dict:
    """
    Usuwa parametry, których konstruktor klasy `cls` nie przyjmuje.
    Dzięki temu config YAML może zawierać dodatkowe pola,
    które nie spowodują błędu.
    """
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in params.items() if k in valid_keys}


def get_model(config: dict):
    """
    Tworzy model na podstawie wpisu w configu YAML:
    
    model:
      name: LegoNet
      in_channels: 3
      num_classes: 10
    """
    model_name = config["model"]["name"]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]
    params = filter_params(model_cls, config["model"])
    return model_cls(**params)