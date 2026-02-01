"""Centralised registry of models."""

from typing import Dict, Type


class ModelZoo:
    """Contains a directory of all available models.

    Thanks to @atenrev because I shamelessly copied this construct from him.
    """

    _models: Dict[str, Type] = {}
    _configs: Dict[str, Type] = {}

    @classmethod
    def register_model(cls, cl) -> None:
        """Register a model in the Zoo alongside its configuration."""
        assert (
            cl.__name__ not in cls._models
        ), "Overriding an existing model in the Model Zoo"

        cls._models[cl.__name__] = cl
        cls._configs[cl.__name__] = cl.MODEL_CONFIG
        return cl

    @classmethod
    def get_model(cls, name) -> Type:
        """Return the model type for a given name."""
        return cls._models[name]

    @classmethod
    def get_config(cls, name) -> Type:
        """Return the model configuration type for a given name."""
        return cls._configs[name]
