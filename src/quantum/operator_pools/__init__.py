"""Small registry for local operator pool builders."""

from .polaron_paop import make_pool

__all__ = ["make_pool"]
