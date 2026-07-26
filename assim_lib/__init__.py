"""Flow-matching data assimilation package."""

from typing import TYPE_CHECKING

from .config import TrainingConfig

__all__ = ["Sampler", "TrainingConfig"]

if TYPE_CHECKING:
    from .sampler import Sampler


def __getattr__(name: str):
    if name == "Sampler":
        from .sampler import Sampler

        return Sampler
    raise AttributeError(name)
