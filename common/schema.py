"""The class-name and frequency mapping shared by preparation, training and serving."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .utils import read_json, write_json


@dataclass
class Schema:
    """A single multiclass target: each report has exactly one integer label."""

    classes: list[str]
    counts: list[int] = field(default_factory=list)
    total_samples: int = 0
    dataset: str = ""

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Schema":
        if "heads" in data:
            raise ValueError(
                "This project now uses one integer class label per report. "
                "Regenerate the prepared data with `python data_prep/prepare_asrs.py`."
            )
        return cls(**data)

    def save(self, path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path) -> "Schema":
        return cls.from_dict(read_json(path))
