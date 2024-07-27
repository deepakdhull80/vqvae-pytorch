from enum import Enum


class ModelEnum(Enum):
    AUTO_ENCODER: str = "auto-encoder"
    VARIATIONAL_AUTO_ENCODER: str = "variational-auto-encoder"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
