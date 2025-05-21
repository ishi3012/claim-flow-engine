# claimflowengine/utils/functions.py

from typing import Any


def convert_to_int(x: Any) -> Any:
    return x.astype(int) if hasattr(x, "astype") else x
