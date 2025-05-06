from dataclasses import dataclass
from typing import Optional

@dataclass
class Denial:
    reason: str
    code: Optional[str] = None
    severity: int = 3
    appealability_score: float = 0.0
    auto_appealable: bool = False
    routing_tag: Optional[str] = None
