from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass(repr=False)
class Node:
    index: int
    neighbours: List[Node] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"{{index: {self.index}, neighbours: [{','.join([str(n.index) for n in self.neighbours])}]}}"


@dataclass
class Graph:
    nodes: List[Node]
