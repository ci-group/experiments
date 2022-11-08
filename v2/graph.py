from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class Node:
    index: int
    neighbours: List[Node] = field(default_factory=list)


@dataclass
class Graph:
    nodes: List[Node]
