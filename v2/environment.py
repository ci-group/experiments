from revolve2.core.modular_robot import Body
from typing import Dict
from revolve2.core.physics import Terrain
from dataclasses import dataclass
from environment_name import EnvironmentName


@dataclass
class Environment:
    body: Body
    dof_map: Dict[int, int]
    terrain: Terrain
    name: EnvironmentName
