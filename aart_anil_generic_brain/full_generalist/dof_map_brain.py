from abc import ABC
from typing import List, Dict

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, Brain
from dof_map_controller import DofMapController


class DofMapBrain(Brain, ABC):
    _inner_brain: Brain
    _dof_map: Dict[int, int]

    def __init__(self, inner_brain: Brain, dof_map: Dict[int, int]) -> None:
        """
        :inner_brain: The brain that creates the inner controller.
        :output_map: Module id to inner controller dof index
        """

        self._inner_brain = inner_brain
        self._dof_map = dof_map

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        inner_controller = self._inner_brain.make_controller(
            body, dof_ids
        )  # most inner brains will ignore these parameters but pass them anyway
        assert len(self._dof_map) == len(dof_ids)
        assert all(dof_id in self._dof_map.keys() for dof_id in dof_ids)
        assert all(index < len(dof_ids) for index in self._dof_map.values())

        output_map = [self._dof_map[mod_id] for mod_id in dof_ids]

        return DofMapController(inner_controller, output_map)
