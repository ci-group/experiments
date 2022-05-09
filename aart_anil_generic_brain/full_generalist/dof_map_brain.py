from abc import ABC
from typing import List

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, Brain
from dof_map_controller import DofMapController


class DofMapBrain(Brain, ABC):
    _inner_brain: Brain
    _output_map: List[int]

    def __init__(self, inner_brain: Brain, output_map: List[int]) -> None:
        """
        :inner_brain: The brain that creates the inner controller.
        :output_map: Map from dof to the dof in the inner controller.
        """

        self._inner_brain = inner_brain
        self._output_map = output_map

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        inner_controller = self._inner_brain.make_controller(
            body, dof_ids
        )  # most inner brains will ignore these parameters but pass them anyway
        return DofMapController(inner_controller, self._output_map)
