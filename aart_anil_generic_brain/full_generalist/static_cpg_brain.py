"""
CPG brain.
Active hinges are connected if they are within 2 jumps in the modular robot tree structure.
That means, NOT grid coordinates, but tree distance.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from revolve2.actor_controllers.cpg import Cpg as ControllerCpg

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brain
import numpy as np
import numpy.typing as npt


class StaticCpgBrain(Brain, ABC):
    _initial_state: npt.NDArray[np.float_]
    _num_output_neurons: int
    _weight_matrix: npt.NDArray[np.float_]
    _dof_ranges: npt.NDArray[np.float_]

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        num_output_neurons: int,
        weight_matrix: npt.NDArray[np.float_],
        dof_ranges: npt.NDArray[np.float_],
    ) -> None:
        self._initial_state = initial_state
        self._num_output_neurons = num_output_neurons
        self._weight_matrix = weight_matrix
        self._dof_ranges = dof_ranges

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        return ControllerCpg(
            self._initial_state,
            self._num_output_neurons,
            self._weight_matrix,
            self._dof_ranges,
        )
