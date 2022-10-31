from revolve2.actor_controller import ActorController
from revolve2.serialization import StaticData, Serializable
from typing import List


class DofMapController(ActorController):
    _inner_controller: ActorController
    _output_map: List[int]

    def __init__(
        self, inner_controller: ActorController, output_map: List[int]
    ) -> None:
        self._inner_controller = inner_controller
        self._output_map = output_map

        # verify all indices in map are valid
        targets_len = len(self._inner_controller.get_dof_targets())
        assert all(
            [index < targets_len for index in self._output_map]
        ), "output_map not compatible with dof targets provided by inner_controller."

    def step(self, dt: float) -> None:
        self._inner_controller.step(dt)

    def get_dof_targets(self) -> List[float]:
        targets = self._inner_controller.get_dof_targets()
        return [targets[index] for index in self._output_map]

    def serialize(self) -> StaticData:
        raise NotImplementedError()

    @classmethod
    def deserialize(cls, data: StaticData) -> Serializable:
        raise NotImplementedError()
