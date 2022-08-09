from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment as PhysicsEnv,
    PosedActor,
    Runner,
)
from revolve2.runners.isaacgym import LocalRunner
from dataclasses import dataclass
import math
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from dof_map_brain import DofMapBrain
from revolve2.core.modular_robot import Body, ModularRobot
from pyrr import Vector3, Quaternion
from typing import List, Dict
import numpy.typing as npt
import numpy as np


@dataclass
class Environment:
    body: Body
    dof_map: Dict[int, int]


@dataclass
class Setting:
    environment: Environment
    genotype: npt.NDArray[np.float_]  # Nx1 array


class Evaluator:
    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 60

    _cpg_network_structure: CpgNetworkStructure

    _runner: Runner
    _controllers: List[ActorController]

    def __init__(self, cpg_network_structure: CpgNetworkStructure) -> None:
        self._cpg_network_structure = cpg_network_structure
        self._runner = LocalRunner(headless=True)

    async def evaluate(self, settings: List[Setting]) -> List[float]:
        batch = Batch(
            simulation_time=self.SIMULATION_TIME,
            sampling_frequency=self.SAMPLING_FREQUENCY,
            control_frequency=self.CONTROL_FREQUENCY,
            control=self._control,
        )

        for setting in settings:
            weight_matrix = (
                self._cpg_network_structure.make_connection_weights_matrix_from_params(
                    np.clip(setting.genotype, 0.0, 1.0) * 4.0 - 2.0
                )
            )
            initial_state = self._cpg_network_structure.make_uniform_state(
                math.sqrt(2) / 2.0
            )
            dof_ranges = self._cpg_network_structure.make_uniform_dof_ranges(1.0)

            inner_brain = BrainCpgNetworkStatic(
                initial_state,
                self._cpg_network_structure.num_cpgs,
                weight_matrix,
                dof_ranges,
            )
            brain = DofMapBrain(inner_brain, setting.environment.dof_map)
            actor, controller = ModularRobot(
                setting.environment.body, brain
            ).make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            self._controllers.append(controller)
            env = Environment()
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            batch.environments.append(env)

        batch_results = await self._runner.run_batch(batch)

        fitnesses = [
            self._calculate_fitness(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0],
            )
            for environment_result in batch_results.environment_results
        ]
        return fitnesses

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # distance traveled on the xy plane
        return math.sqrt(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )

    def _control(
        self, environment_index: int, dt: float, control: ActorControl
    ) -> None:
        controller = self._controllers[environment_index]
        controller.step(dt)
        control.set_dof_targets(0, controller.get_dof_targets())
