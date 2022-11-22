from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment as PhysicsEnv,
    PosedActor,
    Runner,
)
from revolve2.runners.mujoco import LocalRunner
from dataclasses import dataclass
import math
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from dof_map_brain import DofMapBrain
from revolve2.core.modular_robot import ModularRobot
from pyrr import Vector3, Quaternion
from typing import List
import numpy as np
from experiment_settings import (
    SIMULATION_TIME,
    SAMPLING_FREQUENCY,
    CONTROL_FREQUENCY,
    ROBOT_INITIAL_Z_OFFSET,
)
from environment import Environment
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)
from revolve2.core.optimization.ea.population import Parameters


@dataclass
class EvaluationDescription:
    environment: Environment
    genotype: Parameters


class Evaluator:
    _cpg_network_structure: CpgNetworkStructure

    _runner: Runner

    def __init__(
        self, cpg_network_structure: CpgNetworkStructure, headless: bool
    ) -> None:
        self._cpg_network_structure = cpg_network_structure
        self._runner = LocalRunner(headless=headless, num_simulators=1)

    async def evaluate(self, eval_descrs: List[EvaluationDescription]) -> List[float]:
        batch = Batch(
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
        )

        for eval_descr in eval_descrs:
            weight_matrix = (
                self._cpg_network_structure.make_connection_weights_matrix_from_params(
                    np.clip(eval_descr.genotype, 0.0, 1.0) * 4.0 - 2.0
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
            brain = DofMapBrain(inner_brain, eval_descr.environment.dof_map)
            actor, controller = ModularRobot(
                eval_descr.environment.body, brain
            ).make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            env = PhysicsEnv(EnvironmentActorController(controller))
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0
                            - bounding_box.offset.z
                            + ROBOT_INITIAL_Z_OFFSET,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            env.static_geometries.extend(eval_descr.environment.terrain.static_geometry)
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
