"""
Optimize multiple bodies at the same time using the same cpg controller mapped uising dof map controller
Fitness unit is somewhat m/s, calculated using square(avg(sqrt(fitnesses)))
"""

import math
from random import Random
from typing import List, Dict

import numpy as np
import numpy.typing as npt
from pyrr import Quaternion, Vector3
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.optimization.ea.openai_es import OpenaiESOptimizer
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from revolve2.runners.isaacgym import LocalRunner
from dof_map_brain import DofMapBrain
from revolve2.actor_controllers.cpg import CpgNetworkStructure


class OpenaiESMultiBodyOptimizer(OpenaiESOptimizer):
    _bodies: List[Body]
    _dof_maps: List[Dict[int, int]]

    _runner: Runner
    _controllers: List[ActorController]

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int

    _cpg_network_structure: CpgNetworkStructure

    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        population_size: int,
        sigma: float,
        learning_rate: float,
        robot_bodies: List[Body],
        dof_maps: List[Dict[int, int]],
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        cpg_network_structure: CpgNetworkStructure,
        headless: bool,
    ) -> None:
        nprng = np.random.Generator(
            np.random.PCG64(rng.randint(0, 2**63))
        )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
        initial_mean = nprng.uniform(
            size=cpg_network_structure.num_params,
            low=0,
            high=1.0,
        )

        await super().ainit_new(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            rng=rng,
            population_size=population_size,
            sigma=sigma,
            learning_rate=learning_rate,
            initial_mean=initial_mean,
        )

        self._init_runner(headless)

        self._bodies = robot_bodies
        self._dof_maps = dof_maps
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._cpg_network_structure = cpg_network_structure

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        robot_bodies: List[Body],
        dof_maps: List[Dict[int, int]],
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        cpg_network_structure: CpgNetworkStructure,
        headless: bool,
    ) -> bool:
        if not await super().ainit_from_database(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            rng=rng,
        ):
            return False

        self._init_runner(headless)

        self._bodies = robot_bodies
        self._dof_maps = dof_maps
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self._cpg_network_structure = cpg_network_structure

        return True

    def _init_runner(self, headless: bool) -> None:
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=headless)

    async def _evaluate_population(
        self,
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        self._controllers = []

        for robot, dof_map in zip(self._bodies, self._dof_maps):
            for params in population:
                weight_matrix = (
                    self._cpg_network_structure.make_weight_matrix_from_params(params)
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
                brain = DofMapBrain(inner_brain, dof_map)
                actor, controller = ModularRobot(
                    robot, brain
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

        fitnesses = np.array(
            [
                self._calculate_fitness(
                    batch_results.environment_results[i]
                    .environment_states[0]
                    .actor_states[0],
                    batch_results.environment_results[i]
                    .environment_states[-1]
                    .actor_states[0],
                )
                for i in range(len(population) * len(self._bodies))
            ]
        )
        fitnesses.resize(len(self._bodies), len(population))
        return np.average(np.sqrt(fitnesses), axis=0) ** 2

    def _control(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # distance traveled on the xy plane
        return math.sqrt(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_number != self._num_generations
