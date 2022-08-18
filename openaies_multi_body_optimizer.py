"""
Optimize multiple bodies at the same time using the same cpg controller mapped uising dof map controller
Fitness unit is somewhat m/s, calculated using square(avg(sqrt(fitnesses)))
"""

from random import Random
from typing import List, Dict

import numpy as np
import numpy.typing as npt
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.optimization.ea.openai_es import OpenaiESOptimizer
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from evaluator import Evaluator, Setting as EvaluationSetting, Environment as EvalEnv


class OpenaiESMultiBodyOptimizer(OpenaiESOptimizer):
    _bodies: List[Body]
    _dof_maps: List[Dict[int, int]]

    _evaluator: Evaluator
    _controllers: List[ActorController]

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_evaluations: int

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
        num_evaluations: int,
        cpg_network_structure: CpgNetworkStructure,
        headless: bool,
    ) -> None:
        nprng = np.random.Generator(
            np.random.PCG64(rng.randint(0, 2**63))
        )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
        initial_mean = nprng.uniform(
            size=cpg_network_structure.num_connections,
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

        self._evaluator = Evaluator(cpg_network_structure)

        self._bodies = robot_bodies
        self._dof_maps = dof_maps
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_evaluations = num_evaluations
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
        num_evaluations: int,
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

        self._evaluator = Evaluator(cpg_network_structure)

        self._bodies = robot_bodies
        self._dof_maps = dof_maps
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_evaluations = num_evaluations
        self._cpg_network_structure = cpg_network_structure

        return True

    async def _evaluate_population(
        self,
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        evalsets: List[EvaluationSetting] = []

        for body, dof_map in zip(self._bodies, self._dof_maps):
            for params in population:
                evalsets.append(EvaluationSetting(EvalEnv(body, dof_map), params))

        fitnesses = np.array(await self._evaluator.evaluate(evalsets))

        fitnesses.resize(len(self._bodies), len(population))
        return np.average(np.sqrt(fitnesses), axis=0) ** 2

    def _must_do_next_gen(self) -> bool:
        return (
            self._population_size * len(self._bodies) * self.generation_number
            < self._num_evaluations
        )
