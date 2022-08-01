"""
TODO descr
"""

from __future__ import annotations
import math
from random import Random
from typing import List, Dict

import numpy as np
import numpy.typing as npt
from pyrr import Quaternion, Vector3
from sqlalchemy import null
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkStatic
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    PosedActor,
    Runner,
    Environment as PhysicsEnv,
)
from revolve2.runners.mujoco import LocalRunner
from dof_map_brain import DofMapBrain
from revolve2.core.optimization import Process
from dataclasses import dataclass
from typing import Optional, Dict
from revolve2.actor_controllers.cpg import CpgNetworkStructure
import numpy as np
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from revolve2.core.database.serializers import DbNdarray1xn, Ndarray1xnSerializer


@dataclass
class Genotype:
    genotype: npt.NDArray[np.float_]  # Nx1 array
    db_id: Optional[int]


@dataclass
class Environment:
    body: Body
    dof_map: Dict[int, int]


@dataclass
class GraphNode:
    environment: Environment
    genotype: Genotype
    edges: List[GraphNode]
    fitness: Optional[float]


@dataclass
class Setting:
    environment: Environment
    genotype: npt.NDArray[np.float_]  # Nx1 array


class GraphGeneralistOptimizer(Process):
    _num_generations: int
    _graph_nodes: List[GraphNode]
    _generation_index: int

    _runner: Runner
    _controllers: List[ActorController]

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _rng: Random

    _cpg_network_structure: CpgNetworkStructure

    _database: AsyncEngine

    _process_id: int

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        num_generations: int,
        graph_nodes: List[GraphNode],
        cpg_network_structure: CpgNetworkStructure,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        headless: bool,
    ) -> None:
        self._rng = rng
        self._num_generations = num_generations
        self._graph_nodes = graph_nodes
        self._generation_index = 0
        self._cpg_network_structure = cpg_network_structure
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._database = database
        self._process_id = process_id

        self._init_runner(headless)

        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await Ndarray1xnSerializer.create_tables(session)

    async def ainit_from_database(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
    ) -> bool:
        return False

    def _init_runner(self, headless: bool) -> None:
        self._runner = LocalRunner(headless=headless)

    async def run(self) -> None:
        if self._generation_index == 0:
            fitnesses = await self._evaluate(
                [Setting(node.environment, node.genotype) for node in self._graph_nodes]
            )
            for node, fitness in zip(self._graph_nodes, fitnesses):
                node.fitness = fitness
            await self._save_generation()
            self._generation_index += 1

        while self._generation_index < self._num_generations:
            possible_new_genotypes = [
                self._get_new_genotype(node) for node in self._graph_nodes
            ]
            fitnesses = await self._evaluate(
                [
                    Setting(node.environment, possible_new_genotype)
                    for node, possible_new_genotype in zip(
                        self._graph_nodes, possible_new_genotypes
                    )
                ]
            )
            for node, fitness, possible_new_genotype in zip(
                self._graph_nodes, fitnesses, possible_new_genotypes
            ):
                if fitness > node.fitness:
                    node.genotype = possible_new_genotype
                    node.fitness = fitness
            await self._save_generation()
            self._generation_index += 1

    async def _evaluate(self, settings: List[Setting]) -> List[float]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        self._controllers = []

        for setting in settings:
            weight_matrix = (
                self._cpg_network_structure.make_connection_weights_matrix_from_params(
                    setting.genotype.genotype
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
            robot = ModularRobot(setting.environment.body, brain)
            actor, controller = robot.make_actor_and_controller()
            bounding_box = actor.calc_aabb()
            self._controllers.append(controller)
            env = PhysicsEnv()
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

    def _control(
        self, environment_index: int, dt: float, control: ActorControl
    ) -> None:
        controller = self._controllers[environment_index]
        controller.step(dt)
        control.set_dof_targets(0, controller.get_dof_targets())

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # distance traveled on the xy plane
        return math.sqrt(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )

    async def _save_generation(self) -> None:
        async with AsyncSession(self._database) as session:
            async with session.begin():
                new_genotypes = [
                    node.genotype
                    for node in self._graph_nodes
                    if node.genotype.db_id is None
                ]

                genotype_ids = await Ndarray1xnSerializer.to_database(
                    session, [node.genotype for node in new_genotypes]
                )

                dbgenotypes = [
                    DbGenotype(nparray1xn_id=id)
                    for gen, id in zip(new_genotypes, genotype_ids)
                ]
                session.add_all(dbgenotypes)
                await session.flush()
                new_genotype_ids = [
                    gen.id for gen in dbgenotypes if gen.id is not None
                ]  # cannot be none because not nullable but adding check for mypy
                assert len(new_genotype_ids) == len(
                    dbgenotypes
                )  # just to be sure because of check above
                for new_gen, new_id in zip(new_genotypes, new_genotype_ids):
                    new_gen.db_id = int(new_id)

                dbnodes = [
                    DbGraphGeneralistOptimizerGraphNodeState(
                        process_id=self._process_id,
                        gen_num=self._generation_index,
                        graph_index=i,
                        genotype_id=node.genotype.db_id,
                        fitness=node.fitness,
                    )
                    for i, node in enumerate(self._graph_nodes)
                ]
                session.add_all(dbnodes)

    def _get_new_genotype(self, node: GraphNode) -> Genotype:
        choice = self._rng.randrange(0, 2)

        if choice == 0:  # innovate
            nprng = np.random.Generator(
                np.random.PCG64(self._rng.randint(0, 2**63))
            )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
            permutation = nprng.standard_normal(len(node.genotype.genotype))
            return Genotype(node.genotype.genotype + permutation, None)
        else:  # migrate
            neighbours_index = self._rng.randrange(0, len(node.edges))
            chosen_neighbour = node.edges[neighbours_index]
            return chosen_neighbour.genotype


DbBase = declarative_base()


class DbGenotype(DbBase):
    __tablename__ = "graph_generalist_genotype"
    id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    nparray1xn_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )


class DbGraphGeneralistOptimizerGraphNodeState(DbBase):
    __tablename__ = "graph_generalist_optimizer_population"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    graph_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    genotype_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbGenotype.id), nullable=False
    )
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
