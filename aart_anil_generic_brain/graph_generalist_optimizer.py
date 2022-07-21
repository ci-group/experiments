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
from revolve2.runners.isaacgym import LocalRunner
from dof_map_brain import DofMapBrain
from revolve2.core.optimization import Process
from dataclasses import dataclass
from typing import Optional, Dict
from revolve2.actor_controllers.cpg import CpgNetworkStructure
import numpy as np
import sqlalchemy
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
from revolve2.core.database.serializers import DbNdarray1xn, Ndarray1xnSerializer


@dataclass
class Environment:
    body: Body
    dof_map: Dict[int, int]


@dataclass
class GraphNode:
    environment: Environment
    genotype: npt.NDArray[np.float_]  # Nx1 array
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
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=headless)

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

        for node in self._graph_nodes:
            weight_matrix = self._cpg_network_structure.make_weight_matrix_from_params(
                node.genotype
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
            brain = DofMapBrain(inner_brain, node.environment.dof_map)
            robot = ModularRobot(node.environment.body, brain)
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
                genotype_ids = await Ndarray1xnSerializer.to_database(
                    session, [node.genotype for node in self._graph_nodes]
                )
                dbnodes = [
                    DbGraphGeneralistOptimizerGraphNodeState(
                        process_id=self._process_id,
                        gen_num=self._generation_index,
                        graph_index=i,
                        genotype=genotype_id,
                        fitness=node.fitness,
                    )
                    for i, (node, genotype_id) in enumerate(
                        zip(self._graph_nodes, genotype_ids)
                    )
                ]
                session.add_all(dbnodes)

    def _get_new_genotype(self, node: GraphNode) -> npt.NDArray[np.float_]:  # Nx1 array
        choice = self._rng.randrange(0, 2)

        if choice == 0:  # innovate
            nprng = np.random.Generator(
                np.random.PCG64(self._rng.randint(0, 2**63))
            )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
            permutation = nprng.standard_normal(len(node.genotype))
            return node.genotype + permutation
        else:  # migrate
            neighbours_index = self._rng.randrange(0, len(node.edges))
            chosen_neighbour = node.edges[neighbours_index]
            return chosen_neighbour.genotype


DbBase = declarative_base()


class DbGraphGeneralistOptimizerGraphNodeState(DbBase):
    __tablename__ = "graph_generalist_optimizer_population"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    graph_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    genotype = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
