"""
TODO descr
"""

from __future__ import annotations
from random import Random
from typing import List, Dict

import numpy as np
import numpy.typing as npt
from sqlalchemy import null
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession

from revolve2.core.optimization import ProcessIdGen
from revolve2.runners.isaacgym import LocalRunner
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
from evaluator import Evaluator, Setting as EvaluationSetting, Environment as EvalEnv
from revolve2.core.modular_robot import Body


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
    _num_evaluations: int
    _graph_nodes: List[GraphNode]
    _generation_index: int

    _evaluator: Evaluator

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    """Innovation learning rate"""
    _learning_rate: float

    _rng: Random

    _cpg_network_structure: CpgNetworkStructure

    _database: AsyncEngine

    _process_id: int

    _evaluation_count: int

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        num_evaluations: int,
        graph_nodes: List[GraphNode],
        cpg_network_structure: CpgNetworkStructure,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        learning_rate: float,
        headless: bool,
    ) -> None:
        self._rng = rng
        self._num_evaluations = num_evaluations
        self._graph_nodes = graph_nodes
        self._generation_index = 0
        self._cpg_network_structure = cpg_network_structure
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._learning_rate = learning_rate
        self._database = database
        self._process_id = process_id

        self._evaluation_count = 0

        self._evaluator = Evaluator(cpg_network_structure)

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
                [
                    Setting(node.environment, node.genotype.genotype)
                    for node in self._graph_nodes
                ]
            )
            self._evaluation_count += len(self._graph_nodes)
            for node, fitness in zip(self._graph_nodes, fitnesses):
                node.fitness = fitness
            await self._save_generation()
            self._generation_index += 1

        while self._evaluation_count < self._num_evaluations:
            print(f"Gen: {self._generation_index} Evals: {self._evaluation_count}")

            possible_new_genotypes = [
                self._get_new_genotype(node) for node in self._graph_nodes
            ]
            fitnesses = await self._evaluate(
                [
                    Setting(node.environment, possible_new_genotype.genotype)
                    for node, possible_new_genotype in zip(
                        self._graph_nodes, possible_new_genotypes
                    )
                ]
            )
            self._evaluation_count += len(possible_new_genotypes)
            for node, fitness, possible_new_genotype in zip(
                self._graph_nodes, fitnesses, possible_new_genotypes
            ):
                if fitness > node.fitness:
                    node.genotype = possible_new_genotype
                    node.fitness = fitness
            await self._save_generation()
            self._generation_index += 1

    async def _evaluate(self, settings: List[Setting]) -> List[float]:
        evalsets = [
            EvaluationSetting(
                EvalEnv(setting.environment.body, setting.environment.dof_map),
                setting.genotype,
            )
            for setting in settings
        ]
        fitnesses = await self._evaluator.evaluate(evalsets)
        return fitnesses

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
            return Genotype(
                node.genotype.genotype + self._learning_rate * permutation, None
            )
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
