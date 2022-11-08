"""Optimize a neural network for solving XOR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from revolve2.core.database import (
    SerializableIncrementableStruct,
)
from revolve2.core.database.std import Rng
from revolve2.core.optimization.ea.algorithms import de_offspring, bounce_parameters
from revolve2.core.optimization.ea.population import (
    Individual,
    Parameters,
    SerializableMeasures,
)
from revolve2.core.optimization.ea.population.pop_list import PopList, replace_if_better
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from evaluator import Evaluator, Setting as EvaluationSetting, Environment as EvalEnv
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from revolve2.core.modular_robot import Body
from graph import Graph, Node

Genotype = Parameters


@dataclass
class Measures(SerializableMeasures, table_name="measures"):
    """Measures of a genotype/phenotype."""

    fitness: Optional[float] = None


class Population(PopList[Genotype, Measures], table_name="population"):
    """A population of individuals consisting of the above Genotype and Measures."""

    pass


@dataclass
class ProgramState(
    SerializableIncrementableStruct,
    table_name="program_state",
):
    """State of the program."""

    rng: Rng
    population: Population
    generation_index: int


@dataclass
class Environment:
    body: Body
    dof_map: Dict[int, int]


@dataclass
class Setting:
    environment: Environment
    genotype: Genotype


class GraphOptimizer:
    """Program that optimizes the neural network parameters."""

    num_evaluations: int
    standard_deviation: float

    bodies: List[Body]
    dof_maps: List[Dict[int, int]]
    graph: Graph

    db: AsyncEngine
    evaluator: Evaluator

    state: ProgramState

    async def run(
        self,
        rng: Rng,
        database: AsyncEngine,
        robot_bodies: List[Body],
        dof_maps: List[Dict[int, int]],
        graph: Graph,
        cpg_network_structure: CpgNetworkStructure,
        headless: bool,
        num_evaluations: int,
        standard_deviation: float,
    ) -> None:
        """Run the program."""
        self.db = database
        self.evaluator = Evaluator(cpg_network_structure, headless=headless)
        self.bodies = robot_bodies
        self.dof_maps = dof_maps
        self.graph = graph
        self.num_evaluations = num_evaluations
        self.standard_deviation = standard_deviation

        async with self.db.begin() as conn:
            await ProgramState.prepare_db(conn)

        if not await self.load_state():
            initial_genotypes = [
                Genotype(
                    [
                        float(v)
                        for v in rng.rng.random(
                            size=cpg_network_structure.num_connections
                        )
                    ]
                )
                for _ in range(len(self.graph.nodes))
            ]

            settings = [
                Setting(Environment(body, dof_map), genotype)
                for body, dof_map, genotype in zip(
                    self.bodies, self.dof_maps, initial_genotypes
                )
            ]
            all_measures = await self.measure(settings)

            initial_population = [
                Individual(genotype, measures)
                for genotype, measures in zip(initial_genotypes, all_measures)
            ]

            initial_generation_index = 0

            self.state = ProgramState(
                rng=rng,
                population=initial_population,
                generation_index=initial_generation_index,
            )

            await self.save_state()

        # TODO eval count
        while (self.state.generation_index + 1) * len(self.graph.nodes) * len(
            self.bodies
        ) < self.num_evaluations:
            await self.evolve()
            await self.save_state()

    async def save_state(self) -> None:
        """Save the state of the program."""
        async with AsyncSession(self.db) as ses:
            async with ses.begin():
                await self.state.to_db(ses)

    async def load_state(self) -> bool:
        """
        Load the state of the program.

        :returns: True if could be loaded from database. False if no data available.
        """
        async with AsyncSession(self.db) as ses:
            async with ses.begin():
                maybe_state = await ProgramState.from_db_latest(ses, 1)
                if maybe_state is None:
                    return False
                else:
                    self.state = maybe_state
                    return True

    async def evolve(self) -> None:
        """Iterate one generation further."""
        self.state.generation_index += 1

        possible_new_genotypes = [
            self.get_new_genotype(
                node, self.state.population, self.state.rng, self.standard_deviation
            )
            for node in self.graph.nodes
        ]

        settings = [
            Setting(Environment(body, dof_map), possible_new_genotype)
            for body, dof_map, possible_new_genotype in zip(
                self.bodies, self.dof_maps, possible_new_genotypes
            )
        ]

        possible_new_individuals = Population(
            [
                Individual(g, m)
                for g, m in zip(possible_new_genotypes, await self.measure(settings))
            ]
        )

        original_selection, offspring_selection = replace_if_better(
            self.state.population, possible_new_individuals, measure="fitness"
        )

        self.state.population = Population.from_existing_populations(  # type: ignore # TODO
            [self.state.population, possible_new_individuals],
            [original_selection, offspring_selection],
            [
                "fitness",
            ],
        )

    @staticmethod
    def get_new_genotype(
        node: Node, pop: Population, rng: Rng, standard_deviation: float
    ) -> Genotype:
        choice = rng.rng.integers(0, 2)

        if choice == 0:  # innovate
            permutation = standard_deviation * rng.rng.standard_normal(
                len(pop[node.index].genotype)
            )
            return bounce_parameters(Genotype(pop[node.index].genotype + permutation))
        else:  # migrate
            neighbours_index = rng.rng.integers(0, len(node.neighbours))
            chosen_neighbour = node.neighbours[neighbours_index]
            return pop[chosen_neighbour.index].genotype

    async def measure(self, settings: List[Setting]) -> List[Measures]:
        """
        Measure all provided genotypes.

        :param pop: The genotypes.
        :returns: Measures for the genotypes.
        """
        evalsets: List[EvaluationSetting] = []

        evalsets = [
            EvaluationSetting(
                EvalEnv(setting.environment.body, setting.environment.dof_map),
                setting.genotype,
            )
            for setting in settings
        ]
        fitnesses = await self.evaluator.evaluate(evalsets)
        return [Measures(fitness=fitness) for fitness in fitnesses]
