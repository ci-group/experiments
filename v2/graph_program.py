"""Optimize a neural network for solving XOR."""

from __future__ import annotations

from dataclasses import dataclass
from tkinter import E
from typing import List, Optional

from revolve2.core.database import (
    SerializableIncrementableStruct,
    SerializableFrozenSingleton,
    SerializableFrozenList,
    SerializableStruct,
    SerializableList,
)
from environment_name import EnvironmentName
from revolve2.core.database.std import Rng
from revolve2.core.optimization.ea.algorithms import bounce_parameters
from revolve2.core.optimization.ea.population import (
    Individual,
    Parameters,
    SerializableMeasures,
)
from revolve2.core.optimization.ea.population.pop_list import PopList, replace_if_better
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from evaluator import Evaluator, EvaluationDescription
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from graph import Graph, Node
from environment import Environment
import logging
from bodies import make_cpg_network_structure
from revolve2.core.database import open_async_database_sqlite
import numpy as np

Genotype = Parameters


class VisitedEnvironments(
    SerializableList[int],
    table_name="visited_environments",
    value_column_name="environment",
):
    pass


@dataclass
class GenotypeWithMeta(SerializableStruct, table_name="genotype_with_meta"):
    genotype: Genotype
    visited_environments: VisitedEnvironments


@dataclass
class GenotypeWithHeritage(SerializableStruct, table_name="genotype_with_heritage"):
    genotype: GenotypeWithMeta
    came_from: int


@dataclass
class Measures(SerializableMeasures, table_name="measures"):
    """Measures of a genotype/phenotype."""

    fitness: Optional[float] = None


class Population(PopList[GenotypeWithHeritage, Measures], table_name="population"):
    """A population of individuals consisting of the above Genotype and Measures."""

    pass


class EnvironmentNames(
    SerializableFrozenList[EnvironmentName],
    table_name="environment_names",
    value_column_name="name",
):
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
    performed_evaluations: int


@dataclass
class ProgramRoot(SerializableFrozenSingleton, table_name="program_root"):
    environment_names: EnvironmentNames
    program_state: ProgramState


@dataclass
class Setting:
    environment: Environment
    genotype: Genotype


class Program:
    """Program that optimizes the neural network parameters."""

    num_evaluations: int
    standard_deviation: float
    migration_probability: float

    environments: List[Environment]
    graph: Graph

    db: AsyncEngine
    evaluator: Evaluator

    state: ProgramState

    async def run(
        self,
        database_name: str,
        headless: bool,
        rng_seed: int,
        environments: List[Environment],
        graph: Graph,
        num_evaluations: int,
        standard_deviation: float,
        migration_probability: float,
        num_simulators: int,
    ) -> None:
        """Run the program."""
        logging.info("Program start.")

        self.num_evaluations = num_evaluations
        self.standard_deviation = standard_deviation
        self.migration_probability = migration_probability
        self.environments = environments
        self.graph = graph

        cpg_network_structure = make_cpg_network_structure()
        self.evaluator = Evaluator(
            cpg_network_structure, headless=headless, num_simulators=num_simulators
        )

        logging.info("Opening database..")
        self.database = open_async_database_sqlite(database_name, create=True)
        logging.info("Opening database done.")

        logging.info("Creating database structure..")
        async with self.database.begin() as conn:
            await ProgramRoot.prepare_db(conn)
        logging.info("Creating database structure done.")

        logging.info("Trying to load program root from database..")
        if await self.load_root():
            logging.info("Root loaded successfully.")
        else:
            logging.info("Unable to load root. Initializing root..")
            await self.init_root(
                rng_seed=rng_seed,
                cpg_network_structure=cpg_network_structure,
                environments=environments,
            )
            logging.info("Initializing state done.")

        logging.info(
            f"Entering optimization loop. Continuing until around {self.num_evaluations} evaluations."
        )
        while self.root.program_state.performed_evaluations < self.num_evaluations:
            logging.info(
                f"Current # evaluations: {self.root.program_state.performed_evaluations}"
            )
            logging.info("Evolving..")
            await self.evolve()
            logging.info("Evolving done.")

            logging.info("Saving state..")
            await self.save_state()
            logging.info("Saving state done.")
        logging.info("Optimization loop done. Exiting program.")

    async def save_state(self) -> None:
        """Save the state of the program."""
        async with AsyncSession(self.database) as ses:
            async with ses.begin():
                await self.root.program_state.to_db(ses)

    async def init_root(
        self,
        rng_seed: int,
        cpg_network_structure: CpgNetworkStructure,
        environments: List[Environment],
    ) -> None:
        initial_rng = Rng(np.random.Generator(np.random.PCG64(rng_seed)))

        initial_genotypes = [
            GenotypeWithHeritage(
                genotype=GenotypeWithMeta(
                    Genotype(
                        [
                            float(v)
                            for v in initial_rng.rng.random(
                                size=cpg_network_structure.num_connections
                            )
                        ]
                    ),
                    VisitedEnvironments([env_i]),
                ),
                came_from=env_i,
            )
            for env_i in range(len(self.graph.nodes))
        ]

        eval_descrs = [
            EvaluationDescription(env, genotype.genotype.genotype)
            for env, genotype in zip(self.environments, initial_genotypes)
        ]
        logging.info("Measuring initial population..")
        all_measures = await self.measure(eval_descrs)
        logging.info("Measuring initial population done.")

        initial_population = [
            Individual(genotype, measures)
            for genotype, measures in zip(initial_genotypes, all_measures)
        ]

        logging.info("Saving root..")
        state = ProgramState(
            rng=initial_rng,
            population=initial_population,
            generation_index=0,
            performed_evaluations=len(eval_descrs),
        )
        self.root = ProgramRoot(EnvironmentNames([e.name for e in environments]), state)
        async with AsyncSession(self.database) as ses:
            async with ses.begin():
                await self.root.to_db(ses)
        logging.info("Saving root done.")

    async def load_root(self) -> bool:
        """
        Load the root of the program.

        :returns: True if could be loaded from database. False if no data available.
        """
        async with AsyncSession(self.database) as ses:
            async with ses.begin():
                maybe_root = await ProgramRoot.from_db(ses, 1)
                if maybe_root is None:
                    return False
                else:
                    self.root = maybe_root
                    return True

    async def evolve(self) -> None:
        """Iterate one generation further."""
        self.root.program_state.generation_index += 1

        possible_new_genotypes = [
            self.get_new_genotype(
                node,
                self.root.program_state.population,
                self.root.program_state.rng,
                self.standard_deviation,
                self.migration_probability,
            )
            for node in self.graph.nodes
        ]

        eval_descrs = [
            EvaluationDescription(env, genotype.genotype.genotype)
            for env, genotype in zip(self.environments, possible_new_genotypes)
            if genotype is not None
        ]

        measurements = await self.measure(eval_descrs)

        possible_new_individuals: List[Individual[GenotypeWithHeritage, Measures]] = []
        measurements_index = 0
        for orig_genotype, new_genotype in zip(
            self.root.program_state.population, possible_new_genotypes
        ):
            if new_genotype is None:
                possible_new_individuals.append(orig_genotype)
            else:
                possible_new_individuals.append(
                    Individual(new_genotype, measurements[measurements_index])
                )
                measurements_index += 1

        self.root.program_state.performed_evaluations += len(eval_descrs)

        offspring = Population(possible_new_individuals)

        pop_indices = replace_if_better(
            self.root.program_state.population,
            offspring,
            measure="fitness",
        )

        self.root.program_state.population = Population.from_existing_equally_sized_populations(  # type: ignore # TODO
            [self.root.program_state.population, offspring],
            pop_indices,
            [
                "fitness",
            ],
        )

    @staticmethod
    def get_new_genotype(
        node: Node,
        pop: Population,
        rng: Rng,
        standard_deviation: float,
        migration_probability: float,
    ) -> Optional[GenotypeWithHeritage]:
        if rng.rng.random() < migration_probability:  # migrate
            neighbours_index = rng.rng.integers(0, len(node.neighbours))
            chosen_neighbour = node.neighbours[neighbours_index]
            chosen_genotype = pop[chosen_neighbour.index].genotype
            if node.index in chosen_genotype.genotype.visited_environments:
                return None
            else:
                chosen_genotype.genotype.visited_environments.append(node.index)
                return GenotypeWithHeritage(
                    genotype=chosen_genotype.genotype, came_from=chosen_neighbour.index
                )
        else:  # innovate
            permutation = standard_deviation * rng.rng.standard_normal(
                len(pop[node.index].genotype.genotype.genotype)
            )
            new_genotype = GenotypeWithHeritage(
                genotype=GenotypeWithMeta(
                    bounce_parameters(
                        Genotype(
                            pop[node.index].genotype.genotype.genotype + permutation
                        )
                    ),
                    VisitedEnvironments([]),
                ),
                came_from=node.index,
            )
            new_genotype.genotype.visited_environments.append(node.index)
            return new_genotype

    async def measure(self, eval_descrs: List[EvaluationDescription]) -> List[Measures]:
        """
        Measure all provided genotypes.

        :param pop: The genotypes.
        :returns: Measures for the genotypes.
        """
        fitnesses = await self.evaluator.evaluate(eval_descrs)
        return [Measures(fitness=fitness) for fitness in fitnesses]
