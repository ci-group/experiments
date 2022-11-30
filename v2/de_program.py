import logging

from revolve2.core.database import open_async_database_sqlite
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from typing import List
from revolve2.core.database.std import Rng
import numpy as np
from environment import Environment
from environment_name import EnvironmentName
from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import (
    SerializableIncrementableStruct,
    SerializableFrozenList,
    SerializableFrozenSingleton,
)
from dataclasses import dataclass
from revolve2.core.optimization.ea.population import (
    Individual,
    Parameters,
    SerializableMeasures,
)
from typing import Optional
from revolve2.core.optimization.ea.population.pop_list import PopList, replace_if_better
from sqlalchemy.ext.asyncio import AsyncEngine
from bodies import make_cpg_network_structure
from revolve2.core.optimization.ea.algorithms import de_offspring, bounce_parameters
from evaluator import Evaluator, EvaluationDescription

Genotype = Parameters


@dataclass
class Measures(SerializableMeasures, table_name="measures"):
    """Measures of a genotype/phenotype."""

    fitness: Optional[float] = None


class Population(PopList[Genotype, Measures], table_name="population"):
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


@dataclass
class ProgramRoot(SerializableFrozenSingleton, table_name="program_root"):
    environment_names: EnvironmentNames
    program_state: ProgramState


class Program:
    population_size: float
    crossover_probability: float
    differential_weight: float
    num_evaluations: int
    environments: List[Environment]

    root: ProgramRoot
    database: AsyncEngine
    evaluator: Evaluator

    async def run(
        self,
        database_name: str,
        headless: bool,
        rng_seed: int,
        population_size: int,
        crossover_probability: float,
        differential_weight: float,
        num_evaluations: int,
        environments: List[Environment],
        num_simulators: int,
    ) -> None:
        logging.info("Program start.")

        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.differential_weight = differential_weight
        self.num_evaluations = num_evaluations
        self.environments = environments

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
        while (
            performed_evaluations := (self.root.program_state.generation_index + 1)
            * self.population_size
            * len(self.environments)
        ) < self.num_evaluations:
            logging.info(f"Current # evaluations: {performed_evaluations}")
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

        initial_population = Population(
            [
                Individual(
                    Genotype(
                        [
                            float(v)
                            for v in initial_rng.rng.random(
                                size=cpg_network_structure.num_connections
                            )
                        ]
                    ),
                    Measures(),
                )
                for _ in range(self.population_size)
            ]
        )
        logging.info("Measuring initial population..")
        await self.measure(initial_population)
        logging.info("Measuring initial population done.")

        logging.info("Saving root..")
        state = ProgramState(
            rng=initial_rng,
            population=initial_population,
            generation_index=0,
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

    async def measure(self, population: Population) -> None:
        """
        Measure all individuals in a population.

        :param pop: The population.
        """
        evalsets: List[EvaluationDescription] = []

        for environment in self.environments:
            for individual in population:
                evalsets.append(EvaluationDescription(environment, individual.genotype))

        fitnesses = np.array(await self.evaluator.evaluate(evalsets))

        fitnesses.resize(len(self.environments), len(population))
        combined_fitnesses = np.average(np.sqrt(fitnesses), axis=0) ** 2

        for individual, fitness in zip(population, combined_fitnesses):
            individual.measures.fitness = float(fitness)

    async def evolve(self) -> None:
        """Iterate one generation further."""
        self.root.program_state.generation_index += 1

        offspring = Population(
            [
                Individual(bounce_parameters(genotype), Measures())
                for genotype in de_offspring(
                    self.root.program_state.population,
                    self.root.program_state.rng,
                    self.differential_weight,
                    self.crossover_probability,
                )
            ]
        )

        await self.measure(offspring)

        pop_indices = replace_if_better(
            self.root.program_state.population, offspring, measure="fitness"
        )

        self.root.program_state.population = Population.from_existing_equally_sized_populations(  # type: ignore # TODO
            [self.root.program_state.population, offspring],
            pop_indices,
            [
                "fitness",
            ],
        )
