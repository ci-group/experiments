import logging

from revolve2.core.database import open_async_database_sqlite
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from typing import List, Optional
from revolve2.core.database.std import Rng
import numpy as np
from environment import Environment
from environment_name import EnvironmentName
from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import (
    SerializableIncrementableStruct,
    SerializableFrozenList,
    SerializableFrozenSingleton,
    SerializableList,
    SerializableStruct,
)
from dataclasses import dataclass
from revolve2.core.optimization.ea.population import (
    Parameters,
    SerializableMeasures,
)
from sqlalchemy.ext.asyncio import AsyncEngine
from bodies import make_cpg_network_structure
from evaluator import Evaluator, EvaluationDescription
import cma

Genotype = Parameters


class Fitnesses(
    SerializableList[float], table_name="fitnesses", value_column_name="fitness"
):
    """Fitnesses per environment."""


@dataclass
class Measures(SerializableMeasures, table_name="measures"):
    """Measures of a genotype/phenotype."""

    fitnesses: Optional[Fitnesses] = None
    combined_fitness: Optional[float] = None


@dataclass
class Individual(SerializableStruct, table_name="individual"):
    genotype: Genotype
    measures: Measures


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
    mean: Individual
    generation_index: int


@dataclass
class ProgramRoot(SerializableFrozenSingleton, table_name="program_root"):
    environment_names: EnvironmentNames
    program_state: ProgramState


class Program:
    num_evaluations: int
    environments: List[Environment]

    root: ProgramRoot
    database: AsyncEngine
    evaluator: Evaluator

    opt: cma.CMAEvolutionStrategy

    performed_evaluations: int

    async def run(
        self,
        database_name: str,
        headless: bool,
        rng_seed: int,
        num_evaluations: int,
        environments: List[Environment],
        num_simulators: int,
        initial_std: float,
    ) -> None:
        logging.info("Program start.")

        self.num_evaluations = num_evaluations
        self.environments = environments

        self.performed_evaluations = 0

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
                initial_std=initial_std,
            )
            logging.info("Initializing state done.")

        logging.info(
            f"Entering optimization loop. Continuing until around {self.num_evaluations} evaluations."
        )
        while self.performed_evaluations < self.num_evaluations:
            logging.info(f"Current # evaluations: {self.performed_evaluations}")
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
        initial_std: float,
    ) -> None:
        initial_rng = Rng(np.random.Generator(np.random.PCG64(rng_seed)))

        initial_mean = Individual(
            Genotype(
                [
                    float(v)
                    for v in initial_rng.rng.random(
                        size=cpg_network_structure.num_connections
                    )
                ]
            ),
            measures=Measures(),
        )

        await self.measure([initial_mean])
        self.performed_evaluations += 1

        # TODO seed
        options = cma.CMAOptions()
        options.set("bounds", [-1.0, 1.0])
        self.opt = cma.CMAEvolutionStrategy(initial_mean.genotype, initial_std, options)

        logging.info("Saving root..")
        state = ProgramState(
            rng=initial_rng,
            mean=initial_mean,
            generation_index=0,
        )
        self.root = ProgramRoot(EnvironmentNames([e.name for e in environments]), state)
        for name in self.root.environment_names:
            name.clear_id()
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
                    raise NotImplementedError("loading not supported for CMA-ES")
                    self.root = maybe_root
                    return True

    async def measure(self, individuals: List[Individual]) -> None:
        """
        Measure all individuals in a population.

        :param pop: The population.
        """
        evalsets: List[EvaluationDescription] = []

        for environment in self.environments:
            for individual in individuals:
                evalsets.append(EvaluationDescription(environment, individual.genotype))

        fitnesses = np.array(await self.evaluator.evaluate(evalsets))
        fitnesses.resize(len(individuals), len(self.environments))
        combined_fitnesses = np.average(fitnesses, axis=1)

        for individual, seperate_fitnesses, combined_fitness in zip(
            individuals, fitnesses, combined_fitnesses
        ):
            individual.measures.fitnesses = Fitnesses(seperate_fitnesses)
            individual.measures.combined_fitness = float(combined_fitness)

    async def evolve(self) -> None:
        """Iterate one generation further."""
        self.root.program_state.generation_index += 1

        offspring = [
            Individual(genotype=Genotype(s), measures=Measures())
            for s in self.opt.ask()
        ]
        await self.measure(offspring)
        self.performed_evaluations += len(offspring)
        self.opt.tell(
            [o.genotype for o in offspring],
            [-i.measures["combined_fitness"] for i in offspring],
        )
        self.opt.disp()

        m = Measures()
        m["combined_fitness"] = -self.opt.result.fbest
        m["fitnesses"] = []
        self.root.program_state.mean = Individual(
            genotype=Genotype(self.opt.result.xbest), measures=m
        )
