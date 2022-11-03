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


class DEMultiBodyOptimizer:
    """Program that optimizes the neural network parameters."""

    num_evaluations: int
    population_size: int
    crossover_probability: float
    differential_weight: float

    bodies: List[Body]
    dof_maps: List[Dict[int, int]]

    db: AsyncEngine
    evaluator: Evaluator

    state: ProgramState

    async def run(
        self,
        rng: Rng,
        database: AsyncEngine,
        robot_bodies: List[Body],
        dof_maps: List[Dict[int, int]],
        cpg_network_structure: CpgNetworkStructure,
        headless: bool,
        num_evaluations: int,
        population_size: int,
        crossover_probability: float,
        differential_weight: float,
    ) -> None:
        """Run the program."""
        self.db = database
        self.evaluator = Evaluator(cpg_network_structure, headless=headless)
        self.bodies = robot_bodies
        self.dof_maps = dof_maps
        self.num_evaluations = num_evaluations
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.differential_weight = differential_weight

        async with self.db.begin() as conn:
            await ProgramState.prepare_db(conn)

        if not await self.load_state():
            initial_population = Population(
                [
                    Individual(
                        Genotype(
                            [
                                float(v)
                                for v in rng.rng.random(
                                    size=cpg_network_structure.num_connections
                                )
                            ]
                        ),
                        Measures(),
                    )
                    for _ in range(self.population_size)
                ]
            )
            await self.measure(initial_population)

            initial_generation_index = 0

            self.state = ProgramState(
                rng=rng,
                population=initial_population,
                generation_index=initial_generation_index,
            )

            await self.save_state()

        while (self.state.generation_index + 1) * self.population_size * len(
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

        offspring = Population(
            [
                Individual(bounce_parameters(genotype), Measures())
                for genotype in de_offspring(
                    self.state.population,
                    self.state.rng,
                    self.differential_weight,
                    self.crossover_probability,
                )
            ]
        )

        await self.measure(offspring)

        original_selection, offspring_selection = replace_if_better(
            self.state.population, offspring, measure="fitness"
        )

        self.state.population = Population.from_existing_populations(  # type: ignore # TODO
            [self.state.population, offspring],
            [original_selection, offspring_selection],
            [
                "fitness",
            ],
        )

    async def measure(self, population: Population) -> None:
        """
        Measure all individuals in a population.

        :param pop: The population.
        """
        evalsets: List[EvaluationSetting] = []

        for body, dof_map in zip(self.bodies, self.dof_maps):
            for individual in population:
                evalsets.append(
                    EvaluationSetting(EvalEnv(body, dof_map), individual.genotype)
                )

        fitnesses = np.array(await self.evaluator.evaluate(evalsets))

        fitnesses.resize(len(self.bodies), len(population))
        combined_fitnesses = np.average(np.sqrt(fitnesses), axis=0) ** 2

        for individual, fitness in zip(population, combined_fitnesses):
            individual.measures.fitness = float(fitness)
