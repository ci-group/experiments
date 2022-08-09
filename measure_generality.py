import numpy.typing as npt
import numpy as np
from typing import List, Dict
from bodies import make_bodies, make_cpg_network_structure
from evaluator import Evaluator, Setting, Environment
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
import sqlalchemy
from revolve2.core.database import open_async_database_sqlite
import os

EVAL_TIME = 30
SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 60


def get_specialist_brains() -> List[
    List[npt.NDArray[np.float_]]
]:  # runs -> bodies -> brain
    return []  # TODO


def get_generalist_brains() -> List[npt.NDArray[np.float_]]:  # runs -> brain
    return []  # TODO


def get_graph_brains() -> List[List[npt.NDArray[np.float_]]]:  # runs -> bodies -> brain
    return []  # TODO


async def main() -> None:
    if os.path.isdir("final_fitnesses"):
        raise RuntimeError("Database 'final_fitnesses' already exists")
    database = open_async_database_sqlite("final_fitnesses")

    async with AsyncSession(database) as session:
        async with session.begin():
            await (await session.connection()).run_sync(DbBase.metadata.create_all)

    cpg_network_structure = make_cpg_network_structure()
    bodies, dof_maps = make_bodies()

    all_brains = []
    all_brain_names_suffix = []
    for run_i, bodies in enumerate(get_specialist_brains()):
        for body_i, brain in enumerate(bodies):
            all_brains.append(brain)
            all_brain_names_suffix.append(
                f"specialist_run{run_i}_optimizedforbody{body_i}"
            )

    for run_i, brain in enumerate(get_generalist_brains()):
        all_brains.append(brain)
        all_brain_names_suffix.append(f"generalist_run{run_i}")

    for run_i, bodies in enumerate(get_graph_brains()):
        for body_i, brain in enumerate(bodies):
            all_brains.append(brain)
            all_brain_names_suffix.append(f"graph_run{run_i}_node{body_i}")

    evaluator = Evaluator(cpg_network_structure)

    settings = []
    for body, dof_map in zip(bodies, dof_maps):
        for brain in all_brains:
            settings.append(Setting(Environment(body, dof_map), brain))
    fitnesses = await evaluator.evaluate(settings)

    dbfitnesses = []

    fitness_i = 0
    for body_i in range(len(bodies)):
        for brain_name_suffix in all_brain_names_suffix:
            dbfitnesses.append(
                DbFitness(
                    brain_name=f"body{body_i}_{brain_name_suffix}",
                    fitness=fitnesses[fitness_i],
                )
            )
            fitness_i += 1

    async with AsyncSession(database) as session:
        async with session.begin():
            session.add_all(dbfitnesses)


DbBase = declarative_base()


class DbFitness(DbBase):
    """Model for the optimizer itself, containing static parameters."""

    __tablename__ = "fitness"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        primary_key=True,
        autoincrement=True,
    )
    brain_name = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
