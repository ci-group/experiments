import numpy.typing as npt
import numpy as np
from typing import List, Dict, Tuple
from bodies import make_bodies, make_cpg_network_structure
from evaluator import Evaluator, Setting, Environment
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
import sqlalchemy
from revolve2.core.database import open_async_database_sqlite, open_database_sqlite
import os
from revolve2.core.optimization.ea.openai_es import DbOpenaiESOptimizerIndividual
import pandas
from revolve2.core.database.serializers import DbNdarray1xnItem

EVAL_TIME = 30
SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 60
NUM_RUNS = 2
NUM_BODIES = 5


async def get_specialist_brains() -> List[
    Tuple[str, List[List[npt.NDArray[np.float_]]]]
]:  # [params, runs -> bodies -> brain]
    res = []
    for params_str in ["s0.5l0.1"]:  # , "s0.05l0.01"]:
        res_run = []
        for run_i in range(NUM_RUNS):
            res_body = []
            for body_i in range(NUM_BODIES):
                db = open_database_sqlite(
                    f"full_specialist_{params_str}_body{body_i}_run{run_i}"
                )
                individuals = pandas.read_sql(
                    select(DbOpenaiESOptimizerIndividual, DbNdarray1xnItem).filter(
                        DbOpenaiESOptimizerIndividual.individual
                        == DbNdarray1xnItem.nparray1xn_id
                    ),
                    db,
                )
                individuals["fitness"] = individuals["fitness"] / EVAL_TIME
                individuals = individuals[
                    individuals.gen_num == individuals.gen_num.max()
                ]
                all_params = [
                    np.array(
                        individuals[individuals.gen_index == gen_i]
                        .sort_values(by=["array_index"])
                        .value
                    )
                    for gen_i in range(individuals.gen_index.max())
                ]
                avg_params = sum(all_params) / len(all_params)
                res_body.append(avg_params)
            res_run.append(res_body)
        res.append((params_str, res_run))
    return []


def get_generalist_brains() -> List[
    Tuple[str, List[npt.NDArray[np.float_]]]
]:  # [params, runs -> brain]
    res = []
    for params_str in ["s0.5l0.1", "s0.05l0.01"]:
        res_run = []
        for run_i in range(NUM_RUNS):
            db = open_database_sqlite(f"full_generalist_{params_str}_run{run_i}")
            individuals = pandas.read_sql(
                select(DbOpenaiESOptimizerIndividual, DbNdarray1xnItem).filter(
                    DbOpenaiESOptimizerIndividual.individual
                    == DbNdarray1xnItem.nparray1xn_id
                ),
                db,
            )
            individuals["fitness"] = individuals["fitness"] / EVAL_TIME
            individuals = individuals[individuals.gen_num == individuals.gen_num.max()]
            all_params = [
                np.array(
                    individuals[individuals.gen_index == gen_i]
                    .sort_values(by=["array_index"])
                    .value
                )
                for gen_i in range(individuals.gen_index.max())
            ]
            avg_params = sum(all_params) / len(all_params)
            res_run.append(avg_params)
        res.append((params_str, res_run))

    return []


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
    for params, runs in await get_specialist_brains():
        for run_i, bodies in enumerate(runs):
            for body_i, brain in enumerate(bodies):
                all_brains.append(brain)
                all_brain_names_suffix.append(
                    f"specialist_{params}_run{run_i}_optimizedforbody{body_i}"
                )

    for params, runs in get_generalist_brains():
        for run_i, brain in enumerate(runs):
            all_brains.append(brain)
            all_brain_names_suffix.append(f"generalist_{params}_run{run_i}")

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
