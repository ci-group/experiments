import logging
import argparse
from parse_runs_arg import parse_runs_arg
from make_graph import make_graph
from graph import Graph
from typing import List
from environment import Environment
from experiment_settings import GRAPH_PARAMS, RUGGEDNESS_RANGE, BOWLNESS_RANGE
from environment_name import EnvironmentName
from bodies import make_bodies, make_cpg_network_structure
import itertools
from evaluator import Evaluator, EvaluationDescription
import experiments
import os
from revolve2.core.database import open_database_sqlite, open_async_database_sqlite
import pandas
from sqlalchemy.future import select
import graph_program
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy import Integer, Float, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session


DbBase = declarative_base()


class XMeasureTable(DbBase):
    __tablename__ = "cross_measure"

    id = Column(
        Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    solution_env_i = Column(Integer, nullable=False)
    measured_env_i = Column(Integer, nullable=False)
    fitness = Column(Float, nullable=False)


def find_index(in_envs: List[Environment], env_name: EnvironmentName) -> int:
    for i, env in enumerate(in_envs):
        if (
            env.name.body_num == env_name.body_num
            and env.name.bowlness_num == env_name.bowlness_num
            and env.name.ruggedness_num == env_name.ruggedness_num
        ):
            return i
    raise ValueError("Env does not exist")


def select_environments(environments: List[Environment]) -> List[int]:
    num_bodies = len(make_bodies()[0])
    env_names = [
        EnvironmentName(body_num, ruggedness_i, bowlness_i)
        for (body_num, (ruggedness_i, bowlness_i)) in itertools.product(
            range(num_bodies),
            [
                (0, 0),
                (len(RUGGEDNESS_RANGE) - 1, 0),
                (0, len(BOWLNESS_RANGE) - 1),
                (len(RUGGEDNESS_RANGE) - 1, len(BOWLNESS_RANGE) - 1),
                ((len(RUGGEDNESS_RANGE) - 1) // 2, (len(BOWLNESS_RANGE) - 1) // 2),
            ],
        )
    ]
    indices = [find_index(environments, env) for env in env_names]
    return indices


async def measure_generality_graph(
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    run: int,
    num_simulators: int,
    graph_params_i: int,
    env_indices: List[int],
    database_directory_out: str,
) -> None:
    standard_deviation = GRAPH_PARAMS[graph_params_i][0]
    migration_probability = GRAPH_PARAMS[graph_params_i][1]
    alpha1 = GRAPH_PARAMS[graph_params_i][2]
    alpha2 = GRAPH_PARAMS[graph_params_i][3]
    theta1 = GRAPH_PARAMS[graph_params_i][4]
    theta2 = GRAPH_PARAMS[graph_params_i][5]

    database_name = os.path.join(
        database_directory,
        experiments.graph_database_name(
            run,
            standard_deviation,
            migration_probability,
            alpha1,
            alpha2,
            theta1,
            theta2,
        ),
    )

    db = open_database_sqlite(database_name)

    df = pandas.read_sql(
        select(
            graph_program.ProgramState.table,
            graph_program.Population.item_table,
            graph_program.Measures.table,
            EnvironmentName.table,
        ).filter(
            (
                graph_program.ProgramState.table.population
                == graph_program.Population.item_table.list_id
            )
            & (
                graph_program.Population.item_table.measures
                == graph_program.Measures.table.id
            )
            & (
                EnvironmentName.table.id
                == (graph_program.Population.item_table.index + 1)
            )
        ),
        db,
    )
    lastgen = df[(df.generation_index == df.generation_index.max())]

    cpg_network_structure = make_cpg_network_structure()
    evaluator = Evaluator(
        cpg_network_structure,
        headless=True,
        num_simulators=num_simulators,
    )

    database_name_out = os.path.join(
        database_directory_out,
        experiments.graph_database_name(
            run,
            standard_deviation,
            migration_probability,
            alpha1,
            alpha2,
            theta1,
            theta2,
        ),
    )

    genotypes: List[graph_program.Genotype] = []
    measures: List[XMeasureTable] = []

    for env_index in env_indices:
        main_env = environments[env_index]
        individual = lastgen[
            (lastgen.body_num == main_env.name.body_num)
            & (lastgen.ruggedness_num == main_env.name.ruggedness_num)
            & (lastgen.bowlness_num == main_env.name.bowlness_num)
        ]
        assert (
            len(individual)
        ) == 1, "cannot find best individual for these parameters"
        best_params_id = int(individual.genotype)
        measures.append(
            XMeasureTable(
                solution_env_i=env_index,
                measured_env_i=env_index,
                fitness=individual.fitness,
            )
        )

        db2 = open_async_database_sqlite(database_name)
        async with AsyncSession(db2) as session:
            genotype = await graph_program.GenotypeWithMeta.from_db(
                session, best_params_id
            )
            genotypes.append(genotype.genotype)

    evaldescrs: List[EvaluationDescription] = []

    for genotype, env_index in zip(genotypes, env_indices):
        for other_envs_i in env_indices:
            if other_envs_i == env_index:
                continue
            evaldescrs.append(
                EvaluationDescription(environments[other_envs_i], genotype)
            )
    fitnesses = await evaluator.evaluate(evaldescrs)

    f_i = 0
    for genotype, env_index in zip(genotypes, env_indices):
        for other_envs_i in env_indices:
            if other_envs_i == env_index:
                continue
            measures.append(
                XMeasureTable(
                    solution_env_i=env_index,
                    measured_env_i=other_envs_i,
                    fitness=fitnesses[f_i],
                )
            )
            f_i += 1

    outdb = open_database_sqlite(database_name_out, create=True)
    DbBase.metadata.create_all(outdb)
    with Session(outdb) as ses:
        ses.add_all(measures)
        ses.commit()


async def run_graph_all(
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    runs: List[int],
    num_simulators: int,
    database_directory_out: str,
) -> None:
    env_indices = select_environments(environments)

    for run in runs:
        for graph_params_i in range(len(GRAPH_PARAMS)):
            await measure_generality_graph(
                graph=graph,
                environments=environments,
                database_directory=database_directory,
                run=run,
                num_simulators=num_simulators,
                graph_params_i=graph_params_i,
                env_indices=env_indices,
                database_directory_out=database_directory_out,
            )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database_directory", type=str, required=True)
    parser.add_argument("-o", "--database_directory_out", type=str, required=True)
    parser.add_argument("-r", "--runs", type=str, required=True)
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    parser.add_argument("-s", "--num_simulators", type=int, required=True)

    subparsers.add_parser("de")

    subparsers.add_parser("graph")

    args = parser.parse_args()
    runs = parse_runs_arg(args.runs)
    logging.info(f"Running runs {runs} (including last one).")

    graph, environments = make_graph()

    if args.experiment == "graph":
        await run_graph_all(
            graph=graph,
            environments=environments,
            runs=runs,
            database_directory=args.database_directory,
            num_simulators=args.num_simulators,
            database_directory_out=args.database_directory_out,
        )
    # elif args.experiment == "de":
    #     await run_de_all(
    #         graph=graph,
    #         environments=environments,
    #         runs=runs,
    #         database_directory=args.database_directory,
    #         num_simulators=args.num_simulators,
    #     )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
