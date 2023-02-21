from parse_runs_arg import parse_runs_arg
import logging
import argparse
from make_graph import make_graph
from experiment_settings import (
    RUGGEDNESS_RANGE,
    BOWLNESS_RANGE,
    NUM_EVALUATIONS,
    DE_PARAMS,
    CMAES_PARAMS,
)
from typing import List
from environment import Environment
from bodies import make_bodies
from environment_name import EnvironmentName
import itertools
from experiments import run_de, run_cmaes
from graph import Graph, Node
from partition import partition as make_partitions


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


async def run_de_all(
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    runs: List[int],
    num_simulators: int,
) -> None:
    SEED_BASE = 196783256

    de_params_i = 1

    partition_size = DE_PARAMS[de_params_i][3]
    num_partitions = round(len(graph.nodes) / partition_size)
    num_evaluations = round(NUM_EVALUATIONS / num_partitions)

    logging.info("Making partitions..")
    partitions = make_partitions(graph, environments, num_partitions)
    logging.info("Done making partitions.")

    for run in runs:
        for part_i, part in enumerate(partitions):
            await run_de(
                rng_seed=(
                    hash(SEED_BASE) + hash(de_params_i) + hash(run) + hash(part_i)
                ),
                graph=part,
                environments=environments,
                database_directory=database_directory,
                de_params_i=de_params_i,
                num_simulators=num_simulators,
                partition_num=part_i,
                num_evaluations=num_evaluations,
                run=run,
            )


async def run_cmaes_all(
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    runs: List[int],
    num_simulators: int,
) -> None:
    SEED_BASE = 196783255

    cmaes_params_i = 1

    partition_size = CMAES_PARAMS[cmaes_params_i][1]
    num_partitions = round(len(graph.nodes) / partition_size)
    num_evaluations = round(NUM_EVALUATIONS / num_partitions)

    logging.info("Making partitions..")
    partitions = make_partitions(graph, environments, num_partitions)
    logging.info("Done making partitions.")

    for run in runs:
        for part_i, part in enumerate(partitions):
            await run_cmaes(
                rng_seed=(
                    hash(SEED_BASE) + hash(cmaes_params_i) + hash(run) + hash(part_i)
                ),
                graph=part,
                environments=environments,
                database_directory=database_directory,
                cmaes_params_i=cmaes_params_i,
                num_simulators=num_simulators,
                partition_num=part_i,
                num_evaluations=num_evaluations,
                run=run,
            )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database_directory", type=str, required=True)
    parser.add_argument("-r", "--runs", type=str, required=True)
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    parser.add_argument("-s", "--num_simulators", type=int, required=True)

    subparsers.add_parser("de")

    subparsers.add_parser("cmaes")

    args = parser.parse_args()
    runs = parse_runs_arg(args.runs)
    logging.info(f"Running runs {runs} (including last one).")

    graph, environments = make_graph()

    if args.experiment == "de":
        await run_de_all(
            graph=graph,
            environments=environments,
            runs=runs,
            database_directory=args.database_directory,
            num_simulators=args.num_simulators,
        )
    elif args.experiment == "cmaes":
        await run_cmaes_all(
            graph=graph,
            environments=environments,
            runs=runs,
            database_directory=args.database_directory,
            num_simulators=args.num_simulators,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
