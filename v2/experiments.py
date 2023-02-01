from experiment_settings import NUM_EVALUATIONS, DE_PARAMS, GRAPH_PARAMS, CMAES_PARAMS
import argparse
from typing import List
from environment import Environment
import de_program
import os
import graph_program
import logging
from make_graph import make_graph
from partition import partition
from graph import Graph
from parse_runs_arg import parse_runs_arg
import cmaes_program


def graph_database_name(
    run: int,
    standard_deviation: float,
    migration_probability: float,
    alpha1: float,
    alpha2: float,
    theta1: float,
    theta2: float,
) -> str:
    return f"graph_s{standard_deviation}_mp{migration_probability}_a1{alpha1}_a2{alpha2}_t1{theta1}_t2{theta2}_run{run}"


def de_database_name(
    run: int,
    population_size: int,
    crossover_probability: float,
    differential_weight: float,
    partition_size: int,
    partition_num: int,
) -> str:
    return f"de_p{population_size}_cr{crossover_probability}_f{differential_weight}_psize{partition_size}_pnum{partition_num}_run{run}"


def cmaes_database_name(
    run: int,
    partition_size: int,
    partition_num: int,
) -> str:
    return f"de_psize{partition_size}_pnum{partition_num}_run{run}"


async def run_graph(
    rng_seed: int,
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    graph_params_i: int,
    num_simulators: int,
    run: int,
) -> None:
    standard_deviation = GRAPH_PARAMS[graph_params_i][0]
    migration_probability = GRAPH_PARAMS[graph_params_i][1]
    alpha1 = GRAPH_PARAMS[graph_params_i][2]
    alpha2 = GRAPH_PARAMS[graph_params_i][3]
    theta1 = GRAPH_PARAMS[graph_params_i][4]
    theta2 = GRAPH_PARAMS[graph_params_i][5]

    logging.info(
        f"Running graph s{standard_deviation} mp{migration_probability} a1{alpha1} a2{alpha2} t1{theta1} t2{theta2} run{run}"
    )

    await graph_program.Program().run(
        database_name=os.path.join(
            database_directory,
            graph_database_name(
                run=run,
                standard_deviation=standard_deviation,
                migration_probability=migration_probability,
                alpha1=alpha1,
                alpha2=alpha2,
                theta1=theta1,
                theta2=theta2,
            ),
        ),
        headless=True,
        rng_seed=rng_seed,
        environments=environments,
        graph=graph,
        num_evaluations=NUM_EVALUATIONS,
        standard_deviation=standard_deviation,
        migration_probability=migration_probability,
        alpha1=alpha1,
        alpha2=alpha2,
        theta1=theta1,
        theta2=theta2,
        num_simulators=num_simulators,
    )


async def run_graph_all(
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    runs: List[int],
    num_simulators: int,
) -> None:
    SEED_BASE = 732091019

    for run in runs:
        for graph_params_i in range(len(GRAPH_PARAMS)):
            await run_graph(
                rng_seed=(hash(SEED_BASE) + hash(graph_params_i) + hash(run)),
                graph=graph,
                environments=environments,
                database_directory=database_directory,
                graph_params_i=graph_params_i,
                num_simulators=num_simulators,
                run=run,
            )


async def run_de(
    rng_seed: int,
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    de_params_i: int,
    num_simulators: int,
    partition_num: int,
    num_evaluations: int,
    run: int,
) -> None:
    population_size = DE_PARAMS[de_params_i][0]
    crossover_probability = DE_PARAMS[de_params_i][1]
    differential_weight = DE_PARAMS[de_params_i][2]
    partition_size = DE_PARAMS[de_params_i][3]

    logging.info(
        f"Running de p{population_size} cr{crossover_probability} f{differential_weight} psize{partition_size} pnum{partition_num} run{run}"
    )

    used_envs = [environments[node.index] for node in graph.nodes]

    await de_program.Program().run(
        database_name=os.path.join(
            database_directory,
            de_database_name(
                run=run,
                population_size=population_size,
                crossover_probability=crossover_probability,
                differential_weight=differential_weight,
                partition_size=partition_size,
                partition_num=partition_num,
            ),
        ),
        headless=True,
        rng_seed=rng_seed,
        population_size=population_size,
        crossover_probability=crossover_probability,
        differential_weight=differential_weight,
        num_evaluations=num_evaluations,
        environments=used_envs,
        num_simulators=num_simulators,
    )


async def run_de_all(
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    runs: List[int],
    num_simulators: int,
) -> None:
    SEED_BASE = 196783254

    for run in runs:
        for de_params_i in range(len(DE_PARAMS)):
            partition_size = DE_PARAMS[de_params_i][3]
            num_partitions = round(len(graph.nodes) / partition_size)
            num_evaluations = round(NUM_EVALUATIONS / num_partitions)

            logging.info("Making partitions..")
            partitions = partition(graph, environments, num_partitions)
            logging.info("Done making partitions.")
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


async def run_cmaes(
    rng_seed: int,
    graph: Graph,
    environments: List[Environment],
    database_directory: str,
    cmaes_params_i: int,
    num_simulators: int,
    partition_num: int,
    num_evaluations: int,
    run: int,
) -> None:
    partition_size = CMAES_PARAMS[cmaes_params_i][0]

    logging.info(f"Running cmaes psize{partition_size} pnum{partition_num} run{run}")

    used_envs = [environments[node.index] for node in graph.nodes]

    await cmaes_program.Program().run(
        database_name=os.path.join(
            database_directory,
            cmaes_database_name(
                run=run,
                partition_size=partition_size,
                partition_num=partition_num,
            ),
        ),
        headless=True,
        rng_seed=rng_seed,
        num_evaluations=num_evaluations,
        environments=used_envs,
        num_simulators=num_simulators,
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
        )
    elif args.experiment == "de":
        await run_de_all(
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
