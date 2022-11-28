from experiment_settings import (
    NUM_EVALUATIONS,
    RUGGEDNESS_RANGE,
    BOWLNESS_RANGE,
    TERRAIN_SIZE,
    TERRAIN_GRANULARITY,
    DE_PARAMS,
    GRAPH_PARAMS,
)
from bodies import make_bodies
import argparse
from typing import List, Tuple
from environment_name import EnvironmentName
from environment import Environment
from terrain_generator import terrain_generator
from bodies import make_bodies
import de_program
import os
from graph import Graph, Node
import graph_program
import logging
from typed_argparse import Choices


def de_generalist_database_name(
    run: int,
    population_size: int,
    crossover_probability: float,
    differential_weight: float,
) -> None:
    return f"de_generalist_p{population_size}_cr{crossover_probability}_f{differential_weight}_run{run}"


def de_specialist_database_name(
    run: int,
    population_size: int,
    crossover_probability: float,
    differential_weight: float,
    body_i: int,
    ruggedness_i: int,
    bowlness_i: int,
) -> None:
    return f"de_specialist_p{population_size}_cr{crossover_probability}_f{differential_weight}_body{body_i}_ruggedness{ruggedness_i}_bowlness{bowlness_i}_run{run}"


def graph_database_name(run: int, standard_deviation: float) -> None:
    return f"graph_s{standard_deviation}_run{run}"


async def run_de_generalist(
    database_directory: str, runs: List[int], de_params_i: int, num_simulators: int
) -> None:
    SEED_BASE = 23678400

    population_size = DE_PARAMS[de_params_i][0]
    crossover_probability = DE_PARAMS[de_params_i][1]
    differential_weight = DE_PARAMS[de_params_i][2]

    bodies, dof_maps = make_bodies()

    environments: List[Environment] = []

    for ruggedness_i, ruggedness in enumerate(RUGGEDNESS_RANGE):
        for bowlness_i, bowlness in enumerate(BOWLNESS_RANGE):
            terrain = terrain_generator(
                size=TERRAIN_SIZE,
                ruggedness=ruggedness,
                bowlness=bowlness,
                granularity_multiplier=TERRAIN_GRANULARITY,
            )
            for body_i, (body, dof_map) in enumerate(zip(bodies, dof_maps)):
                environments.append(
                    Environment(
                        body,
                        dof_map,
                        terrain,
                        EnvironmentName(body_i, ruggedness_i, bowlness_i),
                    )
                )

    for run in runs:
        logging.info(
            f"Running de generalist p{population_size} cr{crossover_probability} f{differential_weight} run{run}"
        )

        await de_program.Program().run(
            database_name=os.path.join(
                database_directory,
                de_generalist_database_name(
                    run=run,
                    population_size=population_size,
                    crossover_probability=crossover_probability,
                    differential_weight=differential_weight,
                ),
            ),
            headless=True,
            rng_seed=SEED_BASE + run * len(DE_PARAMS) + de_params_i,
            population_size=population_size,
            crossover_probability=crossover_probability,
            differential_weight=differential_weight,
            num_evaluations=NUM_EVALUATIONS,
            environments=environments,
            num_simulators=num_simulators,
        )


async def run_de_specialist(
    database_directory: str,
    runs: List[int],
    body_i: int,
    ruggedness_i: int,
    bowlness_i: int,
    de_params_i: int,
    num_simulators: int,
) -> None:
    SEED_BASE = 196783254

    num_evaluations = NUM_EVALUATIONS / len(
        make_bodies()[0] * len(RUGGEDNESS_RANGE) * len(BOWLNESS_RANGE)
    )

    ruggedness = RUGGEDNESS_RANGE[ruggedness_i]
    bowlness = BOWLNESS_RANGE[bowlness_i]

    population_size = DE_PARAMS[de_params_i][0]
    crossover_probability = DE_PARAMS[de_params_i][1]
    differential_weight = DE_PARAMS[de_params_i][2]

    bodies, dof_maps = make_bodies()

    environment = Environment(
        bodies[body_i],
        dof_maps[body_i],
        terrain_generator(
            size=TERRAIN_SIZE,
            ruggedness=ruggedness,
            bowlness=bowlness,
            granularity_multiplier=TERRAIN_GRANULARITY,
        ),
        EnvironmentName(body_i, ruggedness_i, bowlness_i),
    )

    for run in runs:
        logging.info(
            f"Running de specialist p{population_size} cr{crossover_probability} f{differential_weight} body{body_i} ruggedness{ruggedness_i} bowlness{bowlness_i} run{run}"
        )

        await de_program.Program().run(
            database_name=os.path.join(
                database_directory,
                de_specialist_database_name(
                    run=run,
                    population_size=population_size,
                    crossover_probability=crossover_probability,
                    differential_weight=differential_weight,
                    body_i=body_i,
                    ruggedness_i=ruggedness_i,
                    bowlness_i=bowlness_i,
                ),
            ),
            headless=True,
            rng_seed=SEED_BASE
            + run
            * len(BOWLNESS_RANGE)
            * len(RUGGEDNESS_RANGE)
            * len(bodies)
            * len(DE_PARAMS)
            + bowlness_i * len(RUGGEDNESS_RANGE) * len(bodies) * len(DE_PARAMS)
            + ruggedness_i * len(bodies) * len(DE_PARAMS)
            + body_i * len(DE_PARAMS)
            + de_params_i,
            population_size=population_size,
            crossover_probability=crossover_probability,
            differential_weight=differential_weight,
            num_evaluations=num_evaluations,
            environments=[environment],
            num_simulators=num_simulators,
        )


async def run_de_specialist_all(
    database_directory: str, runs: List[int], de_params_i: int, num_simulators: int
) -> None:
    for run in runs:
        for ruggedness_i in range(len(RUGGEDNESS_RANGE)):
            for bowlness_i in range(len(BOWLNESS_RANGE)):
                for body_i in range(len(make_bodies()[0])):
                    await run_de_specialist(
                        database_directory=database_directory,
                        runs=[run],
                        body_i=body_i,
                        ruggedness_i=ruggedness_i,
                        bowlness_i=bowlness_i,
                        de_params_i=de_params_i,
                        num_simulators=num_simulators,
                    )


def make_graph() -> Tuple[Graph, List[Environment]]:
    bodies, dof_maps = make_bodies()

    nodes: List[Node] = []
    envs: List[Environment] = []

    for ruggedness_i, ruggedness in enumerate(RUGGEDNESS_RANGE):
        for bowlness_i, bowlness in enumerate(BOWLNESS_RANGE):
            terrain = terrain_generator(
                size=TERRAIN_SIZE,
                ruggedness=ruggedness,
                bowlness=bowlness,
                granularity_multiplier=TERRAIN_GRANULARITY,
            )
            for body_i, (body, dof_map) in enumerate(zip(bodies, dof_maps)):
                node = Node(len(nodes))
                nodes.append(node)
                envs.append(
                    Environment(
                        body,
                        dof_map,
                        terrain,
                        EnvironmentName(body_i, ruggedness_i, bowlness_i),
                    )
                )
                nodes[-1].env = envs[-1]

    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:
                continue

            env1 = envs[node1.index]
            env2 = envs[node2.index]

            if (
                abs(env1.name.body_num - env2.name.body_num) <= 1
                and abs(env1.name.ruggedness_num - env2.name.ruggedness_num) <= 1
                and abs(env1.name.bowlness_num - env2.name.bowlness_num) <= 1
            ):
                node1.neighbours.append(node2)

    return Graph(nodes), envs


async def run_graph(
    runs: List[int], database_directory: str, graph_params_i: int, num_simulators: int
) -> None:
    SEED_BASE = 732091019

    standard_deviation = GRAPH_PARAMS[graph_params_i][0]
    migration_probability = GRAPH_PARAMS[graph_params_i][1]

    graph, environments = make_graph()

    for run in runs:
        logging.info(f"Running graph s{standard_deviation} run{run}")

        await graph_program.Program().run(
            database_name=os.path.join(
                database_directory,
                graph_database_name(run=run, standard_deviation=standard_deviation),
            ),
            headless=True,
            rng_seed=SEED_BASE + run * len(GRAPH_PARAMS) + graph_params_i,
            environments=environments,
            graph=graph,
            num_evaluations=NUM_EVALUATIONS,
            standard_deviation=standard_deviation,
            migration_probability=migration_probability,
            num_simulators=num_simulators,
        )


async def run_all(
    experiments: List[str],
    runs: List[int],
    database_directory: str,
    num_simulators: int,
) -> None:
    for run in runs:
        for de_params_i in range(len(DE_PARAMS)):
            if len(experiments) == 0 or "de_generalist" in experiments:
                await run_de_generalist(
                    database_directory=database_directory,
                    runs=[run],
                    de_params_i=de_params_i,
                    num_simulators=num_simulators,
                )
            if len(experiments) == 0 or "de_specialist" in experiments:
                await run_de_specialist_all(
                    database_directory=database_directory,
                    runs=[run],
                    de_params_i=de_params_i,
                    num_simulators=num_simulators,
                )
        for graph_params_i in range(len(GRAPH_PARAMS)):
            if len(experiments) == 0 or "graph" in experiments:
                await run_graph(
                    runs=[run],
                    database_directory=database_directory,
                    graph_params_i=graph_params_i,
                    num_simulators=num_simulators,
                )


def parse_runs_arg(cliarg: str) -> None:
    if cliarg.isnumeric():
        return [int(cliarg)]
    else:
        parts = cliarg.split(":")
        if len(parts) != 2 or not parts[0].isnumeric() or not parts[1].isnumeric():
            raise ValueError()
        low = int(parts[0])
        high = int(parts[1])
        if low > high:
            raise ValueError()
        return [i for i in range(low, high + 1)]


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database_directory", type=str, required=True)
    parser.add_argument("-r", "--runs", type=str, required=True)
    subparsers = parser.add_subparsers(dest="experiment")
    parser.add_argument("-s", "--num_simulators", type=int, required=True)

    de_generalist_parser = subparsers.add_parser("de_generalist")
    de_generalist_parser.add_argument("--de_params_i", type=int, required=True)

    de_specialist_parser = subparsers.add_parser("de_specialist")
    de_specialist_parser.add_argument("--de_params_i", type=int, required=True)

    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("--graph_params_i", type=int, required=True)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument(
        "experiments",
        choices=Choices("de_generalist", "de_specialist", "graph"),
        default=[],
        nargs="*",
    )

    args = parser.parse_args()
    runs = parse_runs_arg(args.runs)
    logging.info(f"Running runs {runs} (including last one).")

    if args.experiment == "de_generalist":
        await run_de_generalist(
            database_directory=args.database_directory,
            runs=runs,
            de_params_i=args.de_params_i,
            num_simulators=args.num_simulators,
        )
    elif args.experiment == "de_specialist":
        await run_de_specialist_all(
            database_directory=args.database_directory,
            runs=runs,
            de_params_i=args.de_params_i,
            num_simulators=args.num_simulators,
        )
    elif args.experiment == "graph":
        await run_graph(
            runs=runs,
            database_directory=args.database_directory,
            graph_params_i=args.graph_params_i,
            num_simulators=args.num_simulators,
        )
    elif args.experiment == "all":
        await run_all(
            runs=runs,
            database_directory=args.database_directory,
            num_simulators=args.num_simulators,
            experiments=args.experiments,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
