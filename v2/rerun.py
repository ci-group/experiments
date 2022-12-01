import argparse
from dataclasses import dataclass
from xml.sax import default_parser_list
from environment import Environment
from evaluator import Evaluator, EvaluationDescription
from bodies import make_cpg_network_structure, make_bodies
from terrain_generator import terrain_generator
from experiment_settings import (
    BOWLNESS_RANGE,
    DE_PARAMS,
    RUGGEDNESS_RANGE,
    TERRAIN_SIZE,
    TERRAIN_GRANULARITY,
    GRAPH_PARAMS,
)
from revolve2.core.optimization.ea.population import Parameters
import pandas
from sqlalchemy.future import select
import de_program
from revolve2.core.database import open_database_sqlite, open_async_database_sqlite
import experiments
import os
from sqlalchemy.ext.asyncio.session import AsyncSession
import graph_program
from environment_name import EnvironmentName
from typing import List, Tuple
from typed_argparse import Choices
from revolve2.core.physics import Terrain


def add_experiment_parsers(parent_parser: argparse.ArgumentParser):
    subparsers = parent_parser.add_subparsers(dest="experiment", required=True)
    de_generalist_parser = subparsers.add_parser("de_generalist")
    de_generalist_parser.add_argument("--de_params_i", type=int, required=True)

    de_specialist_parser = subparsers.add_parser("de_specialist")
    de_specialist_parser.add_argument("--de_params_i", type=int, required=True)
    de_specialist_parser.add_argument("--opt_body_i", type=int, required=True)
    de_specialist_parser.add_argument("--opt_ruggedness_i", type=int, required=True)
    de_specialist_parser.add_argument("--opt_bowlness_i", type=int, required=True)

    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("--graph_params_i", type=int, required=True)
    graph_parser.add_argument("--opt_body_i", type=int, required=True)
    graph_parser.add_argument("--opt_ruggedness_i", type=int, required=True)
    graph_parser.add_argument("--opt_bowlness_i", type=int, required=True)


async def load_best_de_generalist(
    database_directory: str, run: int, de_params_i: int
) -> Tuple[Parameters, float]:
    population_size = DE_PARAMS[de_params_i][0]
    crossover_probability = DE_PARAMS[de_params_i][1]
    differential_weight = DE_PARAMS[de_params_i][2]

    database_name = os.path.join(
        database_directory,
        experiments.de_generalist_database_name(
            run, population_size, crossover_probability, differential_weight
        ),
    )

    db = open_database_sqlite(database_name)

    df = pandas.read_sql(
        select(
            de_program.ProgramState.table,
            de_program.Population.item_table,
            de_program.Measures.table,
        ).filter(
            (
                de_program.ProgramState.table.population
                == de_program.Population.item_table.list_id
            )
            & (
                de_program.Population.item_table.measures
                == de_program.Measures.table.id
            )
        ),
        db,
    )
    lastgen = df[df.generation_index == df.generation_index.max()]
    bestrow = lastgen[lastgen.fitness == lastgen.fitness.max()]
    best_params_id = int(bestrow.genotype)

    db2 = open_async_database_sqlite(database_name)
    async with AsyncSession(db2) as session:
        best_params = await de_program.Genotype.from_db(session, best_params_id)

    return best_params, float(bestrow.fitness)


async def load_best_de_specialist(
    database_directory: str,
    run: int,
    de_params_i: int,
    body_i: int,
    ruggedness_i: int,
    bowlness_i: int,
) -> Tuple[Parameters, float]:
    population_size = DE_PARAMS[de_params_i][0]
    crossover_probability = DE_PARAMS[de_params_i][1]
    differential_weight = DE_PARAMS[de_params_i][2]

    database_name = os.path.join(
        database_directory,
        experiments.de_specialist_database_name(
            run,
            population_size,
            crossover_probability,
            differential_weight,
            body_i,
            ruggedness_i,
            bowlness_i,
        ),
    )

    db = open_database_sqlite(database_name)

    df = pandas.read_sql(
        select(
            de_program.ProgramState.table,
            de_program.Population.item_table,
            de_program.Measures.table,
        ).filter(
            (
                de_program.ProgramState.table.population
                == de_program.Population.item_table.list_id
            )
            & (
                de_program.Population.item_table.measures
                == de_program.Measures.table.id
            )
        ),
        db,
    )
    lastgen = df[df.generation_index == df.generation_index.max()]
    bestrow = lastgen[lastgen.fitness == lastgen.fitness.max()]
    best_params_id = int(bestrow.genotype)

    db2 = open_async_database_sqlite(database_name)
    async with AsyncSession(db2) as session:
        best_params = await de_program.Genotype.from_db(session, best_params_id)

    return best_params, float(bestrow.fitness)


async def load_best_graph(
    database_directory: str,
    run: int,
    graph_params_i: int,
    body_i: int,
    ruggedness_i: int,
    bowlness_i: int,
) -> Tuple[Parameters, float]:
    standard_deviation = GRAPH_PARAMS[graph_params_i][0]
    migration_probability = GRAPH_PARAMS[graph_params_i][1]

    database_name = os.path.join(
        database_directory,
        experiments.graph_database_name(run, standard_deviation, migration_probability),
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
    lastgen = df[
        (df.generation_index == df.generation_index.max())
        & (df.body_num == body_i)
        & (df.ruggedness_num == ruggedness_i)
        & (df.bowlness_num == bowlness_i)
    ]
    best_params_id = int(lastgen.genotype)

    db2 = open_async_database_sqlite(database_name)
    async with AsyncSession(db2) as session:
        best_params = await graph_program.Genotype.from_db(session, best_params_id)

    return best_params, float(lastgen.fitness)


def argparse_range_type(cliarg: str) -> List[int]:
    if cliarg.isnumeric():
        return [int(cliarg)]
    else:
        parts = cliarg.split(":")
        if len(parts) != 2 or not parts[0].isnumeric() or not parts[1].isnumeric():
            raise argparse.ArgumentTypeError(
                "Argument must have format 'number' or 'number:number'"
            )
        low = int(parts[0])
        high = int(parts[1])
        if low > high:
            raise argparse.ArgumentTypeError(
                "Argument must have format 'number' or 'number:number'"
            )
        return [i for i in range(low, high + 1)]


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "experiments",
        choices=Choices("de_generalist", "de_specialist", "graph"),
        default=[],
        nargs="*",
    )
    parser.add_argument("-d", "--database_directory", type=str, required=True)
    parser.add_argument("-r", "--runs", type=argparse_range_type, required=True)
    parser.add_argument("--body_is", type=argparse_range_type, required=True)
    parser.add_argument("--ruggedness_is", type=argparse_range_type, required=True)
    parser.add_argument("--bowlness_is", type=argparse_range_type, required=True)


async def main() -> None:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("-d", "--database_directory", type=str, required=True)
    show_parser.add_argument("-r", "--run", type=int, required=True)
    show_parser.add_argument("--env_body", type=int, required=True)
    show_parser.add_argument("--env_ruggedness", type=int, required=True)
    show_parser.add_argument("--env_bowlness", type=int, required=True)
    show_subparsers = show_parser.add_subparsers(dest="experiment", required=True)
    show_de_generalist = show_subparsers.add_parser("de_generalist")
    show_de_generalist.add_argument("--opt_params", type=int, required=True)
    show_de_specialist = show_subparsers.add_parser("de_specialist")
    show_de_specialist.add_argument("--opt_params", type=int, required=True)
    show_de_specialist.add_argument("--opt_body", type=int, required=True)
    show_de_specialist.add_argument("--opt_ruggedness", type=int, required=True)
    show_de_specialist.add_argument("--opt_bowlness", type=int, required=True)
    show_graph = show_subparsers.add_parser("graph")
    show_graph.add_argument("--opt_params", type=int, required=True)
    show_graph.add_argument("--opt_body", type=int, required=True)
    show_graph.add_argument("--opt_ruggedness", type=int, required=True)
    show_graph.add_argument("--opt_bowlness", type=int, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("-d", "--database_directory", type=str, required=True)
    verify_parser.add_argument("-r", "--runs", type=argparse_range_type, required=True)
    verify_parser.add_argument("-s", "--num_simulators", type=int, required=True)

    args = parser.parse_args()

    cpg_network_structure = make_cpg_network_structure()
    bodies, dof_maps = make_bodies()

    if args.command == "show":
        evaluator = Evaluator(
            cpg_network_structure,
            headless=False,
            num_simulators=1,
        )

        if args.experiment == "de_generalist":
            (params, opt_fitness) = await load_best_de_generalist(
                args.database_directory, args.run, args.opt_params
            )
        elif args.experiment == "de_specialist":
            (params, opt_fitness) = await load_best_de_specialist(
                args.database_directory,
                args.run,
                args.opt_params,
                args.opt_body,
                args.opt_ruggedness,
                args.opt_bowlness,
            )
        elif args.experiment == "graph":
            (params, opt_fitness) = await load_best_graph(
                args.database_directory,
                args.run,
                args.opt_params,
                args.opt_body,
                args.opt_ruggedness,
                args.opt_bowlness,
            )
        else:
            raise NotImplementedError()

        terrain = terrain_generator(
            size=TERRAIN_SIZE,
            ruggedness=RUGGEDNESS_RANGE[args.env_ruggedness],
            bowlness=BOWLNESS_RANGE[args.env_bowlness],
            granularity_multiplier=TERRAIN_GRANULARITY,
        )

        evaldescr = EvaluationDescription(
            Environment(
                bodies[args.env_body],
                dof_maps[args.env_body],
                terrain,
                name=EnvironmentName(
                    args.env_body,
                    args.env_ruggedness,
                    args.env_bowlness,
                ),
            ),
            params,
        )

        fitness = (await evaluator.evaluate([evaldescr]))[0]
        print(f"Fitness during optimization: {opt_fitness}")
        print(f"Rerun fitness: {fitness}")
        print(f"Difference: {abs(opt_fitness - fitness)}")
        print(
            "For de specialist and graph, difference should be 0 if evaluated on the same machine, and most likely be close to 0 if evaluated on a different machine."
        )
    elif args.command == "verify":
        evaluator = Evaluator(
            cpg_network_structure,
            headless=True,
            num_simulators=args.num_simulators,
        )

        @dataclass
        class Case:
            body_i: int
            ruggedness_i: int
            bowlness_i: int
            terrain: Terrain
            de_specialist_params_and_fitnesses: List[Tuple[Parameters, float]]
            graph_params_and_fitnesses: List[Tuple[Parameters, float]]

        for run in args.runs:
            cases: List[Case] = []

            for body_i in range(len(bodies)):
                for ruggedness_i in range(len(RUGGEDNESS_RANGE)):
                    for bowlness_i in range(len(BOWLNESS_RANGE)):
                        print(
                            f"Loading params for case body {body_i} ruggedness {ruggedness_i} bowlness {bowlness_i}"
                        )

                        de_specialist_params_and_fitnesses = [
                            await load_best_de_specialist(
                                args.database_directory,
                                run,
                                de_params_i,
                                body_i,
                                ruggedness_i,
                                bowlness_i,
                            )
                            for de_params_i in range(len(DE_PARAMS))
                        ]
                        graph_params_and_fitnesses = [
                            await load_best_graph(
                                args.database_directory,
                                run,
                                graph_params_i,
                                body_i,
                                ruggedness_i,
                                bowlness_i,
                            )
                            for graph_params_i in range(len(GRAPH_PARAMS))
                        ]
                        terrain = terrain = terrain_generator(
                            size=TERRAIN_SIZE,
                            ruggedness=RUGGEDNESS_RANGE[ruggedness_i],
                            bowlness=BOWLNESS_RANGE[bowlness_i],
                            granularity_multiplier=TERRAIN_GRANULARITY,
                        )
                        cases.append(
                            Case(
                                body_i,
                                ruggedness_i,
                                bowlness_i,
                                terrain,
                                de_specialist_params_and_fitnesses,
                                graph_params_and_fitnesses,
                            )
                        )
                        break
                    break

            evaldescrs: List[EvaluationDescription] = []

            for case in cases:
                for params, _ in case.de_specialist_params_and_fitnesses:
                    evaldescrs.append(
                        EvaluationDescription(
                            Environment(
                                bodies[case.body_i],
                                dof_maps[case.body_i],
                                case.terrain,
                                EnvironmentName(
                                    case.body_i, case.ruggedness_i, case.bowlness_i
                                ),
                            ),
                            params,
                        )
                    )
                for params, _ in case.graph_params_and_fitnesses:
                    evaldescrs.append(
                        EvaluationDescription(
                            Environment(
                                bodies[case.body_i],
                                dof_maps[case.body_i],
                                case.terrain,
                                EnvironmentName(
                                    case.body_i, case.ruggedness_i, case.bowlness_i
                                ),
                            ),
                            params,
                        )
                    )

            print("Starting evaluation..")
            fitnesses = await evaluator.evaluate(evaldescrs)

            fitness_i = 0
            for case in cases:
                for opt_param_i, (params, opt_fitness) in enumerate(
                    case.de_specialist_params_and_fitnesses
                ):
                    if abs(opt_fitness - fitnesses[fitness_i]) != 0.0:
                        print(
                            f"Fitness differs by {abs(opt_fitness - fitnesses[fitness_i])} gsum {sum(params)} for run {run} de specialist (param {opt_param_i}) body {case.body_i} ruggedness {case.ruggedness_i} bowlness {case.bowlness_i}"
                        )
                    fitness_i += 1
                for opt_param_i, (params, opt_fitness) in enumerate(
                    case.graph_params_and_fitnesses
                ):
                    if abs(opt_fitness - fitnesses[fitness_i]) != 0.0:
                        print(
                            f"Fitness differs by {abs(opt_fitness - fitnesses[fitness_i])} gsum {sum(params)} for run {run} graph (param {opt_param_i}) body {case.body_i} ruggedness {case.ruggedness_i} bowlness {case.bowlness_i}"
                        )
                    fitness_i += 1

            print("Done.")

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
