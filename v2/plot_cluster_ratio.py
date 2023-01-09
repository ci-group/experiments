"""Plot the average fitness over all bodies for full_generalist, full_specialist, and graph in a single plot."""

from numpy import average
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from experiment_settings import (
    NUM_EVALUATIONS,
    SIMULATION_TIME,
    DE_PARAMS,
    GRAPH_PARAMS,
    RUGGEDNESS_RANGE,
    BOWLNESS_RANGE,
)
from bodies import make_bodies
from typing import List
import math
import de_program
import experiments
import argparse
import os
import graph_program

num_bodies = len(make_bodies()[0])

fig, ax = plt.subplots()


def plot_full_generalist(ax: Axes, database_directory: str, runs: List[int]) -> None:
    for (
        population_size,
        crossover_probability,
        differential_weight,
    ), plot_color in zip(DE_PARAMS, ["#0000ff", "#ff00aa", "#aaaaff"]):
        dfs_per_run: List[pandas.DataFrame] = []
        for run in runs:
            db = open_database_sqlite(
                os.path.join(
                    database_directory,
                    experiments.de_generalist_database_name(
                        run, population_size, crossover_probability, differential_weight
                    ),
                )
            )
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
            df["combined_fitness"] = df["combined_fitness"] / SIMULATION_TIME * 10
            dfs_per_run.append(df[["generation_index", "combined_fitness"]])

        max_per_run = []
        for df in dfs_per_run:
            max_per_run.append(df.groupby(by="generation_index").max().reset_index())
        max_concatenated = pandas.concat(max_per_run)

        gens = pandas.unique(max_concatenated[["generation_index"]].values.squeeze())
        eval_range = [
            (i + 1) * NUM_EVALUATIONS // len(gens) for i, _ in enumerate(gens)
        ]
        df_evals = pandas.DataFrame(
            {
                "generation_index": gens,
                "evaluation": eval_range,
            }
        )

        with_evals = pandas.merge(
            max_concatenated,
            df_evals,
            left_on="generation_index",
            right_on="generation_index",
            how="left",
        )[["evaluation", "combined_fitness"]]

        describe = with_evals.groupby(by="evaluation").describe()["combined_fitness"]
        mean = describe[["mean"]].values.squeeze()
        std = describe[["std"]].values.squeeze()
        plt.fill_between(eval_range, mean - std, mean + std, color=f"{plot_color}33")
        describe[["mean"]].rename(
            columns={
                "mean": f"Full generalist (p={population_size}, cr={crossover_probability}, f={differential_weight})"
            }
        ).plot(ax=ax, color=plot_color)


def sqrtfitness(x):
    return average([math.sqrt(v) for v in x]) ** 2


def plot_full_specialist(ax: Axes, database_directory: str, runs: List[int]) -> None:
    for (
        population_size,
        crossover_probability,
        differential_weight,
    ), plot_color in zip(DE_PARAMS, ["#ff0000", "#aaaa00", "#ee4466"]):
        dfs_per_run_per_body: List[List[pandas.DataFrame]] = []
        for run in runs:
            dfs_per_body: List[pandas.DataFrame] = []
            for body_i in range(num_bodies):
                for ruggedness_i, _ in enumerate(RUGGEDNESS_RANGE):
                    for bowlness_i, _ in enumerate(BOWLNESS_RANGE):
                        db = open_database_sqlite(
                            os.path.join(
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
                        )
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
                        df["combined_fitness"] = (
                            df["combined_fitness"] / SIMULATION_TIME * 10
                        )
                        dfs_per_body.append(
                            df[["generation_index", "combined_fitness"]]
                        )
            dfs_per_run_per_body.append(dfs_per_body)

        fitness_per_run: List[pandas.DataFrame] = []
        for run in dfs_per_run_per_body:
            body_maxes = []
            for body in run:
                body_maxes.append(
                    body.groupby(by="generation_index").max().reset_index()
                )
            combined_body_fitnesses = (
                pandas.concat(body_maxes)
                .groupby(by="generation_index")
                .agg({"combined_fitness": sqrtfitness})
            )
            fitness_per_run.append(combined_body_fitnesses.reset_index())
        fitnesses_per_run_concat = pandas.concat(fitness_per_run)

        gens = pandas.unique(
            fitnesses_per_run_concat[["generation_index"]].values.squeeze()
        )
        eval_range = [
            (i + 1) * NUM_EVALUATIONS // len(gens) for i, _ in enumerate(gens)
        ]
        df_evals = pandas.DataFrame(
            {
                "generation_index": gens,
                "evaluation": eval_range,
            }
        )

        with_evals = pandas.merge(
            fitnesses_per_run_concat,
            df_evals,
            left_on="generation_index",
            right_on="generation_index",
            how="left",
        )[["evaluation", "combined_fitness"]]

        describe = with_evals.groupby(by="evaluation").describe()["combined_fitness"]
        mean = describe[["mean"]].values.squeeze()
        std = describe[["std"]].values.squeeze()
        plt.fill_between(eval_range, mean - std, mean + std, color=f"{plot_color}33")
        describe[["mean"]].rename(
            columns={
                "mean": f"Full specialist (p={population_size}, cr={crossover_probability}, f={differential_weight})"
            }
        ).plot(ax=ax, color=plot_color)


def plot_graph(ax: Axes, database_directory: str, runs: List[int]) -> None:
    for (
        standard_deviation,
        migration_probability,
        alpha1,
        alpha2,
        theta1,
        theta2,
    ), plot_color in zip(GRAPH_PARAMS, ["#dddd00", "#aaddaa"]):
        fitnesses_per_run: List[pandas.DataFrame] = []
        for run in runs:
            db = open_database_sqlite(
                os.path.join(
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
            )
            df = pandas.read_sql(
                select(
                    graph_program.ProgramState.table,
                    graph_program.Population.item_table,
                    graph_program.Measures.table,
                ).filter(
                    (
                        graph_program.ProgramState.table.population
                        == graph_program.Population.item_table.list_id
                    )
                    & (
                        graph_program.Population.item_table.measures
                        == graph_program.Measures.table.id
                    )
                ),
                db,
            )[["orig_cluster_ratio", "performed_evaluations"]]

            combined_fitnesses_per_gen = (
                df.groupby(by=["performed_evaluations"])
                .agg({"orig_cluster_ratio": sqrtfitness})
                .reset_index()
            )

            fitnesses_per_run.append(combined_fitnesses_per_gen)

        fitnesses = pandas.concat(fitnesses_per_run)

        describe = fitnesses.groupby(by="performed_evaluations").describe()["orig_cluster_ratio"]
        mean = describe[["mean"]].values.squeeze()
        std = describe[["std"]].values.squeeze()
        plt.fill_between(
            fitnesses["performed_evaluations"].unique(),
            mean - std,
            mean + std,
            color=f"{plot_color}33",
        )
        describe[["mean"]].rename(
            columns={
                "mean": f"Graph optimization (std={standard_deviation}, mp={migration_probability}, a1={alpha1}, a2={alpha2}, t1={theta1}, t2={theta2})"
            }
        ).plot(ax=ax, color=plot_color)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--database_directory", type=str, required=True)
parser.add_argument("-r", "--runs", type=str, required=True)
args = parser.parse_args()
runs = experiments.parse_runs_arg(args.runs)

# plot_full_generalist(ax=ax, database_directory=args.database_directory, runs=runs)
# plot_full_specialist(ax=ax, database_directory=args.database_directory, runs=runs)
plot_graph(ax=ax, database_directory=args.database_directory, runs=runs)
ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Cluster ratio")
plt.title("Graph optimization and baseline performance")
plt.show()
