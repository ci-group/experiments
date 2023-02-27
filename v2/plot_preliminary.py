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
    CMAES_PARAMS,
)
from bodies import make_bodies
from typing import List
import math
import de_program
import experiments
import argparse
import os
import graph_program
import random
from labellines import labelLines
from make_graph import make_graph
from experiments import de_database_name, cmaes_database_name
import cmaes_program

num_bodies = len(make_bodies()[0])

fig, ax = plt.subplots()


def plot_de(ax: Axes, database_directory: str, runs: List[int], num_envs: int) -> None:
    (
        population_size,
        crossover_probability,
        differential_weight,
        partition_size,
    ) = DE_PARAMS[1]

    dfs_per_run_per_partition: List[List[pandas.DataFrame]] = []
    for run in runs:
        dfs_per_partition: List[pandas.DataFrame] = []
        for partition_num in range(num_envs // partition_size):
            db = open_database_sqlite(
                os.path.join(
                    database_directory,
                    de_database_name(
                        run,
                        population_size,
                        crossover_probability,
                        differential_weight,
                        partition_size,
                        partition_num,
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
            dfs_per_partition.append(df[["generation_index", "combined_fitness"]])

        dfs_per_run_per_partition.append(dfs_per_partition)

    fitness_per_run: List[pandas.DataFrame] = []
    for run in dfs_per_run_per_partition:
        partition_maxes = []
        for partition in run:
            partition_maxes.append(
                partition.groupby(by="generation_index").max().reset_index()
            )
        combined_partition_fitnesses = (
            pandas.concat(partition_maxes)
            .groupby(by="generation_index")
            .agg({"combined_fitness": "mean"})
        )
        fitness_per_run.append(combined_partition_fitnesses.reset_index())

    # for i, run in enumerate(fitness_per_run):
    #     run.to_csv(
    #         f"results/preliminary/opt_csvs/preliminary_de_p{population_size}_cr{crossover_probability}_f{differential_weight}_psize{partition_size}_run{i}.csv"
    #     )

    fitnesses_per_run_concat = pandas.concat(fitness_per_run)

    gens = pandas.unique(
        fitnesses_per_run_concat[["generation_index"]].values.squeeze()
    )
    eval_range = [(i + 1) * NUM_EVALUATIONS // len(gens) for i, _ in enumerate(gens)]
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

    plot_color = "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])

    plt.fill_between(eval_range, mean - std, mean + std, color=f"{plot_color}33")
    describe[["mean"]].rename(
        columns={
            "mean": f"DE full specialist (p={population_size}, cr={crossover_probability}, f={differential_weight}, psize={partition_size})"
        }
    ).plot(ax=ax, color=plot_color)


def plot_cmaes(
    ax: Axes, database_directory: str, runs: List[int], num_envs: int
) -> None:
    (
        initial_std,
        partition_size,
    ) = CMAES_PARAMS[1]

    dfs_per_run_per_partition: List[List[pandas.DataFrame]] = []
    for run in runs:
        dfs_per_partition: List[pandas.DataFrame] = []
        for partition_num in range(num_envs // partition_size):
            db = open_database_sqlite(
                os.path.join(
                    database_directory,
                    cmaes_database_name(run, partition_size, partition_num),
                )
            )
            df = pandas.read_sql(
                select(
                    cmaes_program.ProgramState.table,
                    cmaes_program.Individual.table,
                    cmaes_program.Measures.table,
                ).filter(
                    (
                        cmaes_program.ProgramState.table.mean
                        == cmaes_program.Individual.table.id
                    )
                    & (
                        cmaes_program.Individual.table.measures
                        == cmaes_program.Measures.table.id
                    )
                ),
                db,
            )

            df["combined_fitness"] = df["combined_fitness"] / SIMULATION_TIME * 10
            dfs_per_partition.append(df[["generation_index", "combined_fitness"]])

        dfs_per_run_per_partition.append(dfs_per_partition)

    fitness_per_run: List[pandas.DataFrame] = []
    for run in dfs_per_run_per_partition:
        partition_maxes = []
        for partition in run:
            partition_maxes.append(
                partition.groupby(by="generation_index").max().reset_index()
            )
        combined_partition_fitnesses = (
            pandas.concat(partition_maxes)
            .groupby(by="generation_index")
            .agg({"combined_fitness": "mean"})
        )
        fitness_per_run.append(combined_partition_fitnesses.reset_index())

    # for i, run in enumerate(fitness_per_run):
    #     # run["combined_fitness"] *= -1
    #     run.to_csv(
    #         f"results/preliminary/opt_csvs/preliminary_cmaes_std{initial_std}_psize{partition_size}_run{i}.csv"
    #     )

    fitnesses_per_run_concat = pandas.concat(fitness_per_run)

    gens = pandas.unique(
        fitnesses_per_run_concat[["generation_index"]].values.squeeze()
    )
    eval_range = [(i + 1) * NUM_EVALUATIONS // len(gens) for i, _ in enumerate(gens)]
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

    plot_color = "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])

    plt.fill_between(eval_range, mean - std, mean + std, color=f"{plot_color}33")
    describe[["mean"]].rename(
        columns={
            "mean": f"CMAES full specialist (std={initial_std}, psize={partition_size})"
        }
    ).plot(ax=ax, color=plot_color)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--database_directory", type=str, required=True)
parser.add_argument("-r", "--runs", type=str, required=True)
args = parser.parse_args()
runs = experiments.parse_runs_arg(args.runs)

num_envs = len(make_graph()[1])

plot_de(
    ax=ax,
    database_directory=args.database_directory,
    runs=runs,
    num_envs=num_envs,
)
plot_cmaes(
    ax=ax,
    database_directory=args.database_directory,
    runs=runs,
    num_envs=num_envs,
)
ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Fitness (approx. cm/s)")
plt.title("Graph optimization and baseline performance")
labelLines(plt.gca().get_lines(), zorder=2.5, align=False)
plt.show()
