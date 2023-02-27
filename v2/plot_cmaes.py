"""Plot the average fitness over all bodies for full_generalist, full_specialist, and graph in a single plot."""

from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from experiment_settings import (
    NUM_EVALUATIONS,
    SIMULATION_TIME,
    CMAES_PARAMS,
)
from bodies import make_bodies
from typing import List
import experiments
import argparse
import os
import random
from labellines import labelLines
from make_graph import make_graph
from experiments import cmaes_database_name
import cmaes_program
from graph import Graph
from environment import Environment
from partition import partition as do_partition

num_bodies = len(make_bodies()[0])

fig, ax = plt.subplots()


def plot_cmaes(
    ax: Axes,
    database_directory: str,
    runs: List[int],
    graph: Graph,
    environments: List[Environment],
) -> None:
    for (initial_std, partition_size) in CMAES_PARAMS:
        num_partitions = len(environments) // partition_size

        dfs_per_run_per_partition: List[List[pandas.DataFrame]] = []
        for run in runs:
            dfs_per_partition: List[pandas.DataFrame] = []
            for partition_num in range(num_partitions):
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

        lowest_last_gen = min(
            [df["generation_index"].max() for df in dfs_per_run_per_partition[0]]
        )

        fitness_per_run: List[pandas.DataFrame] = []
        for run in dfs_per_run_per_partition:
            partition_maxes = []
            for partition in run:
                partition_maxes.append(
                    partition[partition.generation_index <= lowest_last_gen]
                    .groupby(by="generation_index")
                    .max()
                    .reset_index()
                )
            combined_partition_fitnesses = (
                pandas.concat(partition_maxes)
                .groupby(by="generation_index")
                .agg({"combined_fitness": "mean"})
            )
            fitness_per_run.append(combined_partition_fitnesses.reset_index())

        for i, run in enumerate(fitness_per_run):
            run.to_csv(f"results/cmaes/opt_csvs/cmaes_psize{partition_size}_run{i}.csv")

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

        plot_color = "#" + "".join(
            [random.choice("0123456789ABCDEF") for j in range(6)]
        )

        plt.fill_between(eval_range, mean - std, mean + std, color=f"{plot_color}33")
        describe[["mean"]].rename(
            columns={"mean": f"CMAES (psize={partition_size} std={initial_std})"}
        ).plot(ax=ax, color=plot_color)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--database_directory", type=str, required=True)
parser.add_argument("-r", "--runs", type=str, required=True)
args = parser.parse_args()
runs = experiments.parse_runs_arg(args.runs)

graph, envs = make_graph()

plot_cmaes(
    ax=ax,
    database_directory=args.database_directory,
    runs=runs,
    graph=graph,
    environments=envs,
)
ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Fitness (approx. cm/s)")
plt.title("Graph optimization and baseline performance")
labelLines(plt.gca().get_lines(), zorder=2.5, align=False)
plt.show()
