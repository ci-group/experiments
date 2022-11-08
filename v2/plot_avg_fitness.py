"""Plot the average fitness over all bodies for full_generalist, full_specialist, and graph in a single plot."""

from numpy import average
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import de_multi_body_optimizer
import graph_optimizer
from experiment_settings import (
    NUM_RUNS,
    NUM_EVALUATIONS,
    SIMULATION_TIME,
    DE_PARAMS,
    GRAPH_PARAMS,
)
from bodies import make_bodies
from typing import List
import math

num_bodies = len(make_bodies()[0])

fig, ax = plt.subplots()


def plot_full_generalist(ax: Axes) -> None:
    db_prefix = "dbs/full_generalist"

    for (population_size, crossover_probability, differential_weight), (
        colora,
        colorb,
    ) in zip(DE_PARAMS, [("#aaaaff", "#0000ff"), ("#ff00ff", "#ff00aa")]):
        dfs_per_run: List[pandas.DataFrame] = []
        for run in range(NUM_RUNS):
            db = open_database_sqlite(
                f"{db_prefix}_p{population_size}_cr{crossover_probability}_f{differential_weight}_run{run}"
            )
            df = pandas.read_sql(
                select(
                    de_multi_body_optimizer.ProgramState.table,
                    de_multi_body_optimizer.Population.item_table,
                    de_multi_body_optimizer.Measures.table,
                ).filter(
                    (
                        de_multi_body_optimizer.ProgramState.table.population
                        == de_multi_body_optimizer.Population.item_table.list_id
                    )
                    & (
                        de_multi_body_optimizer.Population.item_table.measures
                        == de_multi_body_optimizer.Measures.table.id
                    )
                ),
                db,
            )
            df["fitness"] = df["fitness"] / SIMULATION_TIME * 10
            dfs_per_run.append(df[["generation_index", "fitness"]])

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
        )[["evaluation", "fitness"]]

        describe = with_evals.groupby(by="evaluation").describe()["fitness"]
        mean = describe[["mean"]].values.squeeze()
        std = describe[["std"]].values.squeeze()
        plt.fill_between(eval_range, mean - std, mean + std, color=colora)
        describe[["mean"]].rename(
            columns={
                "mean": f"Full generalist (p{population_size}_cr{crossover_probability}_f{differential_weight})"
            }
        ).plot(ax=ax, color=colorb)


def sqrtfitness(x):
    return average([math.sqrt(v) for v in x]) ** 2


def plot_full_specialist(ax: Axes) -> None:
    db_prefix = "dbs/full_specialist"

    for (population_size, crossover_probability, differential_weight), (
        colora,
        colorb,
    ) in zip(DE_PARAMS, [("#ffaaaa", "#ff0000"), ("#ffff00", "#aaaa00")]):
        dfs_per_run_per_body: List[List[pandas.DataFrame]] = []
        for run in range(NUM_RUNS):
            dfs_per_body: List[pandas.DataFrame] = []
            for body_i in range(num_bodies):
                db = open_database_sqlite(
                    f"{db_prefix}_p{population_size}_cr{crossover_probability}_f{differential_weight}_body{body_i}_run{run}"
                )
                df = pandas.read_sql(
                    select(
                        de_multi_body_optimizer.ProgramState.table,
                        de_multi_body_optimizer.Population.item_table,
                        de_multi_body_optimizer.Measures.table,
                    ).filter(
                        (
                            de_multi_body_optimizer.ProgramState.table.population
                            == de_multi_body_optimizer.Population.item_table.list_id
                        )
                        & (
                            de_multi_body_optimizer.Population.item_table.measures
                            == de_multi_body_optimizer.Measures.table.id
                        )
                    ),
                    db,
                )
                df["fitness"] = df["fitness"] / SIMULATION_TIME * 10
                dfs_per_body.append(df[["generation_index", "fitness"]])
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
                .agg({"fitness": sqrtfitness})
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
        )[["evaluation", "fitness"]]

        describe = with_evals.groupby(by="evaluation").describe()["fitness"]
        mean = describe[["mean"]].values.squeeze()
        std = describe[["std"]].values.squeeze()
        plt.fill_between(eval_range, mean - std, mean + std, color=colora)
        describe[["mean"]].rename(
            columns={
                "mean": f"Full specialist (p{population_size}_cr{crossover_probability}_f{differential_weight})"
            }
        ).plot(ax=ax, color=colorb)


def plot_graph(ax: Axes) -> None:
    db_prefix = "dbs/graph"

    for (standard_deviation), (
        colora,
        colorb,
    ) in zip(GRAPH_PARAMS, [("#aaaa00", "#dddd00")]):
        fitnesses_per_run: List[pandas.DataFrame] = []
        for run in range(NUM_RUNS):
            # db = open_database_sqlite(f"{db_prefix}_s{standard_deviation}_run{run}")
            db = open_database_sqlite(f"dbg_graph")
            df = pandas.read_sql(
                select(
                    graph_optimizer.ProgramState.table,
                    graph_optimizer.Population.item_table,
                    graph_optimizer.Measures.table,
                ).filter(
                    (
                        graph_optimizer.ProgramState.table.population
                        == graph_optimizer.Population.item_table.list_id
                    )
                    & (
                        graph_optimizer.Population.item_table.measures
                        == graph_optimizer.Measures.table.id
                    )
                ),
                db,
            )[["fitness", "performed_evaluations"]]
            df["fitness"] = df["fitness"] / SIMULATION_TIME * 10

            combined_fitnesses_per_gen = (
                df.groupby(by=["performed_evaluations"])
                .agg({"fitness": sqrtfitness})
                .reset_index()
            )

            fitnesses_per_run.append(combined_fitnesses_per_gen)

        fitnesses = pandas.concat(fitnesses_per_run)

        describe = fitnesses.groupby(by="performed_evaluations").describe()["fitness"]
        mean = describe[["mean"]].values.squeeze()
        std = describe[["std"]].values.squeeze()
        # plt.fill_between(eval_range, mean - std, mean + std, color=colora)
        describe[["mean"]].rename(
            columns={"mean": f"Graph optimization (s{standard_deviation})"}
        ).plot(ax=ax, color=colorb)


plot_full_generalist(ax=ax)
plot_full_specialist(ax=ax)
plot_graph(ax=ax)
ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Fitness (approx. cm/s)")
plt.title("Graph optimization and baseline performance")
plt.show()
