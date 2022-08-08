"""Plot the average fitness over all bodies for full_generalist, full_specialist, and graph in a single plot."""

from numpy import average
from revolve2.core.optimization.ea.openai_es import DbOpenaiESOptimizerIndividual
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import math
from graph_generalist_optimizer import DbGraphGeneralistOptimizerGraphNodeState

NUM_RUNS = 2
NUM_BODIES = 5
NUM_EVALS = 10000
EVAL_TIME = 30

fig, ax = plt.subplots()


def plot_full_generalist(ax: Axes) -> None:
    db_prefix = "full_generalist"

    dfs = []
    for run in range(NUM_RUNS):
        db = open_database_sqlite(f"{db_prefix}_run{run}")
        df = pandas.read_sql(
            select(DbOpenaiESOptimizerIndividual),
            db,
        )
        df["fitness"] = df["fitness"] / EVAL_TIME
        dfs.append(df[["gen_num", "fitness"]])
        describe = df.groupby(by="gen_num").describe()["fitness"]

    df_runs = pandas.concat(dfs)

    gens = pandas.unique(df_runs[["gen_num"]].values.squeeze())
    eval_range = [i * NUM_EVALS // len(gens) for i, _ in enumerate(gens)]
    df_evals = pandas.DataFrame(
        {
            "gen_num": gens,
            "evaluation": eval_range,
        }
    )

    with_evals = pandas.merge(
        df_runs, df_evals, left_on="gen_num", right_on="gen_num", how="left"
    )[["evaluation", "fitness"]]

    describe = with_evals.groupby(by="evaluation").describe()["fitness"]
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()
    plt.fill_between(eval_range, mean - std, mean + std, color="#aaaaff")
    describe[["mean"]].rename(columns={"mean": "Full generalist"}).plot(
        ax=ax, color="#0000ff"
    )


def sqrtfitness(x):
    return average([math.sqrt(v) for v in x]) ** 2


def plot_full_specialist(ax: Axes) -> None:
    db_prefix = "full_specialist"

    combined_body_fitnesses_per_run = []
    for run in range(NUM_RUNS):
        seperate_body_fitnesses = []
        for body_i in range(NUM_BODIES):
            db = open_database_sqlite(f"{db_prefix}_body{body_i}_run{run}")
            individuals = pandas.read_sql(
                select(DbOpenaiESOptimizerIndividual),
                db,
            )
            individuals["fitness"] = individuals["fitness"] / EVAL_TIME
            fitness_avged = (
                individuals[["gen_num", "fitness"]]
                .groupby(by="gen_num")
                .agg({"fitness": "mean"})
            )
            seperate_body_fitnesses.append(fitness_avged)
        seperate_body_fitnesses_df = pandas.concat(seperate_body_fitnesses)
        combined_body_fitnesses = seperate_body_fitnesses_df.groupby(by="gen_num").agg(
            {"fitness": sqrtfitness}
        )
        combined_body_fitnesses_per_run.append(combined_body_fitnesses)

    fitnesses_per_run = pandas.concat(combined_body_fitnesses_per_run)
    fitnesses_per_run.reset_index(inplace=True)

    gens = pandas.unique(fitnesses_per_run[["gen_num"]].values.squeeze())
    eval_range = [i * NUM_EVALS // len(gens) for i, _ in enumerate(gens)]
    df_evals = pandas.DataFrame(
        {
            "gen_num": gens,
            "evaluation": eval_range,
        }
    )

    with_evals = pandas.merge(
        fitnesses_per_run, df_evals, left_on="gen_num", right_on="gen_num", how="left"
    )[["evaluation", "fitness"]]

    describe = with_evals.groupby(by="evaluation").describe()["fitness"]
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()
    plt.fill_between(eval_range, mean - std, mean + std, color="#ffaaaa")
    describe[["mean"]].rename(columns={"mean": "Full specialist"}).plot(
        ax=ax, color="#ff0000"
    )


def plot_graph(ax: Axes) -> None:
    db_prefix = "graph_generalist"

    combined_body_fitnesses_per_run = []
    for run in range(NUM_RUNS):
        db = open_database_sqlite(f"{db_prefix}_run{run}")
        seperate_body_fitnesses = pandas.read_sql(
            select(DbGraphGeneralistOptimizerGraphNodeState),
            db,
        )[["gen_num", "graph_index", "fitness"]]
        seperate_body_fitnesses["fitness"] = (
            seperate_body_fitnesses["fitness"] / EVAL_TIME
        )
        combined_body_fitnesses = seperate_body_fitnesses.groupby(by=["gen_num"]).agg(
            {"fitness": sqrtfitness}
        )
        combined_body_fitnesses_per_run.append(combined_body_fitnesses)

    fitnesses_per_run = pandas.concat(combined_body_fitnesses_per_run)
    fitnesses_per_run.reset_index(inplace=True)

    gens = pandas.unique(fitnesses_per_run[["gen_num"]].values.squeeze())
    eval_range = [i * NUM_EVALS // len(gens) for i, _ in enumerate(gens)]
    df_evals = pandas.DataFrame(
        {
            "gen_num": gens,
            "evaluation": eval_range,
        }
    )

    with_evals = pandas.merge(
        fitnesses_per_run, df_evals, left_on="gen_num", right_on="gen_num", how="left"
    )[["evaluation", "fitness"]]

    describe = with_evals.groupby(by="evaluation").describe()["fitness"]
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()
    plt.fill_between(eval_range, mean - std, mean + std, color="#dddd00")
    describe[["mean"]].rename(columns={"mean": "Graph optimization"}).plot(
        ax=ax, color="#aaaa00"
    )


plot_full_generalist(ax=ax)
plot_full_specialist(ax=ax)
plot_graph(ax=ax)
ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Fitness (approx. m/s)")
plt.title("Graph optimization and baseline performance")
plt.show()
