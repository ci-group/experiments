"""Plot the average fitness over all bodies for full_generalist, full_specialist, and graph in a single plot."""

from revolve2.core.optimization.ea.openai_es import DbOpenaiESOptimizerIndividual
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


NUM_RUNS = 2
NUM_BODIES = 5
NUM_EVALS = 50000

fig, ax = plt.subplots()


def plot_full_generalist(ax: Axes) -> None:
    db_prefix = "full_generalist"

    dfs = []
    for run in range(NUM_RUNS):
        db = open_database_sqlite(f"{db_prefix}_run{run}")  # TODO multiple runs
        df = pandas.read_sql(
            select(DbOpenaiESOptimizerIndividual),
            db,
        )
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
    describe[["mean"]].plot(ax=ax, color="#0000ff")


def plot_full_specialist(ax: Axes) -> None:
    db_prefix = "full_specialist"

    dfs_runs = []
    for run in range(1):
        dfs_bodies = []
        for body_i in range(1):
            db = open_database_sqlite(
                f"{db_prefix}_body{body_i}_run{run}"
            )  # TODO multiple runs
            df = pandas.read_sql(
                select(DbOpenaiESOptimizerIndividual),
                db,
            )
            dfs_bodies.append(df[["gen_num", "fitness"]])
        all_bodies = pandas.concat(dfs_bodies)
        describe_bodies = all_bodies.groupby(by="gen_num").describe()["fitness"]
        dfs_runs.append(describe_bodies)

    all_runs = pandas.concat(dfs_runs)

    describe = all_runs.groupby(by="gen_num").describe()["mean"]
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()
    plt.fill_between(range(1, len(mean) + 1), mean - std, mean + std)
    describe[["mean"]].plot(ax=ax)


plot_full_generalist(ax=ax)
# plot_full_specialist(ax=ax)
plt.show()
