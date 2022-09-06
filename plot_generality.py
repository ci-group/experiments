from measure_generality import DbFitness
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import matplotlib.pyplot as plt

DB_NAME = "dbs_isaac/final_fitnesses"
NUM_BODIES = 5
NUM_RUNS = 2


async def main() -> None:
    db = open_database_sqlite(DB_NAME)
    df = pandas.read_sql(
        select(DbFitness),
        db,
    )

    avgedruns = (
        df[["body", "brain_name", "fitness"]]
        .groupby(["body", "brain_name"])
        .mean()
        .reset_index()
    )

    grouped = avgedruns[["brain_name", "fitness"]].groupby(
        ["brain_name"], as_index=False
    )

    mean = grouped.mean()
    std = grouped.std()

    avgedruns[["brain_name", "fitness"]].boxplot(by="brain_name", rot=10)
    plt.suptitle(None)
    plt.title("Aggregate of fitnesses in all environments")
    plt.show()
    # plt.savefig("generalist_measure.png")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
