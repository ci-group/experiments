from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
from experiment_settings import THETA1S, THETA2S, CMAES_PARAMS
import os
import generality
import itertools

database_directory = "results/cmaes/generality"
out_dir = database_directory + "_csvs"
runs = range(0, 10)

thetas = [x for x in itertools.product(THETA1S, THETA2S)]

for run in runs:
    for (initial_std, partition_size) in CMAES_PARAMS:
        print(f"run {run} std {initial_std} psize {partition_size}")

        db = open_database_sqlite(
            os.path.join(database_directory, f"cmaes_psize{partition_size}_run{run}")
        )

        df = pandas.read_sql(
            select(generality.XMeasureTable),
            db,
        )

        df.to_csv(
            os.path.join(
                out_dir,
                f"cmaes_psize{partition_size}_run{run}_fitness.csv",
            )
        )
