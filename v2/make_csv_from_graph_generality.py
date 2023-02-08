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
    THETA1S,
    THETA2S,
    GRAPH_ALPHA1,
    GRAPH_ALPHA2,
    GRAPH_STD,
    GRAPH_PMIG,
)
from bodies import make_bodies
from typing import List
import math
import de_program
import experiments
import argparse
import os
import generality
import itertools

database_directory = "results/graph/generality"
out_dir = database_directory + "_csvs"
runs = range(0, 10)
standard_deviation = GRAPH_STD
migration_probability = GRAPH_PMIG
alpha1 = GRAPH_ALPHA1
alpha2 = GRAPH_ALPHA2

thetas = [x for x in itertools.product(THETA1S, THETA2S)]

for run in runs:
    for (theta1, theta2) in thetas:
        print(f"run {run} theta1 {theta1} theta2 {theta2}")

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
            select(generality.XMeasureTable),
            db,
        )

        df.to_csv(
            os.path.join(
                out_dir,
                f"{experiments.graph_database_name(run, standard_deviation, migration_probability, alpha1, alpha2, theta1, theta2)}_fitness.csv",
            )
        )
