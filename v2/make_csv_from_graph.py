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
import graph_program
import itertools

database_directory = "results/graph/opt"
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
            select(
                graph_program.ProgramState.table,
                graph_program.Population.item_table,
                graph_program.GenotypeWithMeta.table.id,
                graph_program.GenotypeWithMeta.table.genotype.label("genotype_id"),
                graph_program.Measures.table,
            ).filter(
                (
                    graph_program.ProgramState.table.population
                    == graph_program.Population.item_table.list_id
                )
                & (
                    graph_program.Population.item_table.genotype
                    == graph_program.GenotypeWithMeta.table.id
                )
                & (
                    graph_program.Population.item_table.measures
                    == graph_program.Measures.table.id
                )
            ),
            db,
        )

        pivoted = df[["generation_index", "index", "genotype_id"]].pivot(
            index="generation_index", columns="index", values="genotype_id"
        )

        pivoted.to_csv(
            os.path.join(
                out_dir,
                f"{experiments.graph_database_name(run, standard_deviation, migration_probability, alpha1, alpha2, theta1, theta2)}_genotype.csv",
            )
        )

        pivoted2 = df[["generation_index", "index", "fitness"]].pivot(
            index="generation_index", columns="index", values="fitness"
        )

        pivoted2.to_csv(
            os.path.join(
                out_dir,
                f"{experiments.graph_database_name(run, standard_deviation, migration_probability, alpha1, alpha2, theta1, theta2)}_fitness.csv",
            )
        )

        df2 = pandas.read_sql(
            select(graph_program.Parameters.item_table),
            db,
        )
        pivoted3 = df2.pivot(index="parameters_id", columns="index", values="parameter")
        pivoted3.to_csv(
            os.path.join(
                out_dir,
                f"{experiments.graph_database_name(run, standard_deviation, migration_probability, alpha1, alpha2, theta1, theta2)}_parameters.csv",
            )
        )
