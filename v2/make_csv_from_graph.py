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

database_directory = "dbs_finalgraph_dec16"
run = 0
standard_deviation = 0.05
migration_probability = 0.5
alpha1 = 1.0
alpha2 = 8.0
theta1 = 1.0
theta2 = 1.0

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
    f"{experiments.graph_database_name(run, standard_deviation, migration_probability, alpha1, alpha2, theta1, theta2)}_genotype.csv"
)

pivoted2 = df[["generation_index", "index", "fitness"]].pivot(
    index="generation_index", columns="index", values="fitness"
)

pivoted2.to_csv(
    f"{experiments.graph_database_name(run, standard_deviation, migration_probability, alpha1, alpha2, theta1, theta2)}_fitness.csv"
)
