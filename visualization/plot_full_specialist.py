"""Plot seperate lines for all bodies from the full_specialist runs."""

from revolve2.core.optimization.ea.openai_es import DbOpenaiESOptimizerIndividual
from bodies import make_bodies
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import matplotlib.pyplot as plt

db_prefix = "full_specialist"
num_bodies = 1  # len(make_bodies()[0])
PROCESS_ID = 0

for body_i in range(num_bodies):
    db = open_database_sqlite(f"{db_prefix}_body{body_i}_run{0}")
    df = pandas.read_sql(
        select(DbOpenaiESOptimizerIndividual).filter(
            DbOpenaiESOptimizerIndividual.process_id == PROCESS_ID
        ),
        db,
    )

    describe = df.groupby(by="gen_num").describe()["fitness"]
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()
    describe[["mean"]].plot()

plt.show()
