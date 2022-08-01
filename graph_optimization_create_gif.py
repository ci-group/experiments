import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from graph_generalist_optimizer import (
    DbGraphGeneralistOptimizerGraphNodeState,
    DbGenotype,
)
from revolve2.core.database import open_database_sqlite
import pandas
from sqlalchemy.future import select
import os
import PIL

bodies = ["body1", "body2"]
envs = ["flat plane"]

dbname = "dbg_graph_generalist"
db = open_database_sqlite(dbname)
process_id = 0

df = pandas.read_sql(
    select(DbGenotype, DbGraphGeneralistOptimizerGraphNodeState).filter(
        (DbGraphGeneralistOptimizerGraphNodeState.process_id == process_id)
        & (DbGraphGeneralistOptimizerGraphNodeState.genotype_id == DbGenotype.id)
    ),
    db,
)
min_id = df.genotype_id.min()
max_id = df.genotype_id.max()

figsdir = f"./{dbname}_figs"
os.makedirs(figsdir)

for gen_i in range(len(df.gen_num.value_counts())):
    gen = df[df.gen_num == gen_i]

    ids = np.array([gen.genotype_id])
    fitnesses = np.array([gen.fitness])

    fig, ax = plt.subplots()
    im = ax.imshow(ids, vmin=min_id, vmax=max_id)

    ax.set_xticks(np.arange(len(bodies)), labels=bodies)
    ax.set_yticks(np.arange(len(envs)), labels=envs)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(envs)):
        for j in range(len(bodies)):
            text = ax.text(j, i, fitnesses[i, j], ha="center", va="center", color="w")

    ax.set_title(f"Graph optimizer generation {gen_i}")
    fig.tight_layout()
    plt.savefig(os.path.join(figsdir, f"gen_{gen_i}.png"))

frames = [
    PIL.Image.open(image)
    for image in [
        os.path.join(figsdir, f"gen_{gen_i}.png")
        for gen_i in range(len(df.gen_num.value_counts()))
    ]
]
frame_one = frames[0]
frame_one.save(
    os.path.join(figsdir, "evolution.gif"),
    format="GIF",
    append_images=frames,
    save_all=True,
    duration=500,
    loop=0,
)
