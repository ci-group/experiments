from graph import Graph
import os
from make_graph import make_random_graph
import csv

outdir = "random_graph"


def export_graph(file: str, graph: Graph) -> None:
    rows = [
        (node.index, neighbor.index)
        for node in graph.nodes
        for neighbor in node.neighbours
    ]

    with open(os.path.join(outdir, f"graph.csv"), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["node", "neighbor"])
        for row in rows:
            csv_out.writerow(row)


export_graph(outdir, make_random_graph()[0])
