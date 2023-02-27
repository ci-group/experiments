from partition import partition as make_partitions
from make_graph import make_graph
import csv
import os

outdir = "partitions"

graph, envs = make_graph()

for num_partitions in [1, 5, 25, 125]:
    psize = 125 // num_partitions
    with open(os.path.join(outdir, f"psize{psize}.csv"), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["partition_num", "node_num"])
        partitions = make_partitions(graph, envs, num_partitions)
        for i, part in enumerate(partitions):
            for node in part.nodes:
                csv_out.writerow((str(i), str(node.index)))
