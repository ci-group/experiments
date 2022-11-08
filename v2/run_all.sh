python $(dirname "$0")/run_full_generalist.py
python $(dirname "$0")/run_full_specialist.py
seq $(cd $(dirname "$0"); python -c"from experiment_settings import NUM_RUNS;print(NUM_RUNS)") | parallel -j 60 --workdir $PWD python $(dirname "$0")/run_graph_generalist.py {}