from full_gen_spec import run_full_gen_spec
from bodies import make_bodies, make_cpg_network_structure
from experiment_settings import NUM_EVALUATIONS, NUM_RUNS, DE_PARAMS

SEED_BASE = 23678400


async def main() -> None:
    # await run_all_full_generalist_runs()  # For the actual experiments
    await run_full_generalist(
        f"dbg_full_generalist", True, 0, 100, 0.9, 0.8
    )  # For debugging only


async def run_all_full_generalist_runs() -> None:
    for (population_size, crossover_probability, differential_weight) in DE_PARAMS:
        for i in range(NUM_RUNS):
            await run_full_generalist(
                f"dbs/full_generalist_cr{crossover_probability}f{differential_weight}_run{i}",
                True,
                SEED_BASE + i,
                population_size=population_size,
                crossover_probability=crossover_probability,
                differential_weight=differential_weight,
            )


async def run_full_generalist(
    database_name: str,
    headless: bool,
    rng_seed: int,
    population_size: float,
    crossover_probability: float,
    differential_weight: float,
) -> None:
    bodies, dof_maps = make_bodies()
    cpg_network_structure = make_cpg_network_structure()

    await run_full_gen_spec(
        database_name=database_name,
        bodies=bodies,
        dof_maps=dof_maps,
        cpg_network_structure=cpg_network_structure,
        headless=headless,
        rng_seed=rng_seed,
        num_evaluations=NUM_EVALUATIONS,
        population_size=population_size,
        crossover_probability=crossover_probability,
        differential_weight=differential_weight,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
