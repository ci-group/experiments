from full_gen_spec import run_full_gen_spec
from bodies import make_bodies


async def main() -> None:
    # run_all_full_generalist_runs() # For the actual experiments
    run_full_generalist(f"dbg_full_generalist", True, 0)  # For debugging only


async def run_all_full_generalist_runs() -> None:
    NUM_RUNS = 20
    for i in range(NUM_RUNS):
        run_full_generalist(f"full_generalist_run{i}", False, i)


async def run_full_generalist(
    database_name: str, headless: bool, rng_seed: int
) -> None:
    bodies, dof_maps = make_bodies()

    run_full_gen_spec(database_name, bodies, dof_maps, cpg_network, headless, rng_seed)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
