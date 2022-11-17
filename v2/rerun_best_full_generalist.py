from bodies import make_bodies, make_cpg_network_structure

from rerun_best_generic import rerun_best
import argparse


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to fetch brain parameters from.",
    )
    parser.add_argument(
        "body",
        type=int,
        help="Number of the body to simulate.",
    )
    parser.add_argument(
        "-r",
        "--record_to_directory",
        type=str,
        help="If set, videos are recorded and stored in this directory.",
    )
    parser.add_argument(
        "-p",
        "--start_paused",
        action="store_true",
        help="If set, simulation starts paused.",
    )
    args = parser.parse_args()

    await rerun_best(
        database_name=args.database,
        body=args.body,
        record_to_directory=args.record_to_directory,
        start_paused=args.start_paused,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
