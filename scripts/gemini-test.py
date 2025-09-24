
import pyabc
import os
import sys
from argparse import ArgumentParser

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from methabc.simulate import log_model_abc
from methabc.distance import CombinedDistance
from methabc.utils import import_data
from pyabc.sampler import RedisEvalParallelSampler


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str,
                        default="~/methabc/test/processed_final_demes_output.csv",
                        help="Path to the ground truth data file.")
    parser.add_argument("--db_path", type=str,
                        default="gemini_test.db",
                        help="Path for the new ABC database.")
    parser.add_argument("--redis_host", type=str, default="10.10.0.21", help="Redis server host.")
    parser.add_argument("--redis_port", type=int, default=2166, help="Redis server port.")

    args = parser.parse_args()

    # --- Setup ---
    db_path_sqlite = "sqlite:///" + args.db_path
    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    # --- Load Observed Data ---
    observation = import_data(args.data_path)

    # --- PyABC Setup ---
    unif_params = {
        'meth_rate': (-4, -2),
        'demeth_rate': (-4, -2),
        'init_migration_rate': (-3.3, -1),
        'mu_driver_birth': (-5, -2),
        's_driver_birth': (0, .5),
    }

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
    )

    # --- Redis Sampler ---
    print(f"Setting up redis_sampler on {args.redis_host}:{args.redis_port}...")
    redis_sampler = RedisEvalParallelSampler(host=args.redis_host, port=args.redis_port)
    print("Done.")

    distance = CombinedDistance()

    abc = pyabc.ABCSMC(
        log_model_abc,
        prior,
        distance,
        population_size=1000,
        sampler=redis_sampler,
    )

    # --- Run ABC-SMC ---
    abc.new(
        db=db_path_sqlite,
        observed_sum_stat={"data": observation},
    )

    print(f"ABC-SMC run started. DB will be at: {db_path_sqlite}")
    print(f"Connect workers to redis://{args.redis_host}:{args.redis_port}")

    abc.run(max_nr_populations=25, minimum_epsilon=0, min_acceptance_rate=0.01)

    print("ABC-SMC run finished.")
    print(f"Final results and history saved to: {db_path_sqlite}")


if __name__ == "__main__":
    main()
