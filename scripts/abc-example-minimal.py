
import pyabc
import os
import sys
from argparse import ArgumentParser

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from methabc.simulate import log_model_abc, simulate
from methabc.distance import CombinedDistance, AssignmentDistance, SampledDistance
from methabc.utils import import_data
from pyabc.sampler import RedisEvalParallelSampler


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str,
                        default=None,
                        help="Path to the ground truth data file. If not provided, synthetic data will be generated.")
    parser.add_argument("--db_path", type=str,
                        default="abc_example_minimal.db",
                        help="Path for the new ABC database.")
    parser.add_argument("--redis_host", type=str, default="10.10.0.21", help="Redis server host.")
    parser.add_argument("--redis_port", type=int, default=2166, help="Redis server port.")
    parser.add_argument("--distance", type=str, default="sampled",
                        help="Distance function to use. One of: combined, assignment, sampled.")

    args = parser.parse_args()

    # --- Setup ---
    db_path_sqlite = "sqlite:///" + args.db_path
    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    # --- Load Observed Data ---
    if args.data_path:
        print(f"Loading observed data from {args.data_path}...")
        observation = import_data(args.data_path)
    else:
        print("Generating synthetic data...")
        obs_param = {
            'meth_rate': 0.0022,
            'demeth_rate': 0.0018,
            'init_migration_rate': 0.009,
            's_driver_birth': 0.1,
            'mu_driver_birth': 0.0001,
            'deme_carrying_capacity': 100,
        }
        observation = simulate(obs_param)
    print("Done!")

    # --- PyABC Setup ---
    unif_params = {
        'meth_rate': (-4, -2),
        'demeth_rate': (-4, -2),
        'init_migration_rate': (-3.3, -1),
        'mu_driver_birth': (-5, -2),
        's_driver_birth': (0, .4),
    }

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
    )

    # --- Redis Sampler ---
    print(f"Setting up redis_sampler on {args.redis_host}:{args.redis_port}...")
    redis_sampler = RedisEvalParallelSampler(host=args.redis_host, port=args.redis_port)
    print("Done.")

    # --- Distance Function ---
    if args.distance == "combined":
        distance = CombinedDistance(num_bins=20)
    elif args.distance == "assignment":
        distance = AssignmentDistance(num_bins=20)
    elif args.distance == "sampled":
        distance = SampledDistance(num_bins=20, sample_size=10)
    else:
        raise ValueError(f"Unknown distance function: {args.distance}")
    print(f"Using {args.distance} distance function.")

    abc = pyabc.ABCSMC(
        log_model_abc,
        prior,
        distance,
        population_size=10,
        sampler=redis_sampler,
    )

    # --- Run ABC-SMC ---
    abc.new(
        db=db_path_sqlite,
        observed_sum_stat={"data": observation},
    )

    print(f"ABC-SMC run started. DB will be at: {db_path_sqlite}")
    print(f"Connect workers to redis://{args.redis_host}:{args.redis_port}")

    abc.run(max_nr_populations=2, minimum_epsilon=0, min_acceptance_rate=0.01)

    print("ABC-SMC run finished.")
    print(f"Final results and history saved to: {db_path_sqlite}")


if __name__ == "__main__":
    main()
