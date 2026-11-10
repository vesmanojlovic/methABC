from methabc.simulate import log_model_abc
from methabc.distance import total_distance, l2_term, wasserstein_term, compute_deme_matrix
from methabc.utils import import_data, side_counts, write_config, repo_dir
from pyabc.sampler import RedisEvalParallelSampler, MulticoreEvalParallelSampler

import os
import pyabc
import pandas as pd
import multiprocessing
from functools import partial
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("-d", "--data_id", type=str,
                            help="ID of the data set in the data folder")
    data_group.add_argument("--data_path", type=str,
                            help="path to a data file, e.g. from synthetic_data.py")
    parser.add_argument("--ground_truth", type=str,
                        help="parameter file from synthetic_data.py, stored in the database")
    parser.add_argument("-K", "--carrying_capacity", type=int, default=100,
                        help="deme carrying capacity")
    parser.add_argument("-t", "--turnover", type=float, default=0,
                        help="turnover after expansion, as a fraction of expansion time")
    parser.add_argument("--distance", type=str, default="total",
                        choices=["total", "adaptive"],
                        help="total: sum of L2 and Wasserstein terms; "
                             "adaptive: pyabc.AdaptiveAggregatedDistance of the two terms")
    parser.add_argument("--epsilon", type=str, default="median",
                        choices=["median", "quantile", "silk"],
                        help="pyabc epsilon schedule")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="quantile for --epsilon quantile")
    parser.add_argument("--population_size", type=int, default=1000)
    parser.add_argument("--max_populations", type=int, default=25)
    parser.add_argument("--min_acceptance_rate", type=float, default=0.03)
    parser.add_argument("-n", "--n_procs", type=int, default=0,
                        help="number of local processes (0 to use redis)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="redis server host")
    parser.add_argument("--port", type=int, default=2166,
                        help="redis server port")
    args = parser.parse_args()

    unif_params = {
        'meth_rate': (-4, -2),
        'demeth_rate': (-4, -2),
        'init_migration_rate': (-3.3, -0.5),
        'mu_driver_birth': (-5, -2),
        's_driver_birth': (0, .2),
    }

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
    )

    if args.data_path is None:
        data_id = args.data_id
        data_path = os.path.join(repo_dir, "data/individual/tumour_" + data_id + ".csv")
    else:
        data_id = os.path.splitext(os.path.basename(args.data_path))[0]
        data_path = args.data_path

    gt_par = None
    if args.ground_truth is not None:
        gt_par = pd.read_csv(args.ground_truth).iloc[0].to_dict()

    observation = import_data(data_path)

    observed_matrix = compute_deme_matrix(observation)

    left_demes, right_demes = side_counts(observation)

    run_name = f"{data_id}_K{args.carrying_capacity}_t{args.turnover}_{args.distance}_{args.epsilon}"
    run_dir = os.path.join(repo_dir, "tmp", run_name)
    os.makedirs(run_dir, exist_ok=True)
    config_template_path = write_config(
        params={
            "deme_carrying_capacity": args.carrying_capacity,
            "turnover": args.turnover,
        },
        output_dir=run_dir,
    )

    model = partial(
        log_model_abc,
        left_demes=left_demes,
        right_demes=right_demes,
        config_template_path=config_template_path,
    )

    if args.distance == "total":
        distance = total_distance
    elif args.distance == "adaptive":
        distance = pyabc.AdaptiveAggregatedDistance([l2_term, wasserstein_term])

    if args.epsilon == "median":
        eps = pyabc.MedianEpsilon()
    elif args.epsilon == "quantile":
        eps = pyabc.QuantileEpsilon(alpha=args.alpha)
    elif args.epsilon == "silk":
        eps = pyabc.SilkOptimalEpsilon()

    if args.n_procs > 0:
        multiprocessing.set_start_method("fork", force=True)
        sampler = MulticoreEvalParallelSampler(n_procs=args.n_procs)
    else:
        print("Setting up redis_sampler...")
        sampler = RedisEvalParallelSampler(host=args.host, port=args.port)
        print("Done.")

    abc = pyabc.ABCSMC(
        model,
        prior,
        distance,
        population_size=args.population_size,
        eps=eps,
        sampler=sampler,
    )

    abc_id = abc.new(
        db="sqlite:///" + os.path.join(repo_dir, "tmp", "log_model_" + run_name + ".db"),
        observed_sum_stat={"data": observation},
        gt_par=gt_par,
        meta_info={"initial_dist_matrix": observed_matrix},
    )

    history = abc.run(
        max_nr_populations=args.max_populations,
        minimum_epsilon=0,
        min_acceptance_rate=args.min_acceptance_rate,
    )


if __name__ == "__main__":
    main()
