from methabc.simulate import log_model_abc
from methabc.distance import total_distance, compute_deme_matrix
from methabc.utils import import_data
from pyabc.sampler import RedisEvalParallelSampler

import pyabc
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_id", type=str,
                        help="ID of the data set in the data folder")
    args = parser.parse_args()
    data_id = args.data_id

    unif_params = {
        'meth_rate': (-4, -2),
        'demeth_rate': (-4, -2),
        'init_migration_rate': (-3.3, -1),
        'mu_driver_birth': (-5, -2),
        's_driver_birth': (0, .2),
    }

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
    )

    data_path = "data/individual/tumour_" + data_id + ".csv"

    observation = import_data(data_path)

    observed_matrix = compute_deme_matrix(observation)

    print("Setting up redis_sampler...")
    redis_sampler = RedisEvalParallelSampler(host="127.0.0.1", port=2166)
    print("Done.")

    abc = pyabc.ABCSMC(
        log_model_abc,
        prior,
        total_distance,
        population_size=1000,
        sampler=redis_sampler,
    )

    abc_id = abc.new(
        db="sqlite:///" + "tmp/log_model_" + data_id + ".db",
        observed_sum_stat={"data": observation},
        meta_info={"initial_dist_matrix": observed_matrix},
    )

    history = abc.run(max_nr_populations=25, minimum_epsilon=0, min_acceptance_rate=0.03)


if __name__ == "__main__":
    main()
