from methabc.simulate import log_model_abc
from methabc.distance import CombinedDistance
from methabc.utils import import_data
from pyabc.sampler import RedisEvalParallelSampler

import pyabc
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str,
                        help="ID of the data set in the data folder")
    args = parser.parse_args()
    data_path = args.data_path

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

    observation = import_data(data_path)

    print("Setting up redis_sampler...")
    redis_sampler = RedisEvalParallelSampler(host="127.0.0.1", port=2166)
    print("Done.")

    distance = CombinedDistance()

    abc = pyabc.ABCSMC(
        log_model_abc,
        prior,
        distance,
        population_size=1000,
        sampler=redis_sampler,
    )

    abc_id = abc.new(
        db="sqlite:///" + "test/ground_truth_test_1.db",
        observed_sum_stat={"data": observation},
    )

    history = abc.run(max_nr_populations=25, minimum_epsilon=0, min_acceptance_rate=0.03)


if __name__ == "__main__":
    main()