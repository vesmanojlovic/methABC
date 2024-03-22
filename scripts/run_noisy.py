from methabc.simulate import noisy_abc
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
        'init_migration_rate': (0.0005, 0.1),
        'mu_driver_birth': (0, 0.01),
        's_driver_birth': (0, .2),
        'meth_rate': (0, 0.1),
        'demeth_rate': (0, 0.1)
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

    population_size=AdaptivePopulationSize(
                    start_nr_particles=1000,
                    ),

    abc = pyabc.ABCSMC(
        noisy_abc,
        prior,
        distance=total_distance,
        population_size=population_size,
        sampler=redis_sampler,
    )

    abc_id = abc.new(
        db="sqlite:///" + "tmp/inference_" + data_id + ".db",
        observed_sum_stat={"data": observation},
        meta_info={"initial_dist_matrix": observed_matrix},
    )

    history = abc.run(max_nr_populations=20, minimum_epsilon=0)


if __name__ == "__main__":
    main()
