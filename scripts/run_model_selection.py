from methabc.distance import l2_distance, overall_wasserstein, compute_deme_matrix
from methabc.model_selection import simulate_10, simulate_100, simulate_1000
from methabc.utils import import_data
from pyabc.sampler import RedisEvalParallelSampler

import numpy as np
import os
import pyabc


def main():
    unif_params = {
        'meth_rate': (0, 0.1),
        'demeth_rate': (0, 0.1),
        'init_migration_rate': (0, 0.01),
        'mu_driver_birth': (0, 0.001),
        's_driver_birth': (0, 1),
    }
    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
        )

    redis_sampler = RedisEvalParallelSampler(host="127.0.0.1", port=2666)
    models = [simulate_10, simulate_100, simulate_1000]
    distance = pyabc.AdaptiveAggregatedDistance([l2_distance, overall_wasserstein])

    os.makedirs("tmp", exist_ok=True)

    abc = pyabc.ABCSMC(
            models,
            [prior, prior, prior],
            distance,
            population_size=200,
            eps=pyabc.SilkOptimalEpsilon(k=10),
            sampler=redis_sampler,
            )

    db_path = "sqlite:///" + os.path.join(os.getcwd(), "tmp/run_model_selection.db")
    observation = import_data("data/individual/tumour_U.csv")
    abc_id = abc.new(
            db_path,
            observed_sum_stat={"data": observation},
            meta_info={"initial_dist_matrix": compute_deme_matrix(observation)},
            )
    history = abc.run(max_nr_populations=10)


if __name__ == "__main__":
    main()
