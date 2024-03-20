from methabc.distance import total_distance, compute_deme_matrix
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
        'mu_driver_birth': (0, 0.001),
    }

    halfnorm_params = {
        's_driver_birth': 0.05,
    }

    prior10 = pyabc.Distribution(
        init_migration_rate=pyabc.RV("uniform", 0.001, 0.1),
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
        **{key: pyabc.RV("halfnorm", a) for key, a in halfnorm_params.items()},
        )

    prior100 = pyabc.Distribution(
        init_migration_rate=pyabc.RV("uniform", 0.0005, 0.1),
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
        **{key: pyabc.RV("halfnorm", a) for key, a in halfnorm_params.items()},
        )

    prior1000 = pyabc.Distribution(
        init_migration_rate=pyabc.RV("uniform", 0.0001, 0.1),
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
        **{key: pyabc.RV("halfnorm", a) for key, a in halfnorm_params.items()},
        )
    priors = [prior10, prior100, prior1000]

    redis_sampler = RedisEvalParallelSampler(host="127.0.0.1", port=2166)
    models = [simulate_10, simulate_100, simulate_1000]

    os.makedirs("model_selection", exist_ok=True)

    abc = pyabc.ABCSMC(
            models,
            priors,
            total_distance,
            population_size=500,
            sampler=redis_sampler,
            )

    db_path = "sqlite:///" + os.path.join(os.getcwd(), "model_selection/ms_history.db")
    observation = import_data("data/individual/tumour_M.csv")
    abc_id = abc.new(
            db_path,
            observed_sum_stat={"data": observation},
            meta_info={"initial_dist_matrix": compute_deme_matrix(observation)},
            )
    history = abc.run(max_nr_populations=10)


if __name__ == "__main__":
    main()
