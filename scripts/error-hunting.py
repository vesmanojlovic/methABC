from methabc.simulate import log_model_abc
from methabc.distance import CombinedDistance
from methabc.utils import import_data
import pyabc
from pyabc.sampler import SingleCoreSampler

def main():
    data_path = "test/processed_final_demes_output.csv"

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

    distance = CombinedDistance()

    sampler = SingleCoreSampler()

    abc = pyabc.ABCSMC(
        log_model_abc,
        prior,
        distance,
        population_size=1,
        sampler=sampler
    )

    abc.new(
        db="sqlite:///test/error_hunting.db",
        observed_sum_stat={"data": observation},
    )

    history = abc.run(max_nr_populations=1, minimum_epsilon=1.0)

if __name__ == "__main__":
    main()