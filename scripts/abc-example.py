from methabc.simulate import simulate_abc, simulate
from methabc.distance import CombinedDistance, AssignmentDistance, SampledDistance
from pyabc.sampler import RedisEvalParallelSampler

import pyabc


def main():
    unif_params = {
        'init_migration_rate': (0.001, 0.1),
        'mu_driver_birth': (0, 0.01),
        's_driver_birth': (0, .5),
        'meth_rate': (0, 0.5),
        'demeth_rate': (0, 0.5)
    }

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
    )

    obs_param = {
        'meth_rate': 0.0022,
        'demeth_rate': 0.0018,
        'init_migration_rate': 0.009,
        's_driver_birth': 0.1,
        'mu_driver_birth': 0.0001,
        'deme_carrying_capacity': 100,
    }
    print("Generating synthetic data...")
    observation = simulate(obs_param)
    print("Done!")
    print("Setting up redis_sampler...")
    redis_sampler = RedisEvalParallelSampler(host="127.0.0.1", port=2666)

    # Choose one of the distance functions below:
    # distance = CombinedDistance(num_bins=20)  # Brute-force
    distance = AssignmentDistance(num_bins=20)  # Hungarian algorithm
    # distance = SampledDistance(num_bins=20, sample_size=100)  # Random sampling

    abc = pyabc.ABCSMC(
        simulate_abc,
        prior,
        distance,
        population_size=200,
        eps=pyabc.SilkOptimalEpsilon(k=10),
        sampler=redis_sampler,
    )

    abc_id = abc.new(
        db="sqlite:///" + "tmp/example.db",
        observed_sum_stat={"data": observation},
        gt_par=obs_param,
    )

    history = abc.run(max_nr_populations=10)


if __name__ == "__main__":
    main()
