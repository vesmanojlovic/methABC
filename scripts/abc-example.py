from methabc.simulate import simulate_abc, simulate
from methabc.demo_distance import l2_distance, overall_wasserstein, compute_deme_matrix
from pyabc.sampler import RedisEvalParallelSampler

import pyabc
import numpy as np

def main():
    unif_params = {
        'meth_rate': (0, 0.01),
        'demeth_rate': (0, 0.01),
        'init_migration_rate': (0, 0.001),
        'mu_driver_birth': (0, 0.0001),
        's_driver_birth': (0, .5),
    }

    discrete_domain = np.arange(2, 200)

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
        deme_carrying_capacity=pyabc.RV(
		"rv_discrete",
		values=(discrete_domain, [1 / len(discrete_domain)] * len(discrete_domain)),
		),
    )

    obs_param = {
	'meth_rate': 0.003,
	'demeth_rate': 0.002,
	'init_migration_rate': 0.0001,
	's_driver_birth': 0.1,
	'mu_driver_birth': 0.0001,
	'deme_carrying_capacity': 100,
    }
    print("Generating synthetic data...")
    observation = simulate(obs_param)
    observed_matrix = compute_deme_matrix(observation)
    print("Done!")
    print("Setting up redis_sampler...")
    redis_sampler = RedisEvalParallelSampler(host="127.0.0.1", port=2666)

    transition = pyabc.AggregatedTransition(
	mapping={
	    'p_discrete': pyabc.DiscreteJumpTransition(
	    domain=discrete_domain,
	    ),
	    'p_continuous': pyabc.MultivariateNormalTransition(),
	}
    )

    distance = pyabc.AdaptiveAggregatedDistance([l2_distance, overall_wasserstein])

    abc = pyabc.ABCSMC(
	simulate_abc,
	prior,
	distance,
	population_size=200,
	eps=pyabc.SilkOptimalEpsilon(k=8),
	transitions=transition,
	sampler=redis_sampler,
    )

    abc_id = abc.new(
	db="sqlite:///" + "tmp/example.db",
	observed_sum_stat={"data": observation},
	gt_par=obs_param,
	meta_info={"initial_dist_matrix": observed_matrix},
    )

    history = abc.run(max_nr_populations=10)


if __name__ == "__main__":
    main()
