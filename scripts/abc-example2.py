from methabc.simulate import simulate_abc, simulate
from methabc.distance import distance, compute_deme_matrix

import pyabc
import numpy as np

def main():
    unif_params = {
        'meth_rate': (0, 0.01),
        'demeth_rate': (0, 0.01),
        'init_migration_rate': (0, 0.001),
        'mu_driver_birth': (0, 0.0001),
        's_driver_birth': (0, 1),
    }

    discrete_domain = np.arange(2, 1001)
    
    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
        deme_carrying_capacity=pyabc.RV(
		"rv_discrete", 
		values=(discrete_domain, [1 / len(discrete_domain)] * len(discrete_domain)),
		),
    )

    obs_param = {
	'meth_rate': 0.005,
	'demeth_rate': 0.005,
	'init_migration_rate': 0.0005,
	's_driver_birth': 0.1,
	'mu_driver_birth': 0.00005,
	'deme_carrying_capacity': 120,
    }
    print("Generating synthetic data...")
    observation = simulate(obs_param)
    observed_matrix = compute_deme_matrix(observation)
    print("Done!")

    transition = pyabc.AggregatedTransition(
	mapping={
	    'p_discrete': pyabc.DiscreteJumpTransition(
	    domain=discrete_domain, p_stay=0
	    ),
	    'p_continuous': pyabc.MultivariateNormalTransition(),
	}
    )

    abc = pyabc.ABCSMC(
	simulate_abc,
	prior,
	distance,
	population_size=50,
	eps=pyabc.SilkOptimalEpsilon(k=10),
	transitions=transition,
    )

    abc_id = abc.new(
	db="sqlite:///" + "tmp/example2.db",
	observed_sum_stat={"data": observation},
	gt_par=obs_param,
	meta_info={"initial_dist_matrix": observed_matrix},
    )

    history = abc.run(max_nr_populations=5)


if __name__ == "__main__":
    main()