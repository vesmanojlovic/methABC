from methabc.simulate import simulate_abc, simulate
from methabc.distance import l2_distance
import pyabc

def main():
    halfnorm_params = {
	    'meth_rate': 0.01,
	    'demeth_rate': 0.01,
	    'init_migration_rate': 0.001,
	    'mu_driver_birth': 0.0001,
	    's_driver_birth': 0.1
    }
    discrete_param = (50, 100, 500)
    prior = pyabc.Distribution(
        **{key: pyabc.RV("halfnorm", scale=value) for key, value in halfnorm_params.items()},
        deme_carrying_capacity=pyabc.RV("rv_discrete", values=(discrete_param, [1 / len(discrete_param)] * len(discrete_param)))
    )
    abc = pyabc.ABCSMC(
        simulate_abc,
        prior,
        l2_distance,
        population_size=500,
    )
    # initialise observation as initial simulation
    obs_param = prior.rvs()
    observation = simulate(obs_param)
    abc_id = abc.new(
        "sqlite:///" + "tmp/test.db", {"data": observation}
    )
    history = abc.run(max_nr_populations=10, minimum_epsilon=0.05)


if __name__ == "__main__":
    main()
