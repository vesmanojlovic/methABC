from methabc.simulate import simulate_abc
from methabc.distance import distance
import pyabc

def main():
    unif_params = {
	    'meth_rate': (0.005, 0.1),
	    'demeth_rate': (0.005, 0.1),
	    'init_migration_rate': (0.00001, 0.001),
	    'mu_driver_birth': (0.00001, 0.0001),
    }
    K_prior = (50, 100, 500)
    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a) for key, (a, b) in unif_params.items()},
        s_driver_birth=pyabc.RV("halfnorm", scale=0.05),
        deme_carrying_capacity=pyabc.RV("rv_discrete", values=(K_prior, [1 / len(K_prior)] * len(K_prior)))
    )
    abc = pyabc.ABCSMC(
	simulate_abc,
	prior,
	distance,
	population_size=100,
    )
    abc_id = abc.new(
	"sqlite:///" + "../tmp/test.db", {"data": observation}
    )
    history = abc.run(max_nr_populations=10, minimum_epsilon=0.1)


if __name__ == "__main__":
    main()
