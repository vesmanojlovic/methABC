from .simulate import simulate_abc
import pyabc


def simulate_10(params):
    params['deme_carrying_capacity'] = 10
    mig_rate_lower = 0.001
    mig_rate_upper = 0.1
    mig_rate_distro = pyabc.RV("uniform", mig_rate_lower, mig_rate_upper -
                    mig_rate_lower)
    params['init_migration_rate'] = mig_rate_distro.rvs()
    return simulate_abc(params)


def simulate_100(params):
    params['deme_carrying_capacity'] = 100
    mig_rate_lower = 0.0005
    mig_rate_upper = 0.1
    mig_rate_distro = pyabc.RV("uniform", mig_rate_lower, mig_rate_upper -
                    mig_rate_lower)
    params['init_migration_rate'] = mig_rate_distro.rvs()
    return simulate_abc(params)


def simulate_1000(params):
    params['deme_carrying_capacity'] = 1000
    mig_rate_lower = 0.0001
    mig_rate_upper = 0.1
    mig_rate_distro = pyabc.RV("uniform", mig_rate_lower, mig_rate_upper -
                    mig_rate_lower)
    params['init_migration_rate'] = mig_rate_distro.rvs()
    return simulate_abc(params)
