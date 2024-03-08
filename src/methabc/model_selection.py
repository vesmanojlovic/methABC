from .simulate import simulate_abc


def simulate_10(params):
    params['deme_carrying_capacity'] = 10
    return simulate_abc(params)


def simulate_100(params):
    params['deme_carrying_capacity'] = 100
    return simulate_abc(params)


def simulate_1000(params):
    params['deme_carrying_capacity'] = 1000
    return simulate_abc(params)
