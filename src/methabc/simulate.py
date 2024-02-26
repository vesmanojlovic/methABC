import pandas as pd
import numpy as np
import subprocess
from io import BytesIO
import scipy.stats as ss

def simulate(
    meth_rate=0.005,
    demeth_rate=0.005,
    init_migration_rate=0.0001,
    mu_driver_birth=0.00001,
    deme_carrying_capacity=100,
    s_driver_birth=0.1,
    seed=None,
    model_path='../resources/methdemon/bin/methdemon',
    config_template_path='../resources/config_example.dat',
    sim_config_path='../tmp/tmp_config.dat',
):
    """
    Run the methdemon simulations defined
    in an external C++ binary.

    Returns: pandas.DataFrame
        Columns: Generations, Identity, Parent, OriginTime, AverageArray
        Generations: Simulation time of writing the line
        Identity: deme identity (ordinal number)
        Parent: identity of parent deme
        OriginTime: generation at which deme split
        AverageArray: fCpG array of the deme
    """
    if seed is None:
        seed = np.random.randint(2**15 - 2) + 1

    #params = prior.rvs()
    #params[seed] = seed

    params = {
       'meth_rate': meth_rate,
        'demeth_rate': demeth_rate,
        'init_migration_rate': init_migration_rate,
        'mu_driver_birth': mu_driver_birth,
        's_driver_birth': s_driver_birth,
        'deme_carrying_capacity': deme_carrying_capacity,
        'seed': seed,
    }
    output_dir, sim_config = write_config(
        params=params,
        config_template_path=config_template_path,
        output_path=sim_config_path,
    )
    
    re = subprocess.run(
        [
            model_path,
            output_dir,
            sim_config,
        ],
        stdout=subprocess.PIPE,
    )

    df = pd.read_table(
        BytesIO(re.stdout),
        delimiter=',',
        header=0,
    )
    df.AverageArray = df.AverageArray.apply(lambda x: np.fromstring(x, sep=';'))
    df = df[np.floor(df['Generation'])==np.floor(df.Generation.max())]

    return df

def simulate_abc(params):
    res = simulate(**params)
    return {"data": res}
