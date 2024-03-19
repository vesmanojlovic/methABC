import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .utils import write_config


def simulate(
        params,
    ):
    """
    Simulate data using the methdemon model.
    Args:
        params: parameters to use in simulation, drawn from prior
    Returns:
        df: pd.DataFrame with the final output of the simulation
    """
    seed = np.random.randint(2**15 - 2) + 1
    if seed==66:
        seed += 1

    # temporary directory to store the config and outputs
    tempdir = tempfile.mkdtemp()
    config_path = write_config(params=params, output_dir=tempdir)
    model_path = "resources/methdemon/bin/methdemon"

    config_dir, config_name = os.path.split(config_path)

    try:
        result = subprocess.run(
            [model_path, config_dir, config_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True  # Forces an exception if the command fails
        )
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with: {e.output.decode()}, {e.stderr.decode()}")
        with open(os.path.join(tempdir, "config.dat"), "r") as f:
            print(f.read())
        return None

    data_path = os.path.join(config_dir, "final_demes.csv")
    df = pd.read_csv(data_path, header=0)
    df.AverageArray = df.AverageArray.apply(lambda x: np.fromstring(x, sep=";"))
    df = df[np.floor(df.Generation) == np.floor(df.Generation.max())]
    df = df.reset_index(drop=True)

    return df

def simulate_abc(params):
    """
    Wrapper around simulate to be used with pyabc.
    Args:
        params: parameters drawn from prior to use in simulation
    Returns:
        A dictionary with the simulated data.
    """
    res = simulate(params)
    return {"data": res}

