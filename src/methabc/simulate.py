import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .utils import write_config, repo_dir


def simulate(
        params,
        left_demes=-1,
        right_demes=-1,
        config_template_path=os.path.join(repo_dir, "resources/config_template.dat"),
        timeout=1800,
    ):
    """
    Simulate data using the methdemon model.
    Args:
        params: parameters to use in simulation, drawn from prior
        left_demes: number of sampled glands on side A (-1 for no constraint)
        right_demes: number of sampled glands on side B (-1 for no constraint)
        config_template_path: path to the template config
        timeout: seconds before the simulation is abandoned
    Returns:
        df: pd.DataFrame with the final output of the simulation
    """
    seed = np.random.randint(2**15 - 2) + 1
    if seed==66:
        seed += 1
    params = {**params, "seed": seed, "left_demes": left_demes, "right_demes": right_demes}

    # temporary directory to store the config and outputs
    tempdir = tempfile.mkdtemp()
    config_path = write_config(
        params=params,
        config_template_path=config_template_path,
        output_dir=tempdir,
    )
    model_path = os.path.join(repo_dir, "resources/methdemon/bin/methdemon")

    config_dir, config_name = os.path.split(config_path)

    try:
        result = subprocess.run(
            [model_path, config_dir, config_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,  # Forces an exception if the command fails
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with: {e.output.decode()}, {e.stderr.decode()}")
        with open(os.path.join(tempdir, "config.dat"), "r") as f:
            print(f.read())
        shutil.rmtree(tempdir)
        return None
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out with params: {params}")
        shutil.rmtree(tempdir)
        return None

    data_path = os.path.join(config_dir, "final_demes.csv")
    df = pd.read_csv(data_path, header=0)
    shutil.rmtree(tempdir)
    df.AverageArray = df.AverageArray.apply(lambda x: np.fromstring(x, sep=";"))
    df = df[df.Generation == df.Generation.max()]
    df = df.reset_index(drop=True)

    return df


def simulate_abc(params, **kwargs):
    """
    Wrapper around simulate to be used with pyabc.
    Args:
        params: parameters drawn from prior to use in simulation
        kwargs: passed to simulate
    Returns:
        A dictionary with the simulated data.
    """
    res = simulate(params, **kwargs)
    return {"data": res}


def noisy_abc(params, **kwargs):
    df = simulate(params, **kwargs)
    if df is None:
        return {"data": None}
    noise = [
            list(np.random.beta(
                2,
                2,
                size=len(df.AverageArray.iloc[0])
                )/100) for _ in range(df.shape[0])
            ]
    for i in range(df.shape[0]):
        df.AverageArray.iloc[i] += noise[i]
    return {"data": df}


def log_model_abc(log_params, **kwargs):
    """
    Wrapper around simulate to be used with pyabc.
    Args:
        log_params: log-transformed parameters drawn from prior to use in simulation
        kwargs: passed to simulate
    Returns:
        A dictionary with the simulated data.
    """
    params = {k: 10**v for k, v in log_params.items() if k != "s_driver_birth"}
    params["s_driver_birth"] = log_params["s_driver_birth"]
    res = simulate(params, **kwargs)
    return {"data": res}
