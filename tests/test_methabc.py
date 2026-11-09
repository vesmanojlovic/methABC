import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from methabc.distance import total_distance, total_distance_loop, distance_terms
from methabc.simulate import simulate, simulate_abc
from methabc.utils import import_data, side_counts, write_config, repo_dir

needs_methdemon = pytest.mark.skipif(
    not os.path.exists(os.path.join(repo_dir, "resources/methdemon/bin/methdemon")),
    reason="methdemon not compiled",
)

params = {
    'meth_rate': 0.002,
    'demeth_rate': 0.002,
    'init_migration_rate': 0.01,
    'mu_driver_birth': 0.0001,
    's_driver_birth': 0.1,
}


def random_sim(num_left, num_right, num_loci):
    num_demes = num_left + num_right
    return pd.DataFrame({
        'Generation': 100.0,
        'Deme': range(num_demes),
        'Side': ['left'] * num_left + ['right'] * num_right,
        'Population': 100,
        'OriginTime': np.random.uniform(0, 50, num_demes),
        'AverageArray': [np.random.uniform(0, 1, num_loci) for _ in range(num_demes)],
    })


@pytest.mark.parametrize("data_id", ["D", "E", "P"])
def test_total_distance(data_id):
    observation = import_data(os.path.join(repo_dir, "data/individual/tumour_" + data_id + ".csv"))
    num_left, num_right = side_counts(observation)
    for num_loci in [observation.shape[0], 1200]:
        sim = random_sim(num_left, num_right, num_loci)
        res = total_distance_loop({'data': sim}, {'data': observation})
        assert total_distance({'data': sim}, {'data': observation}) == pytest.approx(res)
        assert total_distance({'data': observation}, {'data': sim}) == pytest.approx(res)


def test_distance_terms():
    observation = import_data(os.path.join(repo_dir, "data/individual/tumour_E.csv"))
    sim = random_sim(4, 4, 1200)
    l2, wass = distance_terms({'data': sim}, {'data': observation})
    assert l2 + wass == pytest.approx(total_distance({'data': sim}, {'data': observation}))


def test_total_distance_failed_sim():
    observation = import_data(os.path.join(repo_dir, "data/individual/tumour_E.csv"))
    assert total_distance({'data': None}, {'data': observation}) == 1000
    sim = random_sim(8, 8, observation.shape[0])
    assert total_distance({'data': sim}, {'data': observation}) == 1000


def test_side_counts():
    assert side_counts(import_data(os.path.join(repo_dir, "data/individual/tumour_D.csv"))) == (2, 6)
    assert side_counts(import_data(os.path.join(repo_dir, "data/individual/tumour_P.csv"))) == (3, 5)
    assert side_counts(import_data(os.path.join(repo_dir, "data/individual/tumour_E.csv"))) == (4, 4)


def test_write_config(tmp_path):
    config_path = write_config(
        params={'seed': 12345, 'left_demes': 2, 'right_demes': 6, 'mu_driver_birth': 10**-4.9},
        output_dir=tmp_path,
    )
    with open(config_path, "r") as f:
        config = f.read()
    assert "seed 12345" in config
    assert "left_demes 2" in config
    assert "right_demes 6" in config
    assert "mu_driver_birth 1.258925412e-05" in config


@needs_methdemon
def test_simulate():
    df = simulate(params, left_demes=2, right_demes=6)
    assert df.shape[0] == 8
    assert len(df[df.Side == 'left']) == 2
    other = simulate(params, left_demes=2, right_demes=6)
    assert not np.allclose(np.stack(df.AverageArray), np.stack(other.AverageArray))


@needs_methdemon
def test_synthetic_data(tmp_path):
    script = os.path.join(repo_dir, "scripts/synthetic_data.py")
    for name in ["a", "b"]:
        subprocess.run([sys.executable, script, "-o", str(tmp_path / name), "-l", "3", "-r", "5"], check=True)
    data = import_data(tmp_path / "a.csv")
    assert side_counts(data) == (3, 5)
    assert data.equals(import_data(tmp_path / "b.csv"))
    gt = pd.read_csv(tmp_path / "a_params.csv").iloc[0]
    assert gt['meth_rate'] == pytest.approx(np.log10(0.0022))


@needs_methdemon
def test_simulate_timeout():
    assert simulate_abc(params, timeout=0.01)['data'] is None
