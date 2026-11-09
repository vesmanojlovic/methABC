import numpy as np
import pandas as pd

from itertools import permutations
from math import factorial
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment


def squared_distance(gland1, gland2):
    """
    Compute the squared distance between two glands' methylation arrays.
    """
    differences = (np.array(gland1) - np.array(gland2)) ** 2
    distance = np.sum(differences)
    total_distance = distance / len(differences)
    return total_distance


def compute_deme_matrix(df):
    """
    Compute the deme matrix for a given dataframe.
    """
    if 'Side' in df.columns:
        if df.shape[0] != 8:
            return 100 * np.ones((8, 8))
        else:
            res = np.zeros((8, 8))
            for i, row1 in df.iterrows():
                for j, row2 in df.iterrows():
                    res[i, j] = squared_distance(
                            row1.AverageArray,
                            row2.AverageArray,
                            )
            return res
    else:
        res = np.zeros((8, 8))
        fcpgs = df.shape[0]
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                res[i, j] = sum((df[col1] - df[col2]) ** 2) / fcpgs
        return res


def l2_distance(dict1, dict2):
    """
    Compute the L_2 distance between two distance matrices.
    """
    df1 = dict1['data']
    df2 = dict2['data']
    x = compute_deme_matrix(df1)
    y = compute_deme_matrix(df2)
    diff = x - y # difference of distance matrices
    res = np.sqrt(np.sum(diff ** 2) / 2) # L_2 norm (only above diagonal)
    return res


def overall_wasserstein(dict1, dict2):
    """
    Compute the Wasserstein distance between simulated tumour and data.
    """
    df1 = dict1['data']
    df2 = dict2['data']
    res = 0
    for i in range(8):
        if 'AverageArray' in df1.columns:
            df2_col = df2.columns[i]
            res += wasserstein_distance(df1.iloc[i].AverageArray, df2[df2_col])
        elif 'AverageArray' in df2.columns:
            df1_col = df1.columns[i]
            res += wasserstein_distance(df1[df1_col], df2.iloc[i].AverageArray)
        else:
            df1_col = df1.columns[i]
            df2_col = df2.columns[i]
            res += wasserstein_distance(df1[df1_col], df2[df2_col])
    return res


def distance_sum(dict1, dict2):
    return l2_distance(dict1, dict2) + overall_wasserstein(dict1, dict2)


def ith_df_perm(df, i):
    """
    Return the ith permutation of the dataframe.
    """
    left_demes = factorial(len(df[df['Side'] == 'left']))
    right_demes = factorial(len(df[df['Side'] == 'right']))
    total_perms = left_demes * right_demes

    if i < 0 or i >= total_perms:
        raise ValueError('Invalid permutation index.')

    left_df = df[df['Side'] == 'left']
    right_df = df[df['Side'] == 'right']

    left_perm_index = i // right_demes
    right_perm_index = i % right_demes

    left_perms = list(permutations(left_df.index))
    right_perms = list(permutations(right_df.index))

    left_perm = left_perms[left_perm_index]
    right_perm = right_perms[right_perm_index]

    perm_df = pd.concat([left_df.loc[list(left_perm)],
                         right_df.loc[list(right_perm)]])

    return perm_df.reset_index(drop=True)


def distance_terms(dict1, dict2):
    """
    Compute the L2 and Wasserstein terms of the total distance between two
    tumours, for the permutation of simulated glands that minimises their sum.
    Args:
        dict1, dict2: simulated and real data (in either order)
    Returns:
        l2, wasserstein (1000, 1000 if the simulation failed)
    """
    if dict1['data'] is None or dict2['data'] is None:
        return 1000, 1000
    if 'Side' in dict1['data'].columns:
        sim_df, fd = dict1['data'], dict2['data'] # simulated data, real data
    else:
        sim_df, fd = dict2['data'], dict1['data']
    if sim_df.shape[0] != fd.shape[1]:
        return 1000, 1000
    sim_df = sim_df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    num_l_demes = len(sim_df[sim_df['Side'] == 'left'])
    num_demes = sim_df.shape[0]

    sim = np.stack(sim_df.AverageArray.values)
    obs = fd.values.T
    sim_matrix = ((sim[:, None, :] - sim[None, :, :]) ** 2).sum(-1) / sim.shape[1]
    obs_matrix = ((obs[:, None, :] - obs[None, :, :]) ** 2).sum(-1) / obs.shape[1]
    wass = np.array([[wasserstein_distance(sim[i], obs[j]) for j in range(num_demes)]
                     for i in range(num_demes)])

    perms = np.array([l + r for l in permutations(range(num_l_demes))
                      for r in permutations(range(num_l_demes, num_demes))])
    l2 = np.sqrt(np.sum((sim_matrix[perms[:, :, None], perms[:, None, :]] - obs_matrix) ** 2, axis=(1, 2)) / 2)
    wass = wass[perms, np.arange(num_demes)].sum(axis=1)
    best = np.argmin(l2 + wass)
    return float(l2[best]), float(wass[best])


def total_distance(dict1, dict2):
    """
    Compute the total distance between two tumours.
    Same result as total_distance_loop.
    """
    return min(1000, sum(distance_terms(dict1, dict2)))


def l2_term(dict1, dict2):
    """
    L2 term of total_distance, for use with pyabc.AdaptiveAggregatedDistance.
    """
    return distance_terms(dict1, dict2)[0]


def wasserstein_term(dict1, dict2):
    """
    Wasserstein term of total_distance, for use with pyabc.AdaptiveAggregatedDistance.
    """
    return distance_terms(dict1, dict2)[1]


def total_distance_loop(dict1, dict2):
    """
    Compute the total distance between two tumours (version used in the paper).
    """
    res = 1000
    if 'Side' in dict1['data'].columns:
        tmp_df = dict1['data'].sort_values(by=['Side',
                                               'OriginTime']).reset_index(drop=True) # simulated data
        fd = dict2['data'] # real data
    else:
        tmp_df = dict2['data'].sort_values(by=['Side',
                                               'OriginTime']).reset_index(drop=True) # simulated data
        fd = dict1['data'] # real data
    num_l_demes = len(tmp_df[tmp_df['Side'] == 'left'])
    num_r_demes = len(tmp_df[tmp_df['Side'] == 'right'])
    iterations = factorial(num_l_demes) * factorial(num_r_demes)
    for i in range(iterations):
        perm_df = ith_df_perm(tmp_df, i)
        tmp_wass = overall_wasserstein({'data': perm_df}, {'data': fd})
        tmp_l2 = l2_distance({'data': perm_df}, {'data': fd})
        tmp_dist = tmp_wass + tmp_l2
        res = min(res, tmp_dist)
    return res
