import numpy as np
from scipy.stats import wasserstein_distance


def squared_distance(gland1, gland2):
    """
    Compute the squared difference for each fCpG site in two glands. Sum and
    normalise.
    """
    differences = (np.array(gland1) - np.array(gland2)) ** 2
    distance = np.sum(differences)
    total_distance = distance / len(differences)
    return total_distance


def compute_deme_matrix(df):
    """
    Compute the distance matrix for a deme. If there are fewer than 8 demes,
    reject the df automatically.
    """
    df = df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
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


def l2_distance(dict1, dict2):
    """
    Compute the distance between two simulated tumours - L_2 norm of the
    difference of their distance matrices (multiplied by 1/2 to only include
                                           above diagonal).
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
    Compute the Wasserstein distance between two simulated tumours. For fewer
    than 8 demes, reject the comparison automatically.
    """
    df1 = dict1['data']
    df2 = dict2['data']
    if df1.shape[0] != 8 or df2.shape[0] != 8:
        return 1000
    res = 0
    for i in range(8):
        res += wasserstein_distance(
                df1.iloc[i].AverageArray,
                df2.iloc[i].AverageArray
                )
    return res
