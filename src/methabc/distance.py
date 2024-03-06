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
    if 'Side' in df.columns:
        df = df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
        if df.shape[0] != 8:
            return 10 * np.ones((8, 8))
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
        for i in range(8):
            for j in range(8):
                res[i, j] = squared_distance(
                        df.iloc[i],
                        df.iloc[j],
                        )
        return res


def distance(dict1, dict2):
    """
    Compute the distance between two tumours - L_2 norm of the difference of
    their distance matrices (multiplied by 1/2 to only include above diagonal).
    """
    df1 = dict1['data']
    df2 = dict2['data']

    x = compute_deme_matrix(df1)
    y = compute_deme_matrix(df2)
    diff = x - y # difference of distance matrices
    res = np.sqrt(np.sum(diff ** 2) / 2) # L_2 norm (only above diagonal)
    return res


def individual_distance(gland1, gland2):
    """
    Compute the Wasserstein distance between two individual demes.
    """
    return wasserstein_distance(gland1, gland2)


def overall_wasserstein(dict1, dict2):
    """
    Compute the Wasserstein distance between two tumours.
    """

    df1 = dict1['data']
    df2 = dict2['data']

    # if df1.shape[0] != 8 or df2.shape[0] != 8:
    #     return 10

    res = 0
    for i in range(8):
        if 'AverageArray' in df1.columns and 'AverageArray' in df2.columns:
            res += individual_distance(df1.iloc[i].AverageArray, df2.iloc[i].AverageArray)
        elif 'AverageArray' in df1.columns:
            res += individual_distance(df1.iloc[i].AverageArray, df2.iloc[i])
        elif 'AverageArray' in df2.columns:
            res += individual_distance(df1.iloc[i], df2.iloc[i].AverageArray)
        else:
            res += individual_distance(df1.iloc[i], df2.iloc[i])

    return res
