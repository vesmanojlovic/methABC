import numpy as np

def squared_distance(gland1, gland2):
    differences = (np.array(gland1) - np.array(gland2)) ** 2
    distance = np.sum(differences)

    total_distance = distance / len(differences)
    return total_distance


def compute_deme_matrix(df):
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
    df1 = dict1['data']
    df2 = dict2['data']

    x = compute_deme_matrix(df1)
    y = compute_deme_matrix(df2)
    diff = x - y
    res = np.sqrt(np.sum(diff ** 2))
    return res
