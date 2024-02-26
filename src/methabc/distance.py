import numpy as np
import pandas as pd


def create_deme_side_dataframe(df):
    """
    Create a new DataFrame where columns are grouped by side, 
    and each column corresponds to a unique deme within each side.
    Each row in these columns contains one value from the respective 'AverageArray'.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with 'AverageArray' as lists.
    Returns:
    pandas.DataFrame: New DataFrame with separate columns for each deme, grouped by side.
    """
    # Create a dictionary to hold the data, grouped by side
    data = {}
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Create a key for each unique combination of deme and side
        side = row['Side']
        deme_key = f"{row['Deme']}"
        # Initialize the side group if not already present
        if side not in data:
            data[side] = {}
        # Append the data to the corresponding key in the side group
        if deme_key not in data[side]:
            data[side][deme_key] = []
        data[side][deme_key].extend(row['AverageArray'])
    # Convert the dictionary to a DataFrame, with columns grouped by side
    reshaped_data = {}
    for side, demes in data.items():
        for deme, values in demes.items():
            column_name = f"{side}_{deme}"
            reshaped_data[column_name] = pd.Series(values)
    reshaped_df = pd.DataFrame(reshaped_data)
    return reshaped_df
    

def l2_distance(gland1, gland2):
    # Calculate absolute differences
    differences = (np.array(gland1) - np.array(gland2)) ** 2
    distance = np.sum(differences)
    # Normalize by the number of sites
    total_distance = distance / len(differences)
    return total_distance


def compute_l2_matrix(df):
    num_glands = df.shape[1]
    if num_glands != 8:
        return 10*np.ones((8, 8))
    l2_matrix = np.zeros((num_glands, num_glands))
    for i in range(num_glands):
        for j in range(num_glands):
            l2_matrix[i, j] = l2_distance(df.iloc[:, i], df.iloc[:, j])
    return l2_matrix


def distance(dfx, dfy):
    dfx = create_deme_side_dataframe(dfx)
    dfy = create_deme_side_dataframe(dfy)
    x = compute_l2_matrix(dfx)
    y = compute_l2_matrix(dfy)
    diff = x - y
    dist = np.sqrt(np.sum(diff**2))
    return dist

