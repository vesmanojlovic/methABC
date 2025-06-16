import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from itertools import combinations, permutations


def squared_distance(gland1, gland2):
    """
    Compute the squared difference for each fCpG site in two glands. Sum and
    normalise.

    Args:
        gland1: Array-like methylation values for first gland
        gland2: Array-like methylation values for second gland

    Returns:
        float: Normalized squared distance
    """
    differences = (np.array(gland1) - np.array(gland2)) ** 2
    distance = np.sum(differences)
    total_distance = distance / len(differences)
    return total_distance


def compute_deme_matrix(df):
    """
    Compute the deme matrix for a given dataframe.

    Args:
        df: DataFrame containing either:
            - Columns with 'Side' and 'AverageArray' (simulation data)
            - 8 columns of methylation values (observed data)

    Returns:
        np.ndarray: 8x8 distance matrix

    Raises:
        ValueError: If input data format is invalid
        TypeError: If df is not a pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    if 'Side' in df.columns:
        # Simulation data format
        if 'AverageArray' not in df.columns:
            raise ValueError("DataFrame with 'Side' column must also have 'AverageArray' column")

        if df.shape[0] != 8:
            return 100 * np.ones((8, 8))

        # Extract methylation arrays and validate
        try:
            arrays = [row.AverageArray for _, row in df.iterrows()]
            arrays = np.array(arrays)  # Shape: (8, n_cpgs)
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Invalid AverageArray data: {e}")

        # Vectorized computation: broadcast and compute all pairwise distances at once
        diff = arrays[:, np.newaxis, :] - arrays[np.newaxis, :, :]  # Shape: (8, 8, n_cpgs)
        squared_diff = diff ** 2  # Element-wise squaring
        distances = np.sum(squared_diff, axis=2) / arrays.shape[1]  # Sum over CpGs and normalize

        return distances

    else:
        # Observed data format
        if df.shape[1] != 8:
            raise ValueError(f"DataFrame without 'Side' column must have exactly 8 columns, got {df.shape[1]}")

        # Convert to numpy array for vectorized operations
        try:
            data = df.values  # Shape: (n_cpgs, 8)
        except Exception as e:
            raise ValueError(f"Failed to convert DataFrame to numpy array: {e}")

        # Vectorized computation using broadcasting
        diff = data[:, :, np.newaxis] - data[:, np.newaxis, :]  # Shape: (n_cpgs, 8, 8)
        squared_diff = diff ** 2
        distances = np.sum(squared_diff, axis=0) / data.shape[0]  # Sum over CpGs and normalize

        return distances


def l2_distance(dict1, dict2, permutation=None):
    """
    Compute the distance between two simulated tumours - L_2 norm of the
    difference of their distance matrices (multiplied by 1/2 to only include
    above diagonal).

    Args:
        dict1: Dictionary with 'data' key containing first DataFrame
        dict2: Dictionary with 'data' key containing second DataFrame
        permutation: Optional permutation to apply to first dataset

    Returns:
        float: L2 distance between distance matrices
    """
    df1 = dict1['data']
    df2 = dict2['data']

    # Apply permutation if provided
    if permutation is not None:
        df1 = df1.iloc[permutation].reset_index(drop=True)

    x = compute_deme_matrix(df1)
    y = compute_deme_matrix(df2)
    diff = x - y
    res = np.sqrt(np.sum(diff ** 2) / 2)
    return res


def overall_wasserstein(dict1, dict2, permutation=None):
    """
    Compute the Wasserstein distance between two simulated tumours. For fewer
    than 8 demes, reject the comparison automatically.

    Args:
        dict1: Dictionary with 'data' key containing first DataFrame
        dict2: Dictionary with 'data' key containing second DataFrame
        permutation: Optional permutation to apply to first dataset

    Returns:
        float: Sum of Wasserstein distances across all glands
    """
    df1 = dict1['data']
    df2 = dict2['data']

    if df1.shape[0] != 8:
        return 1000

    # Check if df2 has the right format
    if 'AverageArray' in df2.columns and df2.shape[0] != 8:
        return 1000
    elif 'AverageArray' not in df2.columns and df2.shape[1] != 8:
        return 1000

    # Apply permutation if provided
    if permutation is not None:
        df1 = df1.iloc[permutation].reset_index(drop=True)

    # Extract arrays based on data format
    if 'AverageArray' in df1.columns:
        arrays1 = [row.AverageArray for _, row in df1.iterrows()]
    else:
        arrays1 = [df1.iloc[:, i].values for i in range(8)]

    if 'AverageArray' in df2.columns:
        arrays2 = [row.AverageArray for _, row in df2.iterrows()]
    else:
        arrays2 = [df2.iloc[:, i].values for i in range(8)]

    res = 0
    for i in range(8):
        res += wasserstein_distance(arrays1[i], arrays2[i])
    return res


def wasserstein_2d(dict1, dict2, permutation=None):
    """
    Compute 2D Wasserstein distances between pairwise correlations of glands.

    For each pair of glands, treats their methylation arrays as 2D points
    and computes the 2-Wasserstein distance between the two point clouds.

    Args:
        dict1: Dictionary with 'data' key containing simulated DataFrame
        dict2: Dictionary with 'data' key containing observed DataFrame
        permutation: Optional permutation to apply to simulated data

    Returns:
        float: Sum of 2D Wasserstein distances across all gland pairs
    """
    df1 = dict1['data']
    df2 = dict2['data']

    # Handle rejection cases
    if df1.shape[0] != 8:
        return 1000

    # Check if df2 has the right format
    if 'AverageArray' in df2.columns and df2.shape[0] != 8:
        return 1000
    elif 'AverageArray' not in df2.columns and df2.shape[1] != 8:
        return 1000

    # Apply permutation if provided
    if permutation is not None:
        df1 = df1.iloc[permutation].reset_index(drop=True)

    # Extract methylation arrays
    try:
        if 'AverageArray' in df1.columns:
            # Simulated data
            arrays1 = np.array([row.AverageArray for _, row in df1.iterrows()])
        else:
            # Observed data (transpose to get 8 arrays)
            arrays1 = df1.values.T

        if 'AverageArray' in df2.columns:
            # Simulated data
            arrays2 = np.array([row.AverageArray for _, row in df2.iterrows()])
        else:
            # Observed data (transpose to get 8 arrays)
            arrays2 = df2.values.T
    except Exception:
        return 1000

    # Validate array shapes
    if arrays1.shape[0] != 8 or arrays2.shape[0] != 8:
        return 1000

    # Ensure arrays have the same length
    min_length = min(arrays1.shape[1], arrays2.shape[1])
    arrays1 = arrays1[:, :min_length]
    arrays2 = arrays2[:, :min_length]

    total_distance = 0

    # Compare all pairs of glands
    for i, j in combinations(range(8), 2):
        try:
            # Get methylation arrays for glands i and j
            points1 = np.column_stack([arrays1[i], arrays1[j]])  # Shape: (n_cpgs, 2)
            points2 = np.column_stack([arrays2[i], arrays2[j]])  # Shape: (n_cpgs, 2)

            # Compute 2D Wasserstein distance using optimal transport
            # Use Euclidean distance matrix between all pairs of points
            cost_matrix = cdist(points1, points2, metric='euclidean')

            # Solve assignment problem (uniform weights assumed)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Sum of optimal assignment costs divided by number of points
            wasserstein_dist = cost_matrix[row_ind, col_ind].sum() / len(points1)
            total_distance += wasserstein_dist
        except Exception:
            # If any pair fails, add penalty
            total_distance += 1000

    return total_distance


def find_optimal_permutation(dict1, dict2):
    """
    Find the optimal permutation of simulated data that minimizes total distance.
    Permutes glands within each side separately and allows side switching
    (2 × 4! × 4! = 1,152 permutations).

    Args:
        dict1: Dictionary with 'data' key containing simulated DataFrame
        dict2: Dictionary with 'data' key containing observed DataFrame

    Returns:
        tuple: (optimal_permutation, min_total_distance)
    """
    df1 = dict1['data']

    # Quick rejection for invalid data
    if df1.shape[0] != 8:
        return list(range(8)), float('inf')

    # Determine side indices based on data format
    if 'Side' in df1.columns:
        # Sort by Side and OriginTime to ensure consistent ordering
        try:
            df1_sorted = df1.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
            side_0_indices = [i for i in range(8) if df1_sorted.iloc[i]['Side'] == 0]
            side_1_indices = [i for i in range(8) if df1_sorted.iloc[i]['Side'] == 1]

            if len(side_0_indices) != 4 or len(side_1_indices) != 4:
                return list(range(8)), float('inf')
        except Exception:
            # Fallback if sorting fails
            return list(range(8)), float('inf')
    else:
        # For observed data, assume first 4 are side 0, last 4 are side 1
        side_0_indices = list(range(4))
        side_1_indices = list(range(4, 8))

    min_distance = float('inf')
    best_permutation = list(range(8))

    # Try both side assignments: original and switched
    side_assignments = [
        (side_0_indices, side_1_indices),  # Original assignment
        (side_1_indices, side_0_indices)   # Switched assignment
    ]

    for mapped_side_0, mapped_side_1 in side_assignments:
        # Try all combinations of permutations within each side
        for perm_0 in permutations(range(4)):
            for perm_1 in permutations(range(4)):
                # Construct full permutation
                full_permutation = list(range(8))

                # Apply permutation within mapped side 0
                for i in range(4):
                    full_permutation[side_0_indices[i]] = mapped_side_0[perm_0[i]]

                # Apply permutation within mapped side 1
                for i in range(4):
                    full_permutation[side_1_indices[i]] = mapped_side_1[perm_1[i]]

                # Calculate total distance for this permutation
                try:
                    l2_dist = l2_distance(dict1, dict2, permutation=full_permutation)
                    wasserstein_dist = overall_wasserstein(dict1, dict2, permutation=full_permutation)
                    wasserstein_2d_dist = wasserstein_2d(dict1, dict2, permutation=full_permutation)

                    total_dist = l2_dist + wasserstein_dist + wasserstein_2d_dist

                    if total_dist < min_distance:
                        min_distance = total_dist
                        best_permutation = full_permutation
                except Exception:
                    # Skip invalid permutations
                    continue

    return best_permutation, min_distance


def total_distance(dict1, dict2):
    """
    Compute total distance with optimal permutation.

    Args:
        dict1: Dictionary with 'data' key containing simulated DataFrame
        dict2: Dictionary with 'data' key containing observed DataFrame

    Returns:
        float: Minimum total distance across all possible permutations
    """
    try:
        optimal_permutation, min_distance = find_optimal_permutation(dict1, dict2)
        return min_distance
    except Exception:
        # Fallback to identity permutation if optimization fails
        try:
            l2_dist = l2_distance(dict1, dict2)
            wasserstein_dist = overall_wasserstein(dict1, dict2)
            wasserstein_2d_dist = wasserstein_2d(dict1, dict2)
            return l2_dist + wasserstein_dist + wasserstein_2d_dist
        except Exception:
            return float('inf')


# Keep these functions for backward compatibility with existing code
def ith_df_perm(df, perm):
    """
    Return the ith permutation of a dataframe.

    Args:
        df: Input DataFrame
        perm: Permutation to apply

    Returns:
        DataFrame: Permuted DataFrame
    """
    return df.iloc[perm].reset_index(drop=True)
