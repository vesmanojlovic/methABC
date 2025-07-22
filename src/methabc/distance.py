import numpy as np
import pandas as pd
import pyabc
import ot as pot

from itertools import permutations
from math import factorial
from scipy.stats import wasserstein_distance


class CombinedDistance(pyabc.Distance):
    """
    A custom, stateful distance class for the methABC project.

    This class implements a combined distance metric that includes:
    1.  1D Wasserstein distance between individual deme methylation arrays.
    2.  L2 distance between the inter-deme distance matrices.
    3.  2D Wasserstein distance between 2D histograms of deme-pair correlations.

    It solves the assignment problem by performing an exhaustive brute-force search
    over all valid permutations, ensuring the true minimum of the combined, scaled
    distance is found.

    Normalization is handled adaptively for each generation using the median
    absolute deviation (MAD) of each distance component, calculated in the
    `initialize` method.
    """

    def __init__(self, num_bins=100):
        super().__init__()
        self.num_bins = num_bins
        self.scale_1d_wass = 1.0
        self.scale_l2 = 1.0
        self.scale_2d_wass = 1.0

        # Pre-calculate cost_matrix for 2D Wasserstein distance
        x_bins = np.linspace(0, 1, self.num_bins)
        y_bins = np.linspace(0, 1, self.num_bins)
        xv, yv = np.meshgrid(x_bins, y_bins)
        self.cost_matrix = np.sqrt((xv.flatten()[:, np.newaxis] - xv.flatten()[np.newaxis, :])**2 + \
                                   (yv.flatten()[:, np.newaxis] - yv.flatten()[np.newaxis, :])**2)

    def initialize(self, t, get_sample, x_0, total_sims):
        """
        Initializes the distance function by calculating normalization constants.
        This method is called by the ABCSMC sampler at the start of each generation.
        """
        sample = get_sample()
        observed_sum_stat = x_0

        raw_distances_1d = []
        raw_distances_l2 = []
        raw_distances_2d = []

        # Use a default, non-optimal permutation for speed during initialization
        for particle in sample.all_particles:
            sim_sum_stat = particle.sum_stat
            dist_1d, dist_l2, dist_2d = self._calculate_distances_for_permutation(
                sim_sum_stat['data'], observed_sum_stat['data'], default_perm=True
            )
            raw_distances_1d.append(dist_1d)
            raw_distances_l2.append(dist_l2)
            raw_distances_2d.append(dist_2d)

        self.scale_1d_wass = pyabc.distance.mad(np.array(raw_distances_1d))
        self.scale_l2 = pyabc.distance.mad(np.array(raw_distances_l2))
        self.scale_2d_wass = pyabc.distance.mad(np.array(raw_distances_2d))

        if self.scale_1d_wass == 0: self.scale_1d_wass = 1.0
        if self.scale_l2 == 0: self.scale_l2 = 1.0
        if self.scale_2d_wass == 0: self.scale_2d_wass = 1.0

    def __call__(self, x, y, t, par):
        """
        The main distance calculation method.
        Performs a brute-force search over all permutations.
        """
        sim_df = x['data']
        real_df = y['data']

        min_total_scaled_dist = float('inf')

        # --- Normal Assignment ---
        dist_normal = self._brute_force_search(sim_df, real_df, mirrored=False)
        
        # --- Mirrored Assignment ---
        dist_mirrored = self._brute_force_search(sim_df, real_df, mirrored=True)

        return min(dist_normal, dist_mirrored)

    def _brute_force_search(self, sim_df, real_df, mirrored):
        """Performs the brute-force search for a given orientation."""
        min_scaled_dist_for_orientation = float('inf')

        sim_left = sim_df[sim_df['Side'] == 'left']
        sim_right = sim_df[sim_df['Side'] == 'right']

        real_A_cols = [col for col in real_df.columns if 'A' in col or 'testA' in col]
        real_B_cols = [col for col in real_df.columns if 'B' in col or 'testB' in col]
        real_A = real_df[real_A_cols]
        real_B = real_df[real_B_cols]

        if mirrored:
            real_left, real_right = real_B, real_A
        else:
            real_left, real_right = real_A, real_B

        left_perms = list(permutations(range(len(sim_left))))
        right_perms = list(permutations(range(len(sim_right))))

        for l_perm in left_perms:
            for r_perm in right_perms:
                perm_sim_left = sim_left.iloc[list(l_perm)]
                perm_sim_right = sim_right.iloc[list(r_perm)]
                perm_sim_df = pd.concat([perm_sim_left, perm_sim_right]).reset_index(drop=True)

                dist_1d, dist_l2, dist_2d = self._calculate_distances_for_permutation(
                    perm_sim_df, real_df
                )
                
                scaled_dist = (dist_1d / self.scale_1d_wass) + \
                              (dist_l2 / self.scale_l2) + \
                              (dist_2d / self.scale_2d_wass)
                
                min_scaled_dist_for_orientation = min(min_scaled_dist_for_orientation, scaled_dist)

        return min_scaled_dist_for_orientation

    def _calculate_distances_for_permutation(self, perm_sim_df, real_df, default_perm=False):
        """Calculates the three distance components for a single, given permutation."""
        
        # 1. 1D Wasserstein Distance
        dist_1d = 0
        for i in range(len(perm_sim_df)):
            dist_1d += wasserstein_distance(perm_sim_df.iloc[i]['AverageArray'], real_df[real_df.columns[i]])

        # 2. L2 Distance
        sim_matrix = self._compute_inter_deme_matrix(perm_sim_df)
        real_matrix = self._compute_inter_deme_matrix(real_df)
        dist_l2 = np.sqrt(np.sum((sim_matrix - real_matrix) ** 2) / 2)

        # 3. 2D Wasserstein Distance
        dist_2d = 0
        if not default_perm: # Avoid this expensive calculation during initialization
            unique_pairs = list(permutations(range(len(perm_sim_df)), 2))
            for i, j in unique_pairs:
                # Simulated data histogram
                sim_hist = np.histogram2d(perm_sim_df.iloc[i]['AverageArray'], perm_sim_df.iloc[j]['AverageArray'], bins=self.num_bins)[0]
                # Real data histogram
                real_hist = np.histogram2d(real_df[real_df.columns[i]], real_df[real_df.columns[j]], bins=self.num_bins)[0]
                
                # Normalize histograms
                if np.sum(sim_hist) > 0: sim_hist /= np.sum(sim_hist)
                if np.sum(real_hist) > 0: real_hist /= np.sum(real_hist)

                # 2D Wasserstein distance using POT
                dist_2d += pot.emd2(sim_hist.flatten(), real_hist.flatten(), self.cost_matrix)

        return dist_1d, dist_l2, dist_2d

    def _compute_inter_deme_matrix(self, df):
        """Computes the inter-deme distance matrix (squared distance)."""
        if 'AverageArray' in df.columns:
            n_demes = len(df)
        else:
            n_demes = len(df.columns)

        matrix = np.zeros((n_demes, n_demes))
        for i in range(n_demes):
            for j in range(n_demes):
                if 'AverageArray' in df.columns:
                    arr1 = df.iloc[i]['AverageArray']
                    arr2 = df.iloc[j]['AverageArray']
                else:
                    arr1 = df[df.columns[i]]
                    arr2 = df[df.columns[j]]
                
                diff = np.array(arr1) - np.array(arr2)
                if len(diff) > 0:
                    matrix[i, j] = np.sum(diff**2) / len(diff)
                else:
                    matrix[i, j] = 0
        return matrix
