import numpy as np
from scipy.stats import uniform, norm, multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns


from inflation.jupyter_helper import quantile_finder


def approx_density(A, B, alpha, bin_count=300, make_symmetric=False):
    """alpha quantile to cut of data"""
    ranges = [quantile_finder(A, alpha), quantile_finder(B, alpha)]
    if make_symmetric:
        min_val = min(ranges[0][0], ranges[1][0])
        max_val = max(ranges[0][1], ranges[1][1])
        ranges = [[min_val, max_val], [min_val, max_val]]

    histo_density, x_edges, y_edges = np.histogram2d(A, B, range=ranges, bins=bin_count, density=True)

    all_masses = []
    cumulative_masses = []
    cumulative_mass = 0
    for x in range(bin_count):
        x_len = x_edges[x+1] - x_edges[x]
        for y in range(bin_count):
            y_len = y_edges[y+1] - y_edges[y]

            cell_mass = histo_density[x,y] * x_len * y_len
            cumulative_mass += cell_mass

            cumulative_masses.append(cumulative_mass)
            all_masses.append(histo_density[x,y])

    #print(f'Total Mass: {cumulative_mass:.3f}')
    cumulative_masses = np.array(cumulative_masses)
    all_masses = np.array(all_masses)

    likelihood_samples = []
    sample_count = 100000
    for sample in uniform.rvs(size=sample_count):
        hitted_cell = sample < cumulative_masses

        # will stop at first True
        idx = np.argmax(hitted_cell)
        likelihood_samples.append(all_masses[idx])

    return likelihood_samples, histo_density, (x_edges, y_edges)

