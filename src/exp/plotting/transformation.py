import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import copy


def find_v_min(pd, X, Y, dataset, data_keeper, alpha):
    # data_keeper removes none-finite data.
    # now we need to look at those data that make up the main part (the one in the center)
    # thus we create an array of indices containing the main part and remove those indices, that are none-finite
    main_part = np.arange(*dataset.important_grid_part)
    pd, X, Y = pd[main_part], X[main_part], Y[main_part]

    x_size = X[1] - X[0]
    # y is the same for an entire line
    line_break = np.argmax(Y != Y[0])
    y_size = Y[line_break] - Y[line_break - 1]
    weight_of_one_cell = x_size * y_size

    # remove none finite parts
    remove_none_finite = np.logical_not(np.isinf(pd))
    pd, X, Y = pd[remove_none_finite], X[remove_none_finite], Y[remove_none_finite]

    log_prob = pd + np.log(weight_of_one_cell)
    # Z = sum(p_i) | log z = log sum exp(pd_i)*size = log(size) + log sum exp(pd_i)
    log_Z = logsumexp(log_prob)

    # y_i = 1/Z p_i | log_y = log p_i - Z
    log_y = log_prob - log_Z

    sorting = np.argsort(log_y)[::-1]
    sorted_log_y = log_y[sorting]
    sorted_true_values = pd[sorting]
    log_interesting_mass = np.log(1 - alpha)

    for N in range(len(sorted_log_y)):
        if logsumexp(sorted_log_y[:N + 1]) >= log_interesting_mass:
            return sorted_true_values[N]
    print('here')
    return pd.min()


# Plot
cmap_data_space = copy.copy(plt.cm.magma)
def plot_data_space(X, Y, pd, log, ax, var, true_pd, dataset, alpha, model, v_min=None, v_max=None, scatter_size=2,
                    plot_circle=True, green_stuff=False):

    ax.patch.set_facecolor('black')
    # ax.axis('equal')
    org_X, org_Y = X, Y
    if log:
        # plot -inf points
        no_pd = np.isinf(pd)
    else:
        pd = np.exp(pd)
        no_pd = pd != pd #np.isclose(pd, 0, atol=1e-5)

    if green_stuff:
        ax.scatter(X[no_pd], Y[no_pd], c='g', s=2)

    pd_bigger_0 = np.logical_not(no_pd)
    # make the darkest value not completely dark
    pd_finite = pd[pd_bigger_0]
    X, Y = X[pd_bigger_0], Y[pd_bigger_0]

    # rate in log space log space / v_min = (1-alpha) v_max => log(1-alpha) + log(v_max) // v_max already logged
    if log:
        if v_min == None:
            v_max = pd.max()
            # v_min = np.log(1 - alpha) + v_max
            v_min = find_v_min(pd, org_X, org_Y, dataset, pd_bigger_0, alpha)

            # otherwise the darkest value is just fully dark
            if var == 0:  # or np.log10(var) <= -6:
                #### this is wrong, we should lift the other values up to this point!!!!!!!
                v_min -= 0.05 * np.abs(v_min)
    else:
        v_min, v_max = None, None
        if var > 0 and true_pd is not None:
            v_max = np.exp(true_pd.max())
            v_min = np.exp(true_pd.min())
            #v_min -= (v_max - v_min) * 0.03  # s.t. you can see small values as well

    if not log and var != 0:
        # plot stuff bigger than maximum white
        cmap_data_space.set_over(color='white')

    if v_min is not None:
        # for speedup reasons
        big_enough = pd_finite >= v_min
        X, Y, pd_finite = X[big_enough], Y[big_enough], pd_finite[big_enough]

    sorted_ = np.argsort(pd_finite)
    sc = ax.scatter(X[sorted_], Y[sorted_], c=pd_finite[sorted_], s=scatter_size, cmap=cmap_data_space, marker='s',
                    linewidths=0, vmin=v_min, vmax=v_max)

    # Add a colorbar
    cbar = plt.gcf().colorbar(sc, ax=ax)
    # this is Dirac
    if model == 'true' and var == 0:
        cbar.set_ticks([0.001])
        cbar.ax.set_yticklabels(['∞'])
        cbar.ax.tick_params(labelsize=10)
    ax.color_bar = cbar

    # Circle
    if plot_circle:
        circle = plt.Circle((0, 0), 0.5, color='cyan', fill=False, alpha=1, linewidth=1.5)
        ax.add_patch(circle)

    ax.set_xticks([])
    ax.set_yticks([])

    def not_smaller_5(a):
        a if np.abs(a) > 0.5 else 0.5

    x_lims = not_smaller_5(X.min()), not_smaller_5(X.max())
    y_lims = not_smaller_5(Y.min()), not_smaller_5(Y.max())
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    return v_min, v_max


cmap_transformation = copy.copy(plt.cm.magma)
cmap_data_space.set_over(color='white')
def plot_transformation(X, Y, pd, log, ax, vmin_max=(None, None), scatter_size=2):
    ax.patch.set_facecolor('black')

    if not log:
        pd = np.exp(pd)
        def exp_not_none(a): return None if a is None else np.exp(a)
        vmin_max = exp_not_none(vmin_max[0]), exp_not_none(vmin_max[1])

    print_order = np.argsort(pd)
    # print_order = np.arange(len(pd))
    # make the darkest value not completely dark
    #vmin = pd.min() - 0.15 * (pd.max() - pd.min())

    # plot exceeder white
    """if vmin_max[1]:
        exceeder_idcs = pd > vmin_max[1]
        ax.scatter(X[exceeder_idcs], Y[exceeder_idcs], c=pd[exceeder_idcs], s=scatter_size,
                   color='white', linewidths=0)
    """
    sc = ax.scatter(X[print_order], Y[print_order], c=pd[print_order], s=scatter_size,
                    cmap='magma', linewidths=0, vmin=vmin_max[0], vmax=vmin_max[1], plotnonfinite=True)

    # Add a colorbar
    cbar = plt.gcf().colorbar(sc, ax=ax)

    ax.color_bar = cbar

    # cbar.formatter.set_powerlimits((0.0001, 100))
    # Circle
    # circle = plt.Circle((0, 0), 0.5, color='cyan', fill=False, alpha=1, linewidth=1.5)
    # ax.add_patch(circle)

    ax.set_xticks([])
    ax.set_yticks([])

    def not_smaller_5(a): a if np.abs(a) > 0.5 else 0.5

    x_lims = not_smaller_5(X.min()), not_smaller_5(X.max())
    y_lims = not_smaller_5(Y.min()), not_smaller_5(Y.max())
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)


def add_circle_by_var(ax, var):
    if var == 0:
        var = 1e-6
    scale = var ** 0.5
    rad = 2 * scale
    circle = plt.Circle((0, 0), rad, color='cyan', fill=False, alpha=1, linewidth=0.5)
    ax.add_patch(circle)


def add_circle(ax, radius, color='cyan'):
    circle = plt.Circle((0, 0), radius, color=color, fill=False, alpha=1, linewidth=0.5)
    ax.add_patch(circle)


def reduce_dim_keep_length_and_angle(Z, max_length=None):
    length = np.linalg.norm(Z, axis=1)
    # keeper = length < max_length if max_length is not None else length == length
    # length = length[keeper]

    ancat = Z[:, 0]  # x
    cat = Z[:, 1]  # y

    keeper = np.logical_and(np.abs(ancat) < max_length, np.abs(cat) < max_length) if max_length else ancat == ancat

    ancat, cat, length = ancat[keeper], cat[keeper], length[keeper]

    angles = np.arctan2(cat, ancat)

    X = length * np.cos(angles)
    Y = length * np.sin(angles)
    return X, Y, length, keeper


def full_extent_all(ax):
    bbox = ax.get_window_extent()
    bbox = Bbox.from_bounds(bbox.x0, bbox.y0 - 20,
                            bbox.width + 130, bbox.height + 25)
    return bbox


def get_nth_std_dev_keeper(y_data, n, std_dev, dataset):
    std_dev_line = n * std_dev if std_dev > 0 else n * dataset.y_granularity_of_grid_plot
    return np.abs(y_data) == std_dev_line


def remove_additional_granularity_points(data_X, data_Y, pd, var, dataset):
    """These points are added for the transformation plot.
    However, they create artifacts in these plots"""
    if var == 0:
        var = 1e-6

    std_dev_keeper = get_nth_std_dev_keeper(data_Y, 1, np.sqrt(var), dataset)
    manifold_keeper = data_Y == 0

    remover = np.logical_not(np.logical_or(std_dev_keeper, manifold_keeper))
    return data_X[remover], data_Y[remover], pd[remover]

