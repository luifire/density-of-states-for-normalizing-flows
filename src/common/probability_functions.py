from scipy.special import logsumexp
import numpy as np


def normalize_q_with_p(q, true_p):
    """Normalizes q with respect to the manifold that is distributed according to p"""
    exp_vals = q - true_p
    logged_norm_const = - np.log(len(exp_vals)) + logsumexp(exp_vals)
    return q - logged_norm_const, logged_norm_const


def normalize_grid(q, steps):
    width = steps[1] - steps[0]
    # to avoid overflows
    q = q - q.max()
    unnormed_pdf = width * np.exp(q)
    sum = np.sum(unnormed_pdf)
    normed_pdf = unnormed_pdf / sum
    normed_pd = normed_pdf / width

    # to deal with 0 during logging
    logged_pd = np.zeros_like(normed_pd)
    zeros = normed_pd == 0
    not_zeros = np.logical_not(zeros)
    logged_pd[zeros], logged_pd[not_zeros] = float('-inf'), np.log(normed_pd[not_zeros])

    return logged_pd, np.log(sum)


def create_dos_samples_of_manifold(model_logpdf, steps, size):
    width = steps[1] - steps[0]
    model_pdf = width * np.exp(model_logpdf)
    model_pdf = model_pdf / model_pdf.sum()
    sampled_model_manifold = np.random.choice(steps, size=size, replace=True, p=model_pdf)

    hit_cell = steps[None, :] >= sampled_model_manifold[:, None]
    # will stop at first True
    idx = np.argmax(hit_cell, axis=1)

    sampled_model_manifold_logpdf = model_logpdf[idx]

    return sampled_model_manifold_logpdf