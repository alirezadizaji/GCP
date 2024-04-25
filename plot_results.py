import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
import jax
import jaxopt
from jax import jit
import opt_einsum as oe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import tqdm

import loss_functions as L
import gcp


def plot_with_fill(ax, x, y, **kwargs):
    values = np.sort(y, axis=-1)
    num_samples = values.shape[-1]
    for r in range(num_samples//2):
        ax.fill_between(
            x,
            values[:, r],
            values[:, -r-1],
            **kwargs
        )


# results = dict(np.load('results.npz', allow_pickle=True))
rank = 'low'
# rank = 'full'

results = dict(np.load(f'results_{rank}.npz', allow_pickle=True))

results_cp = results['results_cp'].item()
results_tt = results['results_tt'].item()

percent_missing = results['percent_missing']

print()



# dist_list = ['normal', 'gamma', 'rayleigh', 'poisson_linear', 'poisson_log', 'bernoulli_odds', 'bernoulli_logit', 'negative_binomial']
dist_list = ['normal', 'poisson_linear', 'bernoulli_logit']


for dist in dist_list:
    fig, ax = plt.subplots()
    ax.plot(percent_missing, results_cp[dist].mean(axis=0), label='GCP')
    ax.plot(percent_missing, results_tt[dist].mean(axis=0), label='GTT')

    if dist == 'normal':
        ax.set_yscale('log')

    ax.set_xlabel('Missing values $\%$')
    ax.set_ylabel('Loss')
    ax.set_title(f'{dist} distribution')
    ax.legend()
    ax.autoscale(enable=True, axis='x', tight=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(f'figures/gcp+gtt_{rank}_{dist}.pdf', transparent=True)


# # for dist, loss in results_cp.items():
# for dist in dist_list:
#     loss = results_cp[dist]
#     num_samples = loss.shape[0]
#     fig, ax = plt.subplots()
#     ax.plot(percent_missing, loss.mean(axis=0))
#     # plot_with_fill(
#     #     ax, percent_missing, loss,
#     #     color=color_list[0],
#     #     alpha=1/num_samples
#     # )
#     ax.set_xlabel('Missing values $\%$')
#     ax.set_ylabel('Loss')
#     ax.set_title(f'GCP ({dist} distribution)')
#     ax.autoscale(enable=True, axis='x', tight=True)
#     fig.tight_layout(pad=0.1)
#     fig.savefig(f'figures/gcp_{dist}.pdf', transparent=True)

# for dist in dist_list:
#     loss = results_tt[dist]
#     num_samples = loss.shape[0]
#     fig, ax = plt.subplots()
#     ax.plot(percent_missing, loss.mean(axis=0))
#     # plot_with_fill(
#     #     ax, percent_missing, loss,
#     #     color=color_list[0],
#     #     alpha=1/num_samples
#     # )
#     ax.set_xlabel('Missing values $\%$')
#     ax.set_ylabel('Loss')
#     ax.set_title(f'GTT ({dist} distribution)')
#     ax.autoscale(enable=True, axis='x', tight=True)
#     fig.tight_layout(pad=0.1)
#     fig.savefig(f'figures/gtt_{dist}.pdf', transparent=True)

# d = jnp.array([2, 3, 4, 5])
# for dist in results.keys():
#     T = gcp.generate_data(d, dist=dist)
#     fun = getattr(L, dist)
#     print(dist, fun(T, T).mean())


# rank = 'full'

rank = 'low'
results = dict(np.load(f'results_{rank}.npz', allow_pickle=True))
results_cp_low = results['results_cp'].item()
results_tt_low = results['results_tt'].item()

rank = 'full'
results = dict(np.load(f'results_{rank}.npz', allow_pickle=True))
results_cp_full = results['results_cp'].item()
results_tt_full = results['results_tt'].item()


percent_missing = results['percent_missing']

dist_list = ['normal', 'poisson_linear', 'bernoulli_logit']


for dist in dist_list:
    fig, ax = plt.subplots()
    ax.plot(percent_missing, results_cp_low[dist].mean(axis=0), label='GCP (low-rank)', marker='o')
    ax.plot(percent_missing, results_tt_low[dist].mean(axis=0), label='GTT (low-rank)', marker='o')
    ax.plot(percent_missing, results_cp_full[dist].mean(axis=0), label='GCP (full-rank)', marker='o')
    ax.plot(percent_missing, results_tt_full[dist].mean(axis=0), label='GTT (full-rank)', marker='o')

    ax.set_yscale('log')

    # if dist == 'normal':
    #     ax.set_yscale('log')

    ax.set_xlabel('Missing values $\%$')
    ax.set_ylabel('Loss')
    ax.set_title(f'{dist} distribution')
    ax.legend()
    ax.autoscale(enable=True, axis='x', tight=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(f'figures/gcp+gtt_low+full_{dist}.pdf', transparent=True)
















