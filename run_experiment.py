import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
import jax
import jaxopt
from jax import jit
import opt_einsum as oe
import matplotlib.pyplot as plt

import tqdm

import loss_functions as L
import gcp


def experiment(N, d, r, percent_missing=None, num_iters=1000, num_samples=10, lr=0.01, seed=0):

    # Set the random seed
    rng = rnd.default_rng(seed)

    if percent_missing is None:
        # percent_missing = np.array([0.1, 0.5, 0.8])
        percent_missing = np.arange(10) / 10
    num_missing = len(percent_missing)

    if isinstance(r, int):
        decomp = 'cp'
        init_factors = gcp.init_cp
        compose = gcp.cp_to_tensor
    else:
        decomp = 'tt'
        init_factors = gcp.init_tt
        compose = gcp.tt_to_tensor

    # dist_list = [
    #     'normal',
    #     'gamma',
    #     'rayleigh',
    #     'poisson_linear',
    #     'poisson_log',
    #     'bernoulli_odds',
    #     'bernoulli_logit',
    #     'negative_binomial',
    # ]
    # dist_list = ['normal', 'poisson_linear', 'bernoulli_logit']
    dist_list = ['poisson_linear', 'normal', 'bernoulli_logit']
    results = {}
    for dist in dist_list:
        results[dist] = np.zeros([num_samples, num_missing])

    # Generate distribution parameters
    U = init_factors(d, r, seed=rng)
    Mtrue = compose(U)
    results['Mtrue'] = Mtrue

    for dist in tqdm.tqdm(dist_list):
        # Clear the JIT cache and choose the loss function
        jax.clear_caches()
        L.loss_fun = getattr(L, dist)
        for i in tqdm.trange(num_samples, leave=False):
            # Generate tensor X for different distributions
            X = gcp.generate_data(Mtrue, decomp=decomp, seed=rng, dist=dist)

            for p in tqdm.trange(num_missing, leave=False):
                # Generate mask
                mask = np.ones(d).ravel()
                missing_size = int(mask.size * percent_missing[p])
                mask[:missing_size] = 0
                rng.shuffle(mask)
                mask = mask.reshape(d).astype(np.float32)

                # Solve the problem
                U0 = init_factors(d, r, seed=rng)
                Uhat, loss_mask, loss_full = gcp.decompose_lbfgs(
                    X, mask, U0,
                    objective_fun=decomp
                )

                results[dist][i, p] = loss_full

    return results


seed = 0
N = 4
d = jnp.array([2, 3, 4, 5])
assert N == len(d)
r = 20

num_iters = 1000
loss = np.zeros(num_iters)
lr = 0.01

# r = jnp.array([2, 6, 20, 5])  # Full-rank
# r = jnp.array([2, 2, 2, 2])  # Low-rank

percent_missing = np.arange(10) / 10
num_samples = 10

rng = rnd.default_rng(seed)
mask = 1 - rng.binomial(n=1, p=percent_missing[0], size=d).astype(np.float32)

r_cp = 4
results_cp = experiment(N, d, r_cp, percent_missing=percent_missing, num_samples=num_samples, lr=lr, seed=seed)

r_tt = jnp.array([2, 2, 2])  # Low-rank
results_tt = experiment(N, d, r_tt, percent_missing=percent_missing, num_samples=num_samples, lr=lr, seed=seed)

np.savez(
    'results_low.npz',
    results_cp=results_cp,
    results_tt=results_tt,
    percent_missing=percent_missing,
    r_cp=r_cp,
    r_tt=r_tt,
    d=d
)


r_tt = jnp.array([2, 6, 5])  # Full-rank
results_tt = experiment(N, d, r_tt, percent_missing=percent_missing, num_samples=num_samples, lr=lr, seed=seed)

r_cp = 24
results_cp = experiment(N, d, r_cp, percent_missing=percent_missing, num_samples=num_samples, lr=lr, seed=seed)

np.savez(
    'results_full.npz',
    results_cp=results_cp,
    results_tt=results_tt,
    percent_missing=percent_missing,
    r_cp=r_cp,
    r_tt=r_tt,
    d=d
)



print()



# dist_list = ['normal', 'gamma', 'rayleigh', 'poisson_linear', 'poisson_log', 'bernoulli_odds', 'bernoulli_logit', 'negative_binomial']
dist_list = ['normal', 'poisson_linear', 'bernoulli_logit']

# for dist, loss in results_cp.items():
for dist in dist_list:
    loss = results_cp[dist]
    fig, ax = plt.subplots()
    ax.plot(percent_missing, loss.mean(axis=0))
    ax.set_xlabel('Missing values $\%$')
    ax.set_ylabel('Loss')
    ax.set_title(f'GCP ({dist} distribution)')
    # ax.axis([0, num_iters, None, None])
    fig.tight_layout()

for dist in dist_list:
    loss = results_tt[dist]
    fig, ax = plt.subplots()
    ax.plot(percent_missing, loss.mean(axis=0))
    ax.set_xlabel('Missing values $\%$')
    ax.set_ylabel('Loss')
    ax.set_title(f'GTT ({dist} distribution)')
    # ax.axis([0, num_iters, None, None])
    fig.tight_layout()




# d = jnp.array([2, 3, 4, 5])
# for dist in results.keys():
#     T = gcp.generate_data(d, dist=dist)
#     fun = getattr(L, dist)
#     print(dist, fun(T, T).mean())


