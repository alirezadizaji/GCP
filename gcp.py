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
import link_functions as Lf


POS_CONSTRAINT = dict(
    normal=False,
    poisson_log=False,
    bernoulli_logit=False,
    gamma=True,
    rayleigh=True,
    poisson_linear=True,
    bernoulli_odds=True,
    negative_binomial=True,
)


@jit
def cp_to_tensor(U):
    N = len(U)
    contract = []
    for i in range(N):
        contract.append(U[i])
        contract.append([f'i{i}', 'r'])
    contract.append([f'i{i}' for i in range(N)])

    return oe.contract(*contract)

@jit
def tt_to_tensor(U):
    N = len(U)
    contract = [U[0], ['i0', 'r0']]
    for i in range(1, N-1):
        contract.append(U[i])
        contract.append([f'r{i-1}', f'i{i}', f'r{i}'])
    contract.append(U[-1])
    contract.append([f'r{N-2}', f'i{N-1}'])
    contract.append([f'i{i}' for i in range(N)])

    return oe.contract(*contract)

@jit
def objective_gcp(U, X, mask):
    return jnp.sum(L.loss_fun(cp_to_tensor(U), X) * mask)

@jit
def objective_gtt(U, X, mask):
    return jnp.sum(L.loss_fun(tt_to_tensor(U), X) * mask)

def _generate_data(d, r, dist='normal', seed=None):
    if isinstance(r, int):
        decomp = 'cp'
        init_factors = init_cp
        compose = cp_to_tensor
    else:
        decomp = 'tt'
        init_factors = init_tt
        compose = tt_to_tensor

    rng = rnd.default_rng(seed)

    U = init_factors(d, r, seed=rng)
    M = compose(U)
    link = getattr(Lf, dist)
    theta = link(M)

    if dist == 'normal':
        X = rng.normal(loc=theta)
        # X = rng.normal(loc=0, size=d)
    elif dist == 'gamma':
        X = rng.gamma(shape=0.1, scale=theta)
    elif dist == 'rayleigh':
        X = rng.rayleigh(scale=theta)
    elif 'poisson' in dist:
        X = rng.poisson(lam=theta)
    elif 'bernoulli' in dist:
        X = rng.binomial(n=1, p=theta, size=d).astype(np.float32)
    elif dist == 'negative_binomial':
        X = rng.negative_binomial(n=2, p=theta, size=d)
    else:
        raise ValueError('Invalid distribution')
    return X


def generate_data(M, dist='normal', decomp='cp', seed=None):
    if decomp == 'cp':
        init_factors = init_cp
        compose = cp_to_tensor
    elif decomp == 'tt':
        init_factors = init_tt
        compose = tt_to_tensor
    else:
        raise ValueError('Invalid decomposition')

    rng = rnd.default_rng(seed)

    # U = init_factors(d, r, seed=rng)
    # M = compose(U)
    link = getattr(Lf, dist)
    theta = link(M)

    if dist == 'normal':
        X = rng.normal(loc=theta)
        # X = rng.normal(loc=0, size=d)
    elif dist == 'gamma':
        X = rng.gamma(shape=0.1, scale=theta)
    elif dist == 'rayleigh':
        X = rng.rayleigh(scale=theta)
    elif 'poisson' in dist:
        X = rng.poisson(lam=theta)
    elif 'bernoulli' in dist:
        X = rng.binomial(n=1, p=theta).astype(np.float32)
    elif dist == 'negative_binomial':
        X = rng.negative_binomial(n=2, p=theta)
    else:
        raise ValueError('Invalid distribution')
    return X



def decompose(
        T, mask, U0,
        loss_fun=None,
        grad_fun=None,
        objective_fun='cp',
        lr=0.01, num_iters=1000,
        seed=None, use_tqdm=False
    ):

    if use_tqdm:
        irange = tqdm.trange
    else:
        irange = range

    # Choose the loss function
    if (L.loss_fun != loss_fun) and (loss_fun is not None):
        jax.clear_caches()
        L.loss_fun = loss_fun

    # Check if the loss function has a positivity constraint
    if POS_CONSTRAINT[L.loss_fun.__name__]:
        non_negativity = True
    else:
        non_negativity = False

    # Define objective function
    if objective_fun == 'cp':
        objective_fun = objective_gcp
    elif objective_fun == 'tt':
        objective_fun = objective_gtt

    # Precompile gradient (this can be substituted by a custom gradient function)
    if grad_fun is None:
        grad = jit(jax.grad(objective_fun))

    # Initialize U
    U = [u.copy() for u in U0]

    # Gradient descent
    loss = np.zeros(num_iters+1)
    loss[0] = objective_fun(U, T, mask)
    for i in irange(num_iters):
        # Compute gradient
        gradU = grad(U, T, mask)

        # Update U
        U = jax.tree_map(lambda x, g: x - lr * g, U, gradU)

        # Apply positivity constraint
        if non_negativity:
            U = jax.tree_map(lambda x: jnp.maximum(x, 0), U)

        # Keep track of the loss
        loss[i+1] = objective_fun(U, T, mask)

    # Normalize loss by number of observed entries
    loss /= mask.sum()

    return U, loss


def decompose_lbfgs(
        X, mask, U0,
        loss_fun=None,
        grad_fun=None,
        objective_fun='cp'
    ):

    # Choose the loss function
    if (L.loss_fun != loss_fun) and (loss_fun is not None):
        jax.clear_caches()
        L.loss_fun = loss_fun

    # Check if the loss function has a positivity constraint
    if POS_CONSTRAINT[L.loss_fun.__name__]:
        # non_negativity = True
        lower_bounds = jax.tree_map(lambda x: x * 0, U0)
        upper_bounds = jax.tree_map(lambda x: x * jnp.inf, U0)
        bounds = (lower_bounds, upper_bounds)
    else:
        # non_negativity = False
        bounds = None

    # Define objective function
    if objective_fun == 'cp':
        objective_fun = objective_gcp
    elif objective_fun == 'tt':
        objective_fun = objective_gtt

    # Solve using L-BFGS-B
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=objective_fun, method="l-bfgs-b")
    result = lbfgsb.run(U0, bounds=bounds, X=X, mask=mask)
    U = result.params
    loss_mask = objective_fun(U, X, mask) / mask.sum()
    loss_full = objective_fun(U, X, np.ones_like(X)).mean()

    return U, loss_mask, loss_full

def init_cp(d, r, seed=None):
    rng = rnd.default_rng(seed)
    N = len(d)
    U = [rng.uniform(size=(d[i], r)) for i in range(N)]
    return U

def init_tt(d, r, seed=None):
    rng = rnd.default_rng(seed)
    N = len(d)

    U = [None] * N
    U[0] = rng.uniform(size=(d[0], r[0]))
    for i in range(1, N-1):
        U[i] = rng.uniform(size=(r[i-1], d[i], r[i]))
    U[-1] = rng.uniform(size=(r[-1], d[-1]))

    return U

def _solve_gtt(T, mask, r, loss_fun=None, grad_fun=None, objective_fun=None, lr=0.01, num_iters=1000, seed=None, use_tqdm=False):

    if use_tqdm:
        irange = tqdm.trange
    else:
        irange = range

    # Set the random number generator
    rng = rnd.default_rng(seed)

    # Choose the loss function
    if (L.loss_fun != loss_fun) and (loss_fun is not None):
        jax.clear_caches()
        L.loss_fun = loss_fun

    # Define objective function
    if objective_fun is None:
        objective_fun = objective_gtt

    # Precompile gradient (this can be substituted by a custom gradient function)
    if grad_fun is None:
        grad = jit(jax.grad(objective_fun))

    # Initialize U
    d = T.shape
    N = len(d)
    U = [None] * N
    U[0] = rng.normal(size=(d[0], r[0]))
    for i in range(1, N-1):
        U[i] = rng.normal(size=(r[i-1], d[i], r[i]))
    U[-1] = rng.normal(size=(r[-1], d[-1]))

    # Gradient descent
    loss = np.zeros(num_iters+1)
    loss[0] = objective_fun(U, T, mask)
    for i in irange(num_iters):
        # Compute gradient
        gradU = grad(U, T, mask)

        # Update U
        U = jax.tree_map(lambda x, g: x - lr * g, U, gradU)

        # Keep track of the loss
        loss[i+1] = objective_fun(U, T, mask)

    return U, loss


if __name__ == '__main__':
    seed = 0
    N = 4
    d = jnp.array([2, 3, 4, 5])
    assert N == len(d)
    r = 20

    num_iters = 1000
    loss = np.zeros(num_iters)
    lr = 0.01

    # Clear the JIT cache and choose the loss function
    dist = 'normal'
    jax.clear_caches()
    L.loss_fun = getattr(L, dist)

    # Generate mask
    mask = 1 - rnd.binomial(n=1, p=0.1, size=d).astype(np.float32)

    # Generate T for different distributions
    T = generate_data(d, seed=seed, dist=dist)

    # r = jnp.array([2, 6, 20, 5])  # Full-rank
    # r = jnp.array([2, 2, 2, 2])  # Low-rank
    r = 2
    U0 = init_cp(d, r, seed=seed)
    decompose_lbfgs(T, mask, U0, objective_fun='cp')
    print()