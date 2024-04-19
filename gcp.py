import jax.numpy as jnp
import numpy as np
import jax.random as rnd
import jax
import jaxopt
from jax import jit
import opt_einsum as oe
import matplotlib.pyplot as plt

import tqdm

import loss_functions as L


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
def objective_gcp(U, T, mask):
    return jnp.sum(L.loss_fun(cp_to_tensor(U), T) * mask) / jnp.sum(mask)

@jit
def objective_gtt(U, T, mask):
    return jnp.sum(L.loss_fun(tt_to_tensor(U), T) * mask) / jnp.sum(mask)

def generate_data(d, seed=None, dist='normal'):

    if dist == 'normal':
        T = rnd.normal(key=seed, shape=d)
    elif dist == 'gamma':
        T = rnd.gamma(key=seed, a=0.1, shape=d)
    elif dist == 'rayleigh':
        T = rnd.rayleigh(key=seed, shape=d)
    elif 'poisson' in dist:
        T = rnd.poisson(key=seed, lam=1.0, shape=d)
    elif 'bernoulli' in dist:
        T = rnd.bernoulli(key=seed, p=0.5, shape=d).astype(jnp.float32)
    elif dist == 'negative_binomial':
        T = rnd.negative_binomial(key=seed, mu=1.0, alpha=1.0, shape=d)
    else:
        raise ValueError('Invalid distribution')
    return T


def solve_gcp(T, mask, r, loss_fun=None, grad_fun=None, objective_fun=None, lr=0.01, num_iters=1000, seed=None):
    if seed is None:
        seed = rnd.PRNGKey(0)

    # Choose the loss function
    if (L.loss_fun != loss_fun) and (loss_fun is not None):
        jax.clear_caches()
        L.loss_fun = loss_fun

    # Define objective function
    if objective_fun is None:
        objective_fun = objective_gcp

    # Precompile gradient (this can be substituted by a custom gradient function)
    if grad_fun is None:
        grad = jit(jax.grad(objective_fun))

    # Initialize U
    d = T.shape
    N = len(d)
    U = [rnd.normal(key=seed, shape=(d[i], r)) for i in range(N)]

    # Gradient descent
    loss = np.zeros(num_iters)
    for i in tqdm.trange(num_iters):
        # Compute gradient
        gradU = grad(U, T, mask)

        # Update U
        U = jax.tree_map(lambda x, g: x - lr * g, U, gradU)

        # Keep track of the loss
        loss[i] = objective_fun(U, T, mask)

    return U, loss


def solve_gtt(T, mask, r, loss_fun=None, grad_fun=None, objective_fun=None, lr=0.01, num_iters=1000, seed=None):
    if seed is None:
        seed = rnd.PRNGKey(0)

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
    U[0] = rnd.normal(key=seed, shape=(d[0], r[0]))
    for i in range(1, N-1):
        U[i] = rnd.normal(key=seed, shape=(r[i-1], d[i], r[i]))
    U[-1] = rnd.normal(key=seed, shape=(r[-1], d[-1]))

    # Gradient descent
    loss = np.zeros(num_iters)
    for i in tqdm.trange(num_iters):
        # Compute gradient
        gradU = grad(U, T, mask)

        # Update U
        U = jax.tree_map(lambda x, g: x - lr * g, U, gradU)

        # Keep track of the loss
        loss[i] = objective_fun(U, T, mask)

    return U, loss


seed = rnd.PRNGKey(0)
N = 4
d = jnp.array([2, 3, 4, 5])
assert N == len(d)
r = 20

num_iters = 1000
loss = np.zeros(num_iters)
lr = 0.01

# Clear the JIT cache and choose the loss function
jax.clear_caches()
L.loss_fun = L.normal
# L.loss_fun = L.gamma
# L.loss_fun = L.bernoulli_logit
dist = L.loss_fun.__name__

# Generate mask
mask = 1 - rnd.bernoulli(key=seed, shape=d, p=0.1).astype(jnp.float32)

# Generate T for different distributions
T = generate_data(d, seed=seed, dist=dist)

# r = jnp.array([2, 6, 20, 5])  # Full-rank
r = jnp.array([2, 2, 2, 2])  # Low-rank

U_hat, loss = solve_gtt(T, mask, r, lr=lr, num_iters=num_iters, seed=seed)

fig, ax = plt.subplots()
ax.plot(loss)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title(f'{dist} - TT')
fig.tight_layout()
ax.axis([0, num_iters, None, None])
