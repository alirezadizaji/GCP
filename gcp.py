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
def objective(U, T, mask):
    return jnp.sum(L.loss_fun(cp_to_tensor(U), T) * mask) / jnp.sum(mask)


def generate_data(N, d, r, seed=None, dist='normal'):

    if dist == 'normal':
        T = rnd.normal(key=seed, shape=d)
        U = [rnd.normal(key=seed, shape=(d[i], r)) for i in range(N)]
    elif dist == 'gamma':
        T = rnd.gamma(key=seed, a=0.1, shape=d)
        U = [rnd.normal(key=seed, shape=(d[i], r)) for i in range(N)]
    elif dist == 'bernoulli':
        T = rnd.bernoulli(key=seed, p=0.5, shape=d).astype(jnp.float32)
        U = [rnd.normal(key=seed, shape=(d[i], r)) for i in range(N)]
    else:
        raise ValueError('Invalid distribution')
    return T, U



seed = rnd.PRNGKey(0)
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
if dist == 'normal':
    L.loss_fun = L.normal
elif dist == 'gamma':
    L.loss_fun = L.gamma
elif dist == 'bernoulli':
    L.loss_fun = L.bernoulli_logit

# Generate mask
mask = 1 - rnd.bernoulli(key=seed, shape=d, p=0.1).astype(jnp.float32)

# Generate T and initialize U for different distributions
T, U = generate_data(N, d, r, seed=seed, dist=dist)


# # Projected gradient descent (constrained optimization)
# projection = jaxopt.projection.projection_non_negative
# pg_solver = jaxopt.ProjectedGradient(fun=objective, projection=projection, maxiter=num_iters, tol=1e-5, verbose=True)
# res = pg_solver.run(U, T=T, mask=mask)

# Gradient descent (unconstrained)
gd_solver = jaxopt.GradientDescent(fun=objective, stepsize=lr, maxiter=num_iters, tol=1e-5, verbose=True)
res = gd_solver.run(U, T, mask)

# Get results
U_hat = res.params
state = res.state

# Compute error
error = objective(U_hat, T, mask)

print(error)

# mask = 1 - rnd.bernoulli(key=seed, shape=d, p=0.0).astype(jnp.float32)
# T = rnd.normal(key=seed, shape=d)
# U = [rnd.normal(key=seed, shape=(d[i], r)) for i in range(N)]
# for i in tqdm.trange(num_iters):
#     # Compute gradient
#     gradU = jax.grad(objective)(U, T, mask)

#     # Update U using gradient descent
#     U = jax.tree_map(lambda x, g: x - lr * g, U, gradU)

#     loss[i] = objective(U, T, mask)

# fig, ax = plt.subplots()
# ax.plot(loss)



# N = 4
# d = jnp.array([2, 3, 4, 5])
# r = jnp.array([2, 6, 20, 5])
# U = [None] * N
# U[0] = rnd.normal(key=seed, shape=(d[0], r[0]))
# for i in range(1, N-1):
#     U[i] = rnd.normal(key=seed, shape=(r[i-1], d[i], r[i]))
# U[-1] = rnd.normal(key=seed, shape=(r[-2], d[-1]))

# T = tt_to_tensor(U)
