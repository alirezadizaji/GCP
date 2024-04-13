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



seed = rnd.PRNGKey(0)
N = 4
d = jnp.array([2, 3, 4, 5])
r = 2
U = [rnd.normal(key=seed, shape=(d[i], r)) for i in range(N)]
T = cp_to_tensor(U)



@jit
def objective(U, T):
    return jnp.sum(L.loss_fun(cp_to_tensor(U), T))



seed = rnd.PRNGKey(0)
N = 4
d = jnp.array([2, 3, 4, 5])
assert N == len(d)
r = 2

num_iters = 1000
loss = np.zeros(num_iters)
lr = 0.01

T = rnd.normal(key=seed, shape=d)
U = [rnd.uniform(key=seed, shape=(d[i], r)) for i in range(N)]
for i in tqdm.trange(num_iters):
    # Compute gradient
    gradU = jax.grad(objective)(U, T)

    # Update U using gradient descent
    U = jax.tree_map(lambda x, g: x - lr * g, U, gradU)

    loss[i] = objective(U, T)

fig, ax = plt.subplots()
ax.plot(loss)



N = 4
d = jnp.array([2, 3, 4, 5])
r = jnp.array([2, 6, 20, 5])
U = [None] * N
U[0] = rnd.normal(key=seed, shape=(d[0], r[0]))
for i in range(1, N-1):
    U[i] = rnd.normal(key=seed, shape=(r[i-1], d[i], r[i]))
U[-1] = rnd.normal(key=seed, shape=(r[-2], d[-1]))

T = tt_to_tensor(U)



print()

