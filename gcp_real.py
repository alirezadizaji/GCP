import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
import jax

from sys import argv

import loss_functions as L
from gcp import decompose_lbfgs, init_cp
from data_preprocessing import process_ocn

if __name__ == '__main__':
    T = process_ocn(argv[1], num_active_users=200)
    seed = 0
    N = 3
    d = jnp.array(T.shape)
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

    # r = jnp.array([2, 6, 20, 5])  # Full-rank
    # r = jnp.array([2, 2, 2, 2])  # Low-rank
    r = 2
    U0 = init_cp(d, r, seed=seed)
    decompose_lbfgs(T, mask, U0, objective_fun='cp')
    print()