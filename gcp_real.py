import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
import jax

from sys import argv

import loss_functions as L
import gcp as G
from data_preprocessing import process_ocn

if __name__ == '__main__':
    T = process_ocn(argv[1], num_active_users=200)
    configs = {
        "init": ["init_cp", "init_tt"],
        "dist": ['normal', 'poisson_log', 'bernoulli_logit'],
        "objective_fun": ["cp", "tt"],
        "r": [2, 5, 10],
        "mask_ratio": [0.1, 0.5, 0.9],
    }
    seed = 0
    N = 3
    d = jnp.array(T.shape)
    assert N == len(d)
    num_iters = 1000


    for init_func_name in configs["init"]:
        for  dist in configs["dist"]:
            for  obj_fun in configs["objective_fun"]:
                for r in configs["r"]:
                    for  mask_ratio in configs["mask_ratio"]:
                        loss = np.zeros(num_iters)
                        lr = 0.01
                    
                        # Generate mask
                        mask = 1 - rnd.binomial(n=1, p=mask_ratio, size=d).astype(np.float32)

                        # Clear the JIT cache and choose the loss function
                        dist = 'normal'
                        jax.clear_caches()
                        L.loss_fun = getattr(L, dist)

                        init_fun = getattr(G, init_func_name)
                        U0 = init_fun(d, r, seed=seed)
                        U, loss_mask, loss_full = G.decompose_lbfgs(T, mask, U0, objective_fun=obj_fun)
                        print(f"*** init_fun: {init_func_name}, mask_ratio: {mask_ratio}, dist: {dist}, obj_fun: {obj_fun}, r: {r} -> loss_mask: {loss_mask}, loss_full: {loss_full}",  flush=True)