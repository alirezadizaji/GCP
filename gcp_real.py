
from argparse import ArgumentParser
import os
from sys import argv

import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
import jax

import loss_functions as L
import gcp as G
from data_preprocessing import process_ocn

if __name__ == '__main__':
    url = 'http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt'
    parser = ArgumentParser(description='Data completion task using GCP and GTT on OCN dataset.')
    parser.add_argument('-f', '--filedir', default=url, type=str, help='OCN file directory.')
    parser.add_argument('-r', '--rank', default=5, type=int, nargs="+", help='Factorization rank.')
    parser.add_argument('-m', '--mask-ratio', type=float, nargs="+", help='Mask ratios to apply on the input to run data completion task.')
    parser.add_argument('-j', '--obj-func', type=str, nargs="+")
    parser.add_argument('-d', '--dist', type=str, nargs="+")
    parser.add_argument('-s', '--save-dir', type=str, default='results')
    args = parser.parse_args()
    print("## Args ##")
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    T = process_ocn(args.filedir, num_active_users=200)

    # configs = {
    #     "dist": ['normal', 'poisson_log', 'bernoulli_logit'],
    #     "objective_fun": ["cp", "tt"],
    #     "r": [2, 5, 10],
    #     "mask_ratio": [0.1, 0.5, 0.9],
    # }

    seed = 0
    N = 3
    d = jnp.array(T.shape)
    assert N == len(d)
    num_iters = 1000

    lr = 0.01

    os.makedirs(args.save_dir, exist_ok=True)
    
    for obj_fun in args.obj_func:
        for dist in args.dist:
            for r in args.rank:
                
                r = int(r)
                if obj_fun == "tt":
                    r = [r] * (N - 1)    

                for mask_ratio in args.mask_ratio:
                    init_func = f"init_{obj_fun}"
                    filename = f"{obj_fun}_{dist}_{r}_{mask_ratio}.npz"
                    sd = os.path.join(args.save_dir, filename)
                    
                    print(f"** Start running init_fun: {init_func}, mask_ratio: {mask_ratio}, dist: {dist}, obj_fun: {obj_fun}, r: {r} ...", flush=True)
                    if os.path.exists(sd):
                        print(f"\tSkip; Done before.")
                        continue
                    
                    loss = np.zeros(num_iters)
                
                    # Generate mask
                    mask = 1 - rnd.binomial(n=1, p=float(mask_ratio), size=d).astype(np.float32)

                    # Clear the JIT cache and choose the loss function
                    jax.clear_caches()
                    L.loss_fun = getattr(L, dist)

                    init_fun = getattr(G, init_func)
                    U0 = init_fun(d, r, seed=seed)
                    U, loss_mask, loss_full = G.decompose_lbfgs(T, mask, U0, objective_fun=obj_fun)

                    print(f"\tDone -> loss_mask: {loss_mask}, loss_full: {loss_full}",  flush=True)
                    np.savez_compressed(sd, loss_mask=loss_mask, loss_full=loss_full)
