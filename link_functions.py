import jax.numpy as jnp
import jax
import jaxopt
from jax import jit

import opt_einsum as oe


@jit
def normal(m):
    mu = m
    return mu

@jit
def gamma(m, k=0.1):
    return m / k

@jit
def rayleigh(m):
    return m * jnp.sqrt(2 / jnp.pi)

@jit
def poisson_linear(m):
    return m

@jit
def poisson_log(m):
    return jnp.exp(m)

@jit
def bernoulli_odds(m):
    return m / (1 + m)

@jit
def bernoulli_logit(m):
    exp_m = jnp.exp(m)
    return exp_m / (1 + exp_m)

@jit
def negative_binomial(m):
    return m / (1 + m)


# Default loss function
loss_fun = normal

